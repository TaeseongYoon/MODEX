from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


# ---------------- Long-tail utilities ----------------
def create_long_tail_distribution(dataset, imbalance_factor):
    targets = np.array(dataset.targets)
    class_counts = Counter(targets)
    num_classes = len(class_counts)
    max_samples = max(class_counts.values())
    class_indices = []
    for cls in range(num_classes):
        num_samples = int(max_samples * (imbalance_factor ** (cls / (num_classes - 1.0))))
        indices = np.where(targets == cls)[0]
        np.random.shuffle(indices)
        class_indices.extend(indices[:num_samples])
    return class_indices


def create_balanced_validation_set(val_subset, train_dataset):
    val_indices = np.array(val_subset.indices)
    val_labels = np.array([train_dataset.targets[i] for i in val_indices])
    _, val_indices_bal = train_test_split(val_indices, test_size=0.5, stratify=val_labels)
    return Subset(train_dataset, val_indices_bal)


# ---------------- Main wrapper ----------------
def load_datasets(ID_dataset, batch_size, val_size=0.05, imbalance_factor=0.0, noise=False):
    name = ID_dataset.upper()
    if name in ["MNIST", "DMNIST"]:
        return dataloaders_mnist(batch_size, val_size, imbalance_factor, noise)
    elif name == "CIFAR-10":
        return dataloaders_cifar10(batch_size, val_size, imbalance_factor, noise)
    elif name == "CIFAR-100":
        return dataloaders_cifar100(batch_size, val_size, imbalance_factor, noise)
    else:
        raise ValueError(f"Unsupported dataset: {ID_dataset}")


# ---------------- Dataset-specific loaders ----------------
def dataloaders_mnist(batch_size, val_size, imbalance_factor=0.0, noise=False, ambig_ratio=0.5):
    root = "./data"
    num_workers = 4
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # ---------------- OOD datasets (always loaded) ----------------
    fmnist_dataset = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    kmnist_dataset = datasets.KMNIST(root=root, train=False, download=True, transform=transform)

    if not noise:
        # ---------------- Clean MNIST ----------------
        base_train = datasets.MNIST(root=root, train=True, download=True, transform=transform)
        base_test  = datasets.MNIST(root=root, train=False, download=True, transform=transform)

        if imbalance_factor == 0:
            train_size = int((1 - val_size) * len(base_train))
            val_size_actual = len(base_train) - train_size
            train_subset, val_subset = random_split(base_train, [train_size, val_size_actual])
            val_subset = create_balanced_validation_set(val_subset, base_train)

        else:
            long_tail_indices = create_long_tail_distribution(base_train, imbalance_factor)
            lt_dataset = Subset(base_train, long_tail_indices)

            train_size = int((1 - val_size) * len(lt_dataset))
            val_size_actual = len(lt_dataset) - train_size
            train_subset, val_subset = random_split(lt_dataset, [train_size, val_size_actual])

            val_subset = create_balanced_validation_set(val_subset, base_train)

        trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
        validloader = DataLoader(val_subset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
        testloader  = DataLoader(base_test,    batch_size=batch_size, shuffle=False, num_workers=num_workers)

    else:
        from ddu_dirty_mnist import DirtyMNIST
        dirty_train = DirtyMNIST(root=root, train=True, download=True)
        dirty_test  = DirtyMNIST(root=root, train=False, download=True)

        if isinstance(dirty_train, torch.utils.data.ConcatDataset) and len(dirty_train.datasets) >= 2:
            clean_len = len(dirty_train.datasets[0])
            total_len = len(dirty_train)
            ambig_len = total_len - clean_len
        else:
            total_len = len(dirty_train)
            clean_len = total_len // 2
            ambig_len = total_len - clean_len

        clean_all = np.arange(0, clean_len)
        ambig_all = np.arange(clean_len, clean_len + ambig_len)

        if imbalance_factor > 0:
            print(f"[INFO] Applying imbalance_factor={imbalance_factor} to DirtyMNIST")

            if not hasattr(dirty_train, "targets"):
                if isinstance(dirty_train, torch.utils.data.ConcatDataset):
                    all_targets = []
                    for sub in dirty_train.datasets:
                        if hasattr(sub, "targets"):
                            all_targets.extend(list(sub.targets))
                        elif hasattr(sub, "labels"):
                            all_targets.extend(list(sub.labels))
                        else:
                            all_targets.extend([sub[i][1] for i in range(len(sub))])
                    dirty_train.targets = all_targets
                else:
                    dirty_train.targets = [dirty_train[i][1] for i in range(len(dirty_train))]

            pool = np.array(create_long_tail_distribution(dirty_train, imbalance_factor), dtype=int)
            clean_pool = pool[pool < clean_len]
            ambig_pool = pool[pool >= clean_len]
        else:
            clean_pool = clean_all
            ambig_pool = ambig_all


        if isinstance(dirty_train, torch.utils.data.ConcatDataset) and len(dirty_train.datasets) >= 1:
            clean_ds = dirty_train.datasets[0]
            if hasattr(clean_ds, "targets"):
                clean_targets = np.array(clean_ds.targets)
            elif hasattr(clean_ds, "labels"):
                clean_targets = np.array(clean_ds.labels)
            else:
                clean_targets = np.array([clean_ds[i][1] for i in range(len(clean_ds))])
        else:
            clean_targets = np.array([dirty_train[i][1] for i in range(clean_len)])

        clean_pool = np.array(clean_pool, dtype=int)
        clean_pool_labels = clean_targets[clean_pool]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
        train_rel, val_rel = next(sss.split(clean_pool.reshape(-1, 1), clean_pool_labels))

        clean_train_pool = clean_pool[train_rel]
        clean_val_idx    = clean_pool[val_rel]


        r = float(ambig_ratio)
        r = min(max(r, 0.0), 1.0)

        GROUP = 10
        use_groups = (ambig_len % GROUP == 0)

        ambig_pool = np.array(ambig_pool, dtype=int)

        if use_groups:
            ambig_offsets = ambig_pool - clean_len
            group_id = ambig_offsets // GROUP
            uniq, cnt = np.unique(group_id, return_counts=True)
            full_groups = uniq[cnt == GROUP]
            max_ambig_train = len(full_groups) * GROUP
        else:
            full_groups = None
            max_ambig_train = len(ambig_pool)
            GROUP = 1

        max_clean_train = len(clean_train_pool)

        def choose_counts(max_clean: int, max_ambig: int, r: float, g: int):
            if r <= 0.0:
                return max_clean, 0
            if r >= 1.0:
                return 0, (max_ambig // g) * g

            T_max = min(max_clean / (1.0 - r), max_ambig / r)
            a = int(np.floor((r * T_max) / g) * g)
            c = int(np.floor(((1.0 - r) / r) * a))
            c = min(c, max_clean)

            a = int(np.floor((r / (1.0 - r)) * c / g) * g)
            a = min(a, (max_ambig // g) * g)
            return c, a

        clean_count, ambig_count = choose_counts(max_clean_train, max_ambig_train, r, GROUP)

        clean_train_sel = np.random.permutation(clean_train_pool)[:clean_count]

        if GROUP == 1:
            ambig_train_sel = np.random.permutation(ambig_pool)[:ambig_count]
        else:
            n_groups = ambig_count // GROUP
            sel_groups = np.random.permutation(full_groups)[:n_groups]

            ambig_train_sel = np.concatenate([
                clean_len + gid * GROUP + np.arange(GROUP) for gid in sel_groups
            ], axis=0)


        train_indices = np.concatenate([clean_train_sel, ambig_train_sel], axis=0)
        val_indices   = np.array(clean_val_idx, dtype=int)

        train_subset = Subset(dirty_train, train_indices.tolist())
        val_subset   = Subset(dirty_train, val_indices.tolist())

        train_subset.clean_count = int(clean_count)
        train_subset.ambig_count = int(ambig_count)
        train_subset.clean_boundary = int(clean_count)

        trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
        validloader = DataLoader(val_subset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
        testloader  = DataLoader(dirty_test,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    fmnist_loader = DataLoader(fmnist_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    kmnist_loader = DataLoader(kmnist_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, validloader, testloader, fmnist_loader, kmnist_loader


def dataloaders_cifar10(batch_size, val_size, imbalance_factor=0.0, noise=False):
    root = "./data"
    num_workers = 4
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize,])
    test_transform = transforms.Compose([transforms.ToTensor(),normalize,])

    base_train = datasets.CIFAR10(root=root, train=True, download=True, transform=train_transform)
    base_valid = datasets.CIFAR10(root=root, train=True, download=True, transform=test_transform)
    base_test = datasets.CIFAR10(root=root, train=False, download=True, transform=test_transform)

    if imbalance_factor == 0:
        num_train = len(base_train)
        indices = np.random.permutation(num_train)
        split = int(np.floor(val_size * num_train))
        train_indices, val_indices = indices[split:], indices[:split]
        train_subset = Subset(base_train, train_indices)
        val_subset = Subset(base_valid, val_indices)
    else:
        long_tail_indices = create_long_tail_distribution(base_train, imbalance_factor)
        lt_dataset = Subset(base_train, long_tail_indices)
        train_size = int((1 - val_size) * len(lt_dataset))
        val_size_actual = len(lt_dataset) - train_size
        train_subset, val_subset = random_split(lt_dataset, [train_size, val_size_actual])

    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader = DataLoader(base_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    svhn_dataset = datasets.SVHN(root=root, split="test", download=True, transform=test_transform)
    cifar100_dataset = datasets.CIFAR100(root=root, train=False, download=True, transform=test_transform)
    svhn_loader = DataLoader(svhn_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    cifar100_loader = DataLoader(cifar100_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, validloader, testloader, svhn_loader, cifar100_loader


class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def dataloaders_cifar100(batch_size, val_size, imbalance_factor=0.0, noise=False):
    root = "./data"
    num_workers = 4
    mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    normalize = transforms.Normalize(mean, std)

    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(p=0.5),transforms.ToTensor(),normalize])
    eval_transform = transforms.Compose([transforms.ToTensor(), normalize])
    ood_transform = transforms.Compose([transforms.Resize(32),  transforms.ToTensor(),normalize])

    base_dataset = datasets.CIFAR100(root=root, train=True, download=True, transform=None)
    base_targets = np.array(base_dataset.targets)

    if imbalance_factor and imbalance_factor > 0:
        lt_indices = np.array(create_long_tail_distribution(base_dataset, imbalance_factor))
        indices_pool, labels_pool = lt_indices, base_targets[lt_indices]
    else:
        indices_pool, labels_pool = np.arange(len(base_dataset)), base_targets

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
    split_train_idx, split_val_idx = next(sss.split(indices_pool, labels_pool))

    train_indices = indices_pool[split_train_idx]
    val_indices = indices_pool[split_val_idx]

    train_subset = TransformedSubset(Subset(base_dataset, train_indices.tolist()), transform=train_transform)
    val_subset = TransformedSubset(Subset(base_dataset, val_indices.tolist()), transform=eval_transform)

    test_dataset = datasets.CIFAR100(root=root, train=False, download=True, transform=eval_transform)

    svhn_dataset = datasets.SVHN(root=root, split="test", download=True, transform=eval_transform)
    tin_dataset = datasets.ImageFolder(root="./data/tiny-imagenet-200/test", transform=ood_transform)

    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    svhn_loader = DataLoader(svhn_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    tin_loader = DataLoader(tin_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, validloader, testloader, svhn_loader, tin_loader

