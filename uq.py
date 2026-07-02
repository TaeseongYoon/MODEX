import os

import torch
import torch.nn.functional as F
import numpy as np
import sklearn.metrics
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset
from PIL import Image


# ---------------- Utils ----------------
def check(x):
    if isinstance(x, np.ndarray):
        x_tensor = torch.tensor(x)
    else:
        x_tensor = x
    nan = torch.sum(torch.isnan(x_tensor))
    inf = torch.sum(torch.isinf(x_tensor))
    if (inf + nan) != 0:
        x_tensor = torch.nan_to_num(x_tensor)
    return x_tensor


def convert_to_native(val):
    if isinstance(val, np.generic):
        return val.item()
    elif isinstance(val, torch.Tensor):
        return val.item() if val.numel() == 1 else val.tolist()
    return val


def to_serializable(obj):
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    else:
        return obj


# ---------------- Moments ----------------
def compute_moments(alpha, w, tau):
    alpha0 = alpha.sum(dim=1, keepdim=True)
    k1 = (w / (alpha0 + tau)).sum(dim=1, keepdim=True)
    k2 = (w / ((alpha0 + tau) * (alpha0 + tau + 1))).sum(dim=1, keepdim=True)

    term1 = alpha**2 * (k2 - k1**2)
    term2 = w * tau * (2 * alpha + tau + 1) / ((alpha0 + tau) * (alpha0 + tau + 1))
    term3 = alpha * k2
    term4 = (w**2 * tau**2) / ((alpha0 + tau)**2)
    term5 = (2 * alpha * w * tau * k1) / (alpha0 + tau)

    mu = alpha * k1 + tau * (w / (alpha0 + tau))
    var = term1 + term2 + term3 - term4 - term5
    return mu, var


# ---------------- Uncertainty Measures ----------------
def AU(alpha, w, tau):
    mu, _ = compute_moments(alpha, w, tau)
    mu_clamped = torch.clamp(mu, 1e-6, 1.0)
    entropy = -torch.sum(mu_clamped * torch.log(mu_clamped), dim=1)
    return entropy


def EU(alpha, w, tau):
    _, var = compute_moments(alpha, w, tau)
    return var.sum(1)


# ---------------- Evaluation ----------------
def uq(model, loader, device):
    model.eval()
    au_list, eu_list = [], []
    with torch.no_grad():
        for x, _ in tqdm(loader):
            alpha, w, tau = model(x.to(device))
            au_list.append(AU(alpha, w, tau).cpu())
            eu_list.append(EU(alpha, w, tau).cpu())
    return (torch.cat(au_list).numpy(),
            torch.cat(eu_list).numpy())


def auroc_aupr(unc_id, unc_ood):
    unc_id, unc_ood = check(unc_id), check(unc_ood)

    bin_labels = np.concatenate([np.ones(unc_id.shape[0]), np.zeros(unc_ood.shape[0])])
    scores = -np.concatenate((unc_id, unc_ood))

    auroc = sklearn.metrics.roc_auc_score(bin_labels, scores)
    aupr = sklearn.metrics.average_precision_score(bin_labels, scores)
    return auroc, aupr


def conf_calibration(model, testloader, device):
    model.eval()
    CORRECT, AU_scores, EU_scores, brier_scores = [], [], [], []
    with torch.no_grad():
        for x, y in testloader:
            alpha, w, tau = model(x.to(device))
            mu, _ = compute_moments(alpha, w, tau)

            y_pred = mu.argmax(1).cpu().numpy()
            correct = (y_pred == y.cpu().numpy()).astype(int)

            y_oh = F.one_hot(y, num_classes=w.shape[1]).to(device)
            brier_score_batch = torch.mean((y_oh - mu) ** 2, dim=1)

            AU_scores.append(AU(alpha, w, tau).cpu())
            EU_scores.append(EU(alpha, w, tau).cpu())
            brier_scores.extend(brier_score_batch.cpu().numpy())
            CORRECT.append(correct)

    CORRECT = np.concatenate(CORRECT)
    AU_scores = torch.cat(AU_scores).numpy()
    EU_scores = torch.cat(EU_scores).numpy()
    BRIER = np.mean(brier_scores)

    alea_id, epis_id = -check(AU_scores), -check(EU_scores)

    AUROC = {
        "AU": sklearn.metrics.roc_auc_score(CORRECT, alea_id),
        "EU": sklearn.metrics.roc_auc_score(CORRECT, epis_id),
    }
    AUPR = {
        "AU": sklearn.metrics.average_precision_score(CORRECT, alea_id),
        "EU": sklearn.metrics.average_precision_score(CORRECT, epis_id),
    }
    return AUROC, AUPR, BRIER


def ood_detection(model, testloader, ood_loader1, ood_loader2, device):
    model.eval()

    alea_id, epis_id = uq(model, testloader, device)
    alea_ood1, epis_ood1 = uq(model, ood_loader1, device)
    alea_ood2, epis_ood2 = uq(model, ood_loader2, device)

    def eval_metrics(id_vals, ood_vals):
        return auroc_aupr(id_vals, ood_vals)

    AUROC = [
        {"AU": eval_metrics(alea_id, alea_ood1)[0],
         "EU": eval_metrics(epis_id, epis_ood1)[0]},
        {"AU": eval_metrics(alea_id, alea_ood2)[0],
         "EU": eval_metrics(epis_id, epis_ood2)[0]}
    ]
    AUPR = [
        {"AU": eval_metrics(alea_id, alea_ood1)[1],
         "EU": eval_metrics(epis_id, epis_ood1)[1]},
        {"AU": eval_metrics(alea_id, alea_ood2)[1],
         "EU": eval_metrics(epis_id, epis_ood2)[1]}
    ]
    return AUROC, AUPR


# ---------------- Distribution Shift Detection ----------------
def dist_shift_detection_mnist(model,testloader,device,base_path="data/mnist_c",batch_size=64):

    corruptions = [ "shot_noise", "impulse_noise", "glass_blur", "motion_blur", "shear", "scale", "rotate", "brightness", "translate", "stripe", "fog", "spatter","dotted_line", "zigzag", "canny_edges"]
    model.eval()
    au_id, eu_id = uq(model, testloader, device)

    AUROC_AU, AUPR_AU = [], []
    AUROC_EU, AUPR_EU = [], []

    normalize = transforms.Normalize((0.5,), (0.5,))

    for ctype in corruptions:
        img_path = os.path.join(base_path, ctype, "test_images.npy")
        lab_path = os.path.join(base_path, ctype, "test_labels.npy")
        if (not os.path.exists(img_path)) or (not os.path.exists(lab_path)):
            continue

        data = np.load(img_path)
        labels = np.load(lab_path)

        data_t = torch.tensor(data)
        if data_t.ndim == 3:
            data_t = data_t.unsqueeze(1)
        elif data_t.ndim == 4 and data_t.shape[-1] == 1:
            data_t = data_t.permute(0, 3, 1, 2)
        data_t = data_t.float() / 255.0
        data_t = normalize(data_t)

        labels_t = torch.tensor(labels).long()
        loader = torch.utils.data.DataLoader(TensorDataset(data_t, labels_t),batch_size=batch_size,shuffle=False)

        au_ood, eu_ood = uq(model, loader, device)

        auroc_au, aupr_au = auroc_aupr(au_id, au_ood)
        AUROC_AU.append(auroc_au)
        AUPR_AU.append(aupr_au)

        auroc_eu, aupr_eu = auroc_aupr(eu_id, eu_ood)
        AUROC_EU.append(auroc_eu)
        AUPR_EU.append(aupr_eu)

    AUROC = {"AU": float(np.mean(AUROC_AU)) if AUROC_AU else float("nan"),
             "EU": float(np.mean(AUROC_EU)) if AUROC_EU else float("nan")}
    AUPR  = {"AU": float(np.mean(AUPR_AU)) if AUPR_AU else float("nan"),
             "EU": float(np.mean(AUPR_EU)) if AUPR_EU else float("nan")}

    return AUROC, AUPR


class CIFARCorruption(torch.utils.data.Dataset):

    def __init__(self, data_uint8, transform):
        self.data = data_uint8
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        x = self.transform(img)
        return x, 0


def dist_shift_detection_cifar(
    ID_dataset, model, testloader, device,
    base_path="data/cifar10_c", batch_size=128,
    reduce="severity_mean"   # "none" | "severity_mean"
):
    CORRUPTIONS = [
        "gaussian_noise","shot_noise","impulse_noise","defocus_blur","glass_blur","motion_blur","zoom_blur",
        "snow","frost","fog","brightness","contrast","elastic_transform","pixelate","jpeg_compression",
        "speckle_noise","gaussian_blur","spatter","saturate"
    ]

    if base_path is None:
        base_path = "data/CIFAR-10-C" if ID_dataset == "CIFAR-10" else "data/CIFAR-100-C"

    if ID_dataset == "CIFAR-10":
        mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    else:
        mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    model.eval()
    au_id, eu_id = uq(model, testloader, device)

    AUROC_raw, AUPR_raw = {}, {}

    for corruption in CORRUPTIONS:
        path = os.path.join(base_path, f"{corruption}.npy")
        if not os.path.exists(path):
            continue

        data = np.load(path)
        dataset = CIFARCorruption(data, transform)

        AUROC_raw[corruption], AUPR_raw[corruption] = {}, {}

        for severity in range(1, 6):
            start = 10000 * (severity - 1)
            end = 10000 * severity
            subset = torch.utils.data.Subset(dataset, list(range(start, end)))
            loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False)

            au_ood, eu_ood = uq(model, loader, device)

            auroc_au, aupr_au = auroc_aupr(au_id, au_ood)
            auroc_eu, aupr_eu = auroc_aupr(eu_id, eu_ood)

            AUROC_raw[corruption][severity] = {"AU": float(auroc_au), "EU": float(auroc_eu)}
            AUPR_raw[corruption][severity]  = {"AU": float(aupr_au),  "EU": float(aupr_eu)}

    if reduce == "none":
        return AUROC_raw, AUPR_raw

    AUROC_sev_mean, AUPR_sev_mean = {}, {}
    for severity in range(1, 6):
        au_list, eu_list = [], []
        for corr in AUROC_raw:
            if severity in AUROC_raw[corr]:
                au_list.append(AUROC_raw[corr][severity]["AU"])
                eu_list.append(AUROC_raw[corr][severity]["EU"])
        AUROC_sev_mean[severity] = {
            "AU": float(np.mean(au_list)) if au_list else float("nan"),
            "EU": float(np.mean(eu_list)) if eu_list else float("nan"),
        }

        au_list, eu_list = [], []
        for corr in AUPR_raw:
            if severity in AUPR_raw[corr]:
                au_list.append(AUPR_raw[corr][severity]["AU"])
                eu_list.append(AUPR_raw[corr][severity]["EU"])
        AUPR_sev_mean[severity] = {
            "AU": float(np.mean(au_list)) if au_list else float("nan"),
            "EU": float(np.mean(eu_list)) if eu_list else float("nan"),
        }

    return AUROC_sev_mean, AUPR_sev_mean


def dist_shift_detection(ID_dataset, model, testloader, device, **kwargs):
    if ID_dataset == "MNIST":
        return dist_shift_detection_mnist(model, testloader, device, **kwargs)
    else:
        return dist_shift_detection_cifar(ID_dataset, model, testloader, device, **kwargs)

