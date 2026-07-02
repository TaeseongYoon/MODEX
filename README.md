# Courtroom Analogy: New Perspective on Uncertainty-Aware Classification

Official implementation of **"Courtroom Analogy: New Perspective on Uncertainty-Aware Classification"**, accepted at **ICML 2026**.

This repository contains the implementation of our proposed model, **MoDEX**, and the full experimental pipeline used in the paper: training, testing, misclassification detection, OOD detection, and distribution shift detection.

> Paper: [arXiv:2605.25616](https://arxiv.org/abs/2605.25616) · BibTeX: see [Citation](#citation)

## Overview

The main entry point is `main.py`, which sequentially runs:

1. Training
2. Testing
3. Misclassification detection
4. OOD detection
5. Distribution shift detection (CIFAR-10 only)

## Code Structure

* **main.py** — entry point; orchestrates training, evaluation, misclassification detection, OOD detection, and distribution shift detection
* **datasets.py** — dataset downloading, loading, and preprocessing utilities
* **models.py** — model architecture definitions, including MoDEX
* **train.py** — training and testing routines
* **uq.py** — uncertainty quantification utilities: uncertainty measures, confidence calibration (misclassification detection), OOD detection, and distribution shift detection

## Installation

```bash
pip install torch torchvision numpy scikit-learn tqdm pillow
pip install ddu-dirty-mnist  # only needed for the noisy MNIST setting (--ID_dataset MNIST)
```

## Data Preparation

Most datasets are downloaded automatically on first run via `torchvision` (`download=True`) into `./data`: MNIST, FashionMNIST, KMNIST, CIFAR-10, CIFAR-100, SVHN. AmbiguousMNIST (used for the noisy MNIST setting via `ddu_dirty_mnist.DirtyMNIST`) also downloads automatically.

The following are **not** auto-downloaded and must be placed manually under `./data`:

| Path | Used for |
|---|---|
| `data/tiny-imagenet-200/test/` | OOD detection when `--ID_dataset CIFAR-100` |
| `data/cifar10_c/` | Distribution shift detection when `--ID_dataset CIFAR-10` (CIFAR-10-C corruption `.npy` files) |
| `data/mnist_c/` | Distribution shift detection for the MNIST setting (MNIST-C corruption folders) |

## Usage

```bash
# Classical setting
python main.py

# Long-tailed setting
python main.py --imbalance_factor 0.01   # or 0.1

# Long-tailed and noisy setting
python main.py --ID_dataset MNIST
```

**Note**: For each experimental setting, hyperparameter configurations should be set according to those reported in the paper to reproduce the best-performing results.

## Environment

- GPU: NVIDIA GeForce RTX 4060
- Python 3.13.11
- PyTorch 2.9.1+cu126
- Torchvision 0.24.1+cu126
- NumPy 2.3.5
- tqdm 4.67.1

## Citation

If you find this work useful, please cite:

```bibtex
@article{yoon2026courtroom,
  title={Courtroom Analogy: New Perspective on Uncertainty-Aware Classification},
  author={Yoon, Taeseong and Kim, Heeyoung},
  journal={arXiv preprint arXiv:2605.25616},
  year={2026}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
