import os
import torch
import json
import argparse
import numpy as np
import random

import models, datasets, train, uq
from models import *
from datasets import *
from train import *
from uq import *


def get_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--dropout_rate", type=float, default=0)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--step_size", type=int, default=100)
    parser.add_argument("--activation", type=str, default="exp")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--tau_guidance", type=float, default=1)
    parser.add_argument("--result_dir", type=str, default="saved_results_modex")
    parser.add_argument("--model_dir", type=str, default="saved_models_modex")
    parser.add_argument("--ID_dataset", type=str, default="CIFAR-10")
    parser.add_argument("--imbalance_factor", type=float, default=0,
                         help="Imbalance factor (e.g., 0 = balanced, 0.01 = heavily imbalanced)")

    return parser.parse_args()


def build_config(args):
    if args.ID_dataset == "MNIST":
        num_classes = 10
        noise = True

    elif args.ID_dataset == "CIFAR-10":
        num_classes = 10
        noise = False

    elif args.ID_dataset == "CIFAR-100":
        num_classes = 100
        noise = False

    else:
        raise ValueError(f"Unknown dataset: {args.ID_dataset}")

    return {
        "ID_dataset": args.ID_dataset,
        "batch_size": 64,
        "val_size": 0.05,
        "val_seed": 12345,
        "imbalance_factor": args.imbalance_factor,
        "noise": noise,
        "dropout_rate": args.dropout_rate,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "tau_guidance": args.tau_guidance,
        "step_size": args.step_size,
        "activation": args.activation,
        "scheduler_type": "step",
        "device": args.device if torch.cuda.is_available() else "cpu",
        "result_dir": args.result_dir,
        "model_dir": args.model_dir,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_classes": num_classes,
        "spect_norm": True,
    }


def run_pipeline(config):
    os.makedirs(config["model_dir"], exist_ok=True)
    os.makedirs(config["result_dir"], exist_ok=True)

    trainloader, validloader, testloader, ood_loader1, ood_loader2 = load_datasets(config["ID_dataset"], config["batch_size"], config["val_size"],config["imbalance_factor"], config["noise"])

    model = MODEX(config["ID_dataset"], config["dropout_rate"], config["spect_norm"],config["device"], config["hidden_dim"], config["num_layers"], config["activation"])

    train(model, config["learning_rate"], config["weight_decay"], config["step_size"], config["num_epochs"], config["tau_guidance"], trainloader, validloader, config["num_classes"], config["device"])

    test_acc = test(model, testloader, config["device"])
    conf_auroc, conf_aupr, brier = conf_calibration(model, testloader, config["device"])
    ood_auroc, ood_aupr = ood_detection(model, testloader, ood_loader1, ood_loader2, config["device"])

    if config["ID_dataset"] == "CIFAR-10" and config["imbalance_factor"] == 0:
        dist_auroc, dist_aupr = dist_shift_detection(config["ID_dataset"], model, testloader, config["device"])
    else:
        dist_auroc, dist_aupr = 0, 0

    result = {"Test Accuracy": test_acc,"CONF AUROC": conf_auroc,"CONF AUPR": conf_aupr,"BRIER": brier,
              "OOD AUROC": ood_auroc,"OOD AUPR": ood_aupr, "DIST AUROC": dist_auroc, "DIST AUPR": dist_aupr}

    return model, result


def save_outputs(model, result, config):
    rand_id = random.randint(10000, 99999)
    suffix = (
        f"_{config['ID_dataset'].lower()}"
        f"_imb{config['imbalance_factor']}"
        f"_lr{config['learning_rate']}"
        f"_wd{config['weight_decay']}"
        f"_hid{config['hidden_dim']}"
        f"_layers{config['num_layers']}"
        f"_id{rand_id}"
    )

    result_filename = f"{suffix}.json"
    result_path = os.path.join(config["result_dir"], result_filename)
    with open(result_path, "w") as f:
        json.dump(to_serializable(result), f, indent=2)
    print(f"[INFO] Saved result to {result_path}")

    model_filename = f"{suffix}.pt"
    model_path = os.path.join(config["model_dir"], model_filename)
    torch.save(model.state_dict(), model_path)
    print(f"[INFO] Saved model to {model_path}")


def main():
    args = get_args()
    config = build_config(args)

    model, result = run_pipeline(config)
    print(result)

    save_outputs(model, result, config)


if __name__ == "__main__":
    main()
