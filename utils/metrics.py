import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_metrics(cm, class_names, eps=1e-8):
    # Per-class accuracy
    diag = cm.diagonal()
    all_acc = diag.sum() / (cm.sum() + eps)
    per_class_acc = diag / (cm.sum(axis=1) + eps)
    macc = per_class_acc.mean()

    # Macro F1
    precision = np.diag(cm) / (np.sum(cm, axis=0) + eps)
    recall = np.diag(cm) / (np.sum(cm, axis=1) + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    macro_f1 = np.mean(f1)
    metrics = {"all_acc": all_acc, "macc": macc, "macro_f1": macro_f1}

    # Per-class metrics
    for idx, name in enumerate(class_names):
        metrics[f"acc_{name}"] = per_class_acc[idx]
        metrics[f"f1_{name}"] = per_class_acc[idx]

    return metrics


def viz_conf_mat(cm, class_names, output_path):
    plt.figure(figsize=(8, 6))
    cm = np.astype(cm, np.int64)
    sns.heatmap(cm, annot=True, fmt="d", cmap="magma_r", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted", fontsize=16)
    plt.ylabel("True", fontsize=16)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.close()
    plt.cla()