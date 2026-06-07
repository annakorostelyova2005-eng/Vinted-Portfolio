# helpers.py

import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np

# -----------------
# Helper: batchify
# -----------------


def batchify(x, y, batch_size=32):
    for i in range(0, x.shape[0], batch_size):
        yield x[i:i+batch_size], y[i:i+batch_size]


# ---------------------------------
# Helper: compute accuracy
# ---------------------------------


def batch_accuracy_from_logits(logits, targets, num_classes=None):
    """
    logits:  (B, 1) for binary or (B, C) for multi-class
    targets: (B,) or (B, 1), values in {0,1,...,C-1}
    """
    # make sure targets are shape (B,)
    targets = targets.view(-1)

    if num_classes is not None and num_classes > 2:
        # multi-class: pick argmax
        preds = logits.argmax(dim=1)          # (B,)
    else:
        # binary: logits -> class via > 0
        preds = (logits.squeeze(-1) > 0).long()

    correct = (preds == targets.long()).sum().item()
    total = targets.numel()

    return correct, total


# ---------------------------------
# Helper: log experiment
# ---------------------------------


def log_experiment(model, csv_path: str = "experiments_log.csv"):
    """
    Append a single experiment summary row to a CSV file.

    Expects the model to have:
      - model.name                 (string)
      - model.train_losses         (list of floats)
      - model.val_losses           (list of floats)
    Optionally (for classification):
      - model.train_accuracies     (list of floats in [0,1])
      - model.val_accuracies       (list of floats in [0,1])
    Optionally (metadata, all optional):
      - model.task                 ("regression" or "classification")
      - model.num_classes          (int)
      - model.dataset              (string)
    """

# --- attributes ---
    name = getattr(model, "name", "unnamed_model")
    task = getattr(model, "task", "unknown")
    num_classes = getattr(model, "num_classes", None)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dataset = getattr(model, "dataset", None)

    train_losses = getattr(model, "train_losses", None)
    val_losses = getattr(model, "val_losses", None)

    if not train_losses or not val_losses:
        raise ValueError(f"Model {name} has no train/val losses attached.")

    n_epochs = len(train_losses)

    initial_train_loss = round(train_losses[0], 4)
    final_train_loss = round(train_losses[-1], 4)
    initial_val_loss = round(val_losses[0], 4)
    final_val_loss = round(val_losses[-1], 4)

    train_accs = getattr(model, "train_accuracies", None)
    val_accs = getattr(model, "val_accuracies", None)

    if task == "classification" and train_accs is not None and len(train_accs) > 0:
        initial_train_acc = round(train_accs[0], 4)
        final_train_acc = round(train_accs[-1], 4)
        initial_val_acc = round(val_accs[0], 4)
        final_val_acc = round(val_accs[-1], 4)
    else:
        initial_train_acc = np.nan
        final_train_acc = np.nan
        initial_val_acc = np.nan
        final_val_acc = np.nan

    # --- overfitting flag (simple heuristic) ---
    if final_train_loss > 0:
        overfitting = final_val_loss > final_train_loss * 1.3
    else:
        overfitting = False

    # --- underfitting flag (small improvement on train loss) ---
    # Measure how much train loss improved over training
    if initial_train_loss > 0:
        rel_improvement = (initial_train_loss - final_train_loss) / initial_train_loss
    else:
        rel_improvement = 0.0

    # Underfitting if:
    #  - not overfitting, AND
    #  - training loss improved less than 20% (tweak threshold if you like)
    underfitting = (not overfitting) and (rel_improvement < 0.2)

    # --- near-random flag (classification only, if we know #classes & val_acc) ---
    if task == "classification" and num_classes is not None and not pd.isna(final_val_acc):
        random_acc = 1.0 / num_classes
        acc_tol = 0.05  # +/- 5% absolute accuracy
        acc_near_random = abs(final_val_acc - random_acc) < acc_tol

        if num_classes > 2:
            random_loss = np.log(num_classes)
        else:
            random_loss = np.log(2.0)

        loss_tol = 0.2  # absolute tolerance
        loss_near_random = abs(final_val_loss - random_loss) < loss_tol

        near_random = acc_near_random and loss_near_random
    else:
        near_random = False

    row = {
        "model_name": name,
        "task": task,
        "num_params": num_params,
        "dataset": dataset if dataset is not None else "-",
        "num_classes": num_classes if num_classes is not None else np.nan,
        "num_epochs": n_epochs,
        "initial_train_loss": initial_train_loss,
        "final_train_loss": final_train_loss,
        "initial_val_loss": initial_val_loss,
        "final_val_loss": final_val_loss,
        "initial_train_acc": initial_train_acc,
        "final_train_acc": final_train_acc,
        "initial_val_acc": initial_val_acc,
        "final_val_acc": final_val_acc,
        "overfitting": overfitting,
        "underfitting": underfitting,
        "near_random": near_random,
    }

    path = Path(csv_path)
    if path.exists():
        df = pd.read_csv(path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(path, index=False)
    print("Logging finished")


# --------------------------------
# Helper: display experiment logs
# --------------------------------


def display_logs(dataset, epochs, task="clasification"):
    df = pd.read_csv('experiments_log.csv')

    if epochs == 'all':
        df = df[df['dataset'] == dataset]
    else:
        df = df[(df['dataset'] == dataset) & (df['num_epochs'] == epochs)]

    if task == 'regression':
        df = df[["model_name",
                 "num_params",
                 "num_epochs",
                 "initial_train_loss",
                 "final_train_loss",
                 "initial_val_loss",
                 "final_val_loss",
                 "overfitting",
                 "underfitting"]]
    return df

# ---------------------------------
# Helper: losses plotting funciton
# ---------------------------------


def compare_losses(*models):
    fig, ax = plt.subplots(len(models), 1, figsize=(10, 2 * len(models)), sharex=True)
    fig.suptitle(f"Train and validation losses for {len(models[0].train_losses)} epochs", y=0.98)
    
    for n, model in enumerate(models):
        train_loss = model.train_losses
        val_loss = model.val_losses
        name = model.name
        epochs = range(len(train_loss))

        ax[n].plot(epochs, train_loss, label="train", color='red')
        ax[n].plot(epochs, val_loss, label="val",   color='green')
        ax[n].set_title(f"{name}")
        ax[n].legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
