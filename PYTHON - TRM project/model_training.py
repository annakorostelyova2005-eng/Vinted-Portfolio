# model_training.py

import torch
import torch.nn.functional as F
import torch.optim as optim

from helpers import batchify, batch_accuracy_from_logits

from trm import TRMModel as DefaultTRMModel
from basic_mlp import MLPBaseline


# ---------------------------------------------------------------------
# 1) TRM training: supports regression and classification
# ---------------------------------------------------------------------

def train_trm(
    trm_cfg,
    x_train,
    y_train,
    x_val,
    y_val,
    num_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    device=None,
    task: str = "regression",      # "regression" or "classification"
    num_classes: int = None,       # needed for multi-class classification
    model_cls=DefaultTRMModel,
):
    """
    Train a TRMModel with the given config and data.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move full datasets to device once (safe even if already on that device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)

    # Decide output dimension
    if task == "regression":
        out_dim = 1
    else:
        out_dim = 1 if (num_classes is None or num_classes <= 2) else num_classes

    # Use model_cls, so you can pass TRMRecOnlyModel etc.
    model = model_cls(trm_cfg, out_dim=out_dim)
    opt = optim.AdamW(model.parameters(), lr=lr)

    model.to(device)
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # ---------- TRAIN ----------
        model.train()
        total_loss = 0.0
        train_n = 0
        correct_train = 0
        total_train = 0

        for xb, yb in batchify(x_train, y_train, batch_size=batch_size):
            # xb, yb are already on device
            B = xb.size(0)

            # 1) Initialize outer state (ACT-style)
            state = model.initial_state(batch_size=B, device=device)

            # 2) Run up to halt_max_steps supervision steps
            outputs = None
            for step in range(trm_cfg.halt_max_steps):
                state, outputs = model(state, xb)
                if bool(state.halted.all()):
                    break

            y_out = outputs["y_pred"]  # (B, 1) or (B, C)

            if task == "regression":
                y_pred = y_out.squeeze(-1)
                if yb.dim() > 1:
                    yb = yb.squeeze(-1)
                loss = F.mse_loss(y_pred, yb)

            elif task == "classification":
                if num_classes is not None and num_classes > 2:
                    logits = y_out
                    yb_long = yb.long().view(-1)
                    loss = F.cross_entropy(logits, yb_long)
                else:
                    logits = y_out.squeeze(-1)
                    yb_float = yb.float().view(-1)
                    loss = F.binary_cross_entropy_with_logits(logits, yb_float)

                c, t = batch_accuracy_from_logits(y_out, yb, num_classes=num_classes)
                correct_train += c
                total_train += t
            else:
                raise ValueError(f"Unknown task type: {task}")

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * B
            train_n += B

        train_epoch_loss = total_loss / train_n
        train_losses.append(train_epoch_loss)

        if task == "classification":
            train_acc = correct_train / max(total_train, 1)
            train_accuracies.append(train_acc)
        else:
            train_accuracies.append(None)

        # ---------- VALIDATION ----------
        model.eval()
        val_total_loss = 0.0
        val_n = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for xb, yb in batchify(x_val, y_val, batch_size=batch_size):
                B = xb.size(0)
                state = model.initial_state(batch_size=B, device=device)

                outputs = None
                for step in range(trm_cfg.halt_max_steps):
                    state, outputs = model(state, xb)
                    if bool(state.halted.all()):
                        break

                y_out = outputs["y_pred"]

                if task == "regression":
                    y_pred = y_out.squeeze(-1)
                    if yb.dim() > 1:
                        yb = yb.squeeze(-1)
                    val_loss = F.mse_loss(y_pred, yb)

                elif task == "classification":
                    if num_classes is not None and num_classes > 2:
                        logits = y_out
                        yb_long = yb.long().view(-1)
                        val_loss = F.cross_entropy(logits, yb_long)
                    else:
                        logits = y_out.squeeze(-1)
                        yb_float = yb.float().view(-1)
                        val_loss = F.binary_cross_entropy_with_logits(logits, yb_float)

                    c, t = batch_accuracy_from_logits(y_out, yb, num_classes=num_classes)
                    correct_val += c
                    total_val += t
                else:
                    raise ValueError(f"Unknown task type: {task}")

                val_total_loss += val_loss.item() * B
                val_n += B

        val_epoch_loss = val_total_loss / val_n
        val_losses.append(val_epoch_loss)

        if task == "classification":
            val_acc = correct_val / max(total_val, 1)
            val_accuracies.append(val_acc)
        else:
            val_accuracies.append(None)

    # attach histories to the model
    model.train_losses = train_losses
    model.val_losses = val_losses
    model.train_accuracies = train_accuracies
    model.val_accuracies = val_accuracies
    model.task = task
    model.num_classes = num_classes
    model.num_epochs = num_epochs

    print("Training / validation finished")
    return model, train_losses, val_losses


# ---------------------------------------------------------------------
# 2) MLP training: same interface (regression / classification)
# ---------------------------------------------------------------------
def train_mlp(
    D,
    x_train,
    y_train,
    x_val,
    y_val,
    num_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    num_layers: int = 2,
    expansion: float = 2.0,
    device=None,
    task: str = "regression",      # "regression" or "classification"
    num_classes: int = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move datasets once
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)

    if task == "regression":
        out_dim = 1
    else:
        out_dim = 1 if (num_classes is None or num_classes <= 2) else num_classes

    model = MLPBaseline(
        hidden_size=D,
        num_layers=num_layers,
        expansion=expansion,
        out_dim=out_dim,
    )
    opt = optim.AdamW(model.parameters(), lr=lr)

    model.to(device)
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # ---------- TRAIN ----------
        model.train()
        total_loss = 0.0
        train_n = 0
        correct_train = 0
        total_train = 0

        for xb, yb in batchify(x_train, y_train, batch_size=batch_size):
            y_out = model(xb)

            if task == "regression":
                y_pred = y_out.squeeze(-1)
                if yb.dim() > 1:
                    yb = yb.squeeze(-1)
                loss = F.mse_loss(y_pred, yb)

            elif task == "classification":
                if num_classes is not None and num_classes > 2:
                    logits = y_out
                    yb_long = yb.long().view(-1)
                    loss = F.cross_entropy(logits, yb_long)
                else:
                    logits = y_out.squeeze(-1)
                    yb_float = yb.float().view(-1)
                    loss = F.binary_cross_entropy_with_logits(logits, yb_float)

                c, t = batch_accuracy_from_logits(y_out, yb, num_classes=num_classes)
                correct_train += c
                total_train += t
            else:
                raise ValueError(f"Unknown task type: {task}")

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * xb.size(0)
            train_n += xb.size(0)

        train_epoch_loss = total_loss / train_n
        train_losses.append(train_epoch_loss)

        if task == "classification":
            train_acc = correct_train / max(total_train, 1)
            train_accuracies.append(train_acc)
        else:
            train_accuracies.append(None)

        # ---------- VAL ----------
        model.eval()
        val_total_loss = 0.0
        val_n = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for xb, yb in batchify(x_val, y_val, batch_size=batch_size):
                y_out = model(xb)

                if task == "regression":
                    y_pred = y_out.squeeze(-1)
                    if yb.dim() > 1:
                        yb = yb.squeeze(-1)
                    val_loss = F.mse_loss(y_pred, yb)

                elif task == "classification":
                    if num_classes is not None and num_classes > 2:
                        logits = y_out
                        yb_long = yb.long().view(-1)
                        val_loss = F.cross_entropy(logits, yb_long)
                    else:
                        logits = y_out.squeeze(-1)
                        yb_float = yb.float().view(-1)
                        val_loss = F.binary_cross_entropy_with_logits(logits, yb_float)

                    c, t = batch_accuracy_from_logits(y_out, yb, num_classes=num_classes)
                    correct_val += c
                    total_val += t
                else:
                    raise ValueError(f"Unknown task type: {task}")

                val_total_loss += val_loss.item() * xb.size(0)
                val_n += xb.size(0)

        val_epoch_loss = val_total_loss / val_n
        val_losses.append(val_epoch_loss)

        if task == "classification":
            val_acc = correct_val / max(total_val, 1)
            val_accuracies.append(val_acc)
        else:
            val_accuracies.append(None)

    model.train_losses = train_losses
    model.val_losses = val_losses
    model.train_accuracies = train_accuracies
    model.val_accuracies = val_accuracies
    model.task = task
    model.num_classes = num_classes
    model.num_epochs = num_epochs

    print("Training / validation finished")
    return model, train_losses, val_losses
