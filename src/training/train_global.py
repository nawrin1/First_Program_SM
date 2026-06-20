import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data.dataset import SleepNightDataset
from src.lib.config import BATCH_SIZE, PRETRAIN_EPOCHS
from src.models.models import TemporalSleepEncoder


def _compute_class_weights(labels, num_classes=5):
    counts = np.bincount(labels, minlength=num_classes)
    total_samples = counts.sum()
    class_weights = np.ones(num_classes, dtype=np.float32)

    for class_idx, count in enumerate(counts):
        if count > 0:
            class_weights[class_idx] = total_samples / (num_classes * count)

    return class_weights


def train_global_encoder(train_nights, verbose=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SleepNightDataset(train_nights, use_windowed=True)
    if len(dataset) == 0:
        raise ValueError("No valid training samples after removing NaN/Inf rows.")

    global_mean = dataset.X.mean(dim=(0, 1), keepdim=True).to(device)
    global_std = dataset.X.std(dim=(0, 1), keepdim=True).to(device)
    global_std = torch.clamp(global_std, min=1e-4)

    class_weights_np = _compute_class_weights(dataset.y.numpy(), num_classes=5)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    encoder = TemporalSleepEncoder().to(device)
    classifier = nn.Linear(64, 5).to(device)

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=1e-3,
        weight_decay=1e-5,
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    if verbose:
        print("Pre-training global backbone with training-set normalization...")

    for epoch in range(PRETRAIN_EPOCHS):
        encoder.train()
        classifier.train()
        total_loss = 0.0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            xb = torch.nan_to_num(xb, nan=0.0, posinf=1.0, neginf=-1.0)
            xb = (xb - global_mean) / global_std

            optimizer.zero_grad()
            logits = classifier(encoder(xb))
            loss = criterion(logits, yb)

            if not torch.isfinite(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=0.5)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        if verbose:
            print(f"Epoch {epoch + 1}/{PRETRAIN_EPOCHS} loss: {avg_loss:.4f}")

        scheduler.step(avg_loss)

    return encoder, classifier, global_mean, global_std
