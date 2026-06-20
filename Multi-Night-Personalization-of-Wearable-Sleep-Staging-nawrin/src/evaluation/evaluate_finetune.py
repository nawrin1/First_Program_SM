import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from src.data.dataset import SleepNightDataset
from torch.utils.data import DataLoader
from src.lib.config import BATCH_SIZE, FT_EPOCHS, FT_LR
from src.lib.utils import compute_metrics


def evaluate_finetune(global_encoder, global_classifier, calib_nights, test_nights, global_mean, global_std):
    device = next(global_encoder.parameters()).device
    ft_encoder = deepcopy(global_encoder).to(device)
    classifier = deepcopy(global_classifier).to(device)

    
    for  param in ft_encoder.parameters():
        param.requires_grad = True
    
    optimizer = optim.Adam(list(ft_encoder.parameters()) + list(classifier.parameters()), lr=1e-5,  weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Fine‑tune on calibration nights
    calib_dataset = SleepNightDataset(calib_nights, use_windowed=True)
    calib_loader = DataLoader(calib_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    ft_encoder.train()
    classifier.train()
    for _ in range(FT_EPOCHS):
        for xb, yb in calib_loader:
            xb, yb = xb.to(device), yb.to(device)
            xb = (xb - global_mean) / global_std
            optimizer.zero_grad()
            emb = ft_encoder(xb)
            logits = classifier(emb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    # Test on query nights
    ft_encoder.eval()
    classifier.eval()
    all_preds = []
    all_trues = []
    for night in test_nights:
        X_np, y_np = night['windowed']
        X_t = torch.tensor(X_np, dtype=torch.float32).to(device)
        X_t = (X_t - global_mean) / global_std
        with torch.no_grad():
            emb = ft_encoder(X_t)
            logits = classifier(emb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_trues.extend(y_np)

    macro_f1, kappa, per_class = compute_metrics(all_preds, all_trues)
    return all_preds, all_trues, macro_f1, kappa, per_class
