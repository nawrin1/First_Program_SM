import torch
import numpy as np
from src.models.models import PrototypicalStager
from src.features.extract_embeddings import extract_embeddings
from src.lib.utils import compute_metrics


def evaluate_prototype(global_encoder, global_prototypes, calib_nights, test_nights,  global_mean, global_std):
    device = next(global_encoder.parameters()).device
    model = PrototypicalStager(global_encoder, global_prototypes).to(device)
    model.backbone.eval()

    # Calibration embeddings
    calib_emb, calib_lab = extract_embeddings(global_encoder, calib_nights,  global_mean, global_std)
    calib_emb = torch.tensor(calib_emb).to(device)
    calib_lab = torch.tensor(calib_lab).to(device)
    personalized = model.compute_personalized_prototypes(calib_emb, calib_lab)

    # Test on each night
    all_preds = []
    all_trues = []
    for night in test_nights:
        X_np, y_np = night['windowed']
        X_t = torch.tensor(X_np, dtype=torch.float32).to(device)
        X_t = (X_t - global_mean) / global_std
        with torch.no_grad():
            sim = model(X_t, personalized)
            preds = torch.argmax(sim, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_trues.extend(y_np)

    macro_f1, kappa, per_class = compute_metrics(all_preds, all_trues)
    return all_preds, all_trues, macro_f1, kappa, per_class
