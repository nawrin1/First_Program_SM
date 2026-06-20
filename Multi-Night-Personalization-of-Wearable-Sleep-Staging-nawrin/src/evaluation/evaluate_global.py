import torch
import numpy as np
from src.lib.utils import compute_metrics

def evaluate_global_baseline(global_encoder, global_classifier, test_nights, global_mean, global_std):
    device = next(global_encoder.parameters()).device
    global_encoder.eval()
    global_classifier.eval()
    
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for night in test_nights:
            X_np, y_np = night['windowed']
            X_t = torch.tensor(X_np, dtype=torch.float32).to(device)
            
            # Scale the test features
            X_t = (X_t - global_mean) / global_std
            
            logits = global_classifier(global_encoder(X_t))
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_trues.extend(y_np)
            
    macro_f1, kappa, per_class = compute_metrics(all_preds, all_trues)
    return all_preds, all_trues, macro_f1, kappa, per_class