import torch
import numpy as np
from torch.utils.data import DataLoader
from src.data.dataset import SleepNightDataset
from src.lib.config import BATCH_SIZE

# def extract_embeddings(encoder, nights_list):
#     """
#     Extracts underlying latents from the backbone encoder across nights
#     using static global scaling matrices to insulate against data leakage.
#     """
#     device = next(encoder.parameters()).device
#     encoder.eval()
#     dataset = SleepNightDataset(nights_list, use_windowed=True)
#     loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
#     all_embs = []
#     all_labels = []
#     with torch.no_grad():
#         for xb, yb in loader:
#             xb, yb = xb.to(device), yb.to(device)
#             mean = xb.mean(dim=(0,1), keepdim=True)
#             std = torch.clamp(xb.std(dim=(0,1), keepdim=True) + 1e-8, min=1e-4)
#             xb = (xb - mean) / std
#             emb = encoder(xb).cpu()
#             all_embs.append(emb)
#             all_labels.append(yb.cpu())
#     # return np.concatenate(all_embs, axis=0), np.concatenate(all_labels, axis=0)
#     return torch.cat(all_embs, dim=0).numpy(), torch.cat(all_labels, dim=0).numpy()


def extract_embeddings(encoder, nights, global_mean, global_std):
    """
        Extracts underlying latents from the backbone encoder across nights
        using static global scaling matrices to insulate against data leakage.
    """
    device = next(encoder.parameters()).device
    encoder.eval()
    
    embeddings_list = []
    labels_list = []
    
    with torch.no_grad():
        for night in nights:
            X_np, y_np = night['windowed']
            X_t = torch.tensor(X_np, dtype=torch.float32).to(device)
            
           
            X_t = (X_t - global_mean) / global_std
            
           
            embs = encoder(X_t)
            
            embeddings_list.append(embs.cpu().numpy())
            labels_list.append(y_np)
            
  
    all_embeddings = np.concatenate(embeddings_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)
    
    return all_embeddings, all_labels