import torch
import torch.nn as nn
import torch.nn.functional as F
from src.lib.config import NUM_FEATURES, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES


class TemporalSleepEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(NUM_FEATURES, 64, kernel_size=3, padding=1)
      
        self.ln = nn.LayerNorm(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.bilstm = nn.LSTM(64, HIDDEN_DIM, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(HIDDEN_DIM * 2, EMBED_DIM)

    def forward(self, x):
       
        x = x.permute(0, 2, 1)          
        x = self.conv(x)                
        x = x.permute(0, 2, 1)          
        x = self.ln(x)                  
        x = self.relu(x)
        x = self.dropout(x)
        
        
        x, _ = self.bilstm(x)           
        center = x.shape[1] // 2
        x = x[:, center, :]            
        return self.fc(x)  


class SimpleMLPEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(81, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
    def forward(self, x):
        return self.net(x)
class PrototypicalStager(nn.Module):
    def __init__(self, backbone, global_prototypes):
        super().__init__()
        self.backbone = backbone
        self.register_buffer('global_prototypes', global_prototypes)

    def compute_personalized_prototypes(self, calib_embeddings, calib_labels):
        personalized = torch.zeros_like(self.global_prototypes)
        for c in range(NUM_CLASSES):
            mask = (calib_labels == c)
            if mask.sum() > 0:
                personalized[c] = calib_embeddings[mask].mean(dim=0)
            else:
                personalized[c] = self.global_prototypes[c]
        return personalized

    def forward(self, query_sequences, personalized_prototypes):
        embeddings = self.backbone(query_sequences)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        prototypes = F.normalize(personalized_prototypes, p=2, dim=1)
        return torch.mm(embeddings, prototypes.t())