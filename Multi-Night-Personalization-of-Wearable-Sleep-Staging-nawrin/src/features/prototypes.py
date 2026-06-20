import torch

from src.features.extract_embeddings import extract_embeddings
from src.lib.config import EMBED_DIM, NUM_CLASSES


def compute_fair_global_prototypes(encoder, training_subjects_dict, global_mean, global_std):
    """
    Compute class prototypes by averaging subject-level prototypes.

    A subject contributes to a class only when that class is present in that
    subject's training nights. This prevents missing classes from being treated
    as zero-valued embeddings during the global prototype average.
    """
    device = next(encoder.parameters()).device
    encoder.eval()

    prototype_sums = torch.zeros(NUM_CLASSES, EMBED_DIM, dtype=torch.float32, device=device)
    prototype_counts = torch.zeros(NUM_CLASSES, dtype=torch.float32, device=device)

    for nights in training_subjects_dict.values():
        embeddings, labels = extract_embeddings(encoder, nights, global_mean, global_std)
        embeddings = torch.tensor(embeddings, dtype=torch.float32, device=device)
        labels = torch.tensor(labels, dtype=torch.long, device=device)

        for class_idx in range(NUM_CLASSES):
            class_mask = labels == class_idx
            if class_mask.any():
                prototype_sums[class_idx] += embeddings[class_mask].mean(dim=0)
                prototype_counts[class_idx] += 1

    missing_classes = prototype_counts == 0
    if missing_classes.any():
        raise ValueError(
            "Cannot compute global prototypes because the training data has no "
            f"examples for class indices {missing_classes.nonzero(as_tuple=True)[0].tolist()}."
        )

    return prototype_sums / prototype_counts.unsqueeze(1)
