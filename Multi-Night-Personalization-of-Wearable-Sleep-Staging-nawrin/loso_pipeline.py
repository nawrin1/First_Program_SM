import os

import numpy as np
import torch
from scipy.stats import wilcoxon

from src.data.data_loader import load_preprocessed_data
from src.evaluation.evaluate_finetune import evaluate_finetune
from src.evaluation.evaluate_global import evaluate_global_baseline
from src.evaluation.evaluate_prototype import evaluate_prototype
from src.features.prototypes import compute_fair_global_prototypes
from src.lib.config import PROCESSED_DIR, RANDOM_SEED, ensure_output_dirs
from src.lib.utils import bootstrap_ci, save_evaluation_artifacts
from src.logger import get_logger
from src.training.train_global import train_global_encoder


torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

logger = get_logger("loso_pipeline")


def run_loso():
    ensure_output_dirs()
    all_subjects = load_preprocessed_data()
    subject_ids = list(all_subjects.keys())
    ft_f1_list = []
    proto_f1_list = []
    global_f1_list = []
    all_global_preds = []
    all_global_trues = []
    all_ft_preds = []
    all_ft_trues = []
    all_proto_preds = []
    all_proto_trues = []

    logger.info(subject_ids)
    print(f"Loaded {len(all_subjects)} subjects")
    print(f"Keys: {list(all_subjects.keys())}")

    for test_idx, test_subj in enumerate(subject_ids):
        logger.info(f"\n=== Fold {test_idx + 1}/{len(subject_ids)}: Test subject {test_subj} ===")

        train_subjects = {s: all_subjects[s] for s in subject_ids if s != test_subj}
        train_nights = [night for nights in train_subjects.values() for night in nights]

        encoder, global_classifier, global_mean, global_std = train_global_encoder(
            train_nights, verbose=True
        )
        global_protos = compute_fair_global_prototypes(
            encoder, train_subjects, global_mean, global_std
        )

        nights = all_subjects[test_subj]
        if len(nights) < 4:
            logger.info(f"Subject {test_subj} has fewer than 4 nights, skipping")
            continue

        calib_nights = nights[1:3]
        test_nights = nights[3:]

        global_preds, global_trues, global_f1, _, _ = evaluate_global_baseline(
            encoder, global_classifier, test_nights, global_mean, global_std
        )
        global_f1_list.append(global_f1)
        all_global_preds.extend(global_preds)
        all_global_trues.extend(global_trues)

        ft_preds, ft_trues, ft_f1, _, _ = evaluate_finetune(
            encoder, global_classifier, calib_nights, test_nights, global_mean, global_std
        )
        ft_f1_list.append(ft_f1)
        all_ft_preds.extend(ft_preds)
        all_ft_trues.extend(ft_trues)

        proto_preds, proto_trues, proto_f1, _, _ = evaluate_prototype(
            encoder, global_protos, calib_nights, test_nights, global_mean, global_std
        )
        proto_f1_list.append(proto_f1)
        all_proto_preds.extend(proto_preds)
        all_proto_trues.extend(proto_trues)

        logger.info(
            f"Fold {test_idx + 1}: Global F1={global_f1:.4f}, "
            f"Fine-tune F1={ft_f1:.4f}, Prototype F1={proto_f1:.4f}"
        )

    ft_f1 = np.array(ft_f1_list)
    global_f1 = np.array(global_f1_list)
    proto_f1 = np.array(proto_f1_list)

    logger.info("\n" + "=" * 50)
    logger.info(
        f"Global: mean F1 = {global_f1.mean():.4f} +/- {global_f1.std():.4f}, "
        f"95% CI = {bootstrap_ci(global_f1)}"
    )
    logger.info(
        f"Fine-tuning: mean F1 = {ft_f1.mean():.4f} +/- {ft_f1.std():.4f}, "
        f"95% CI = {bootstrap_ci(ft_f1)}"
    )
    logger.info(
        f"Prototypical: mean F1 = {proto_f1.mean():.4f} +/- {proto_f1.std():.4f}, "
        f"95% CI = {bootstrap_ci(proto_f1)}"
    )

    if len(proto_f1) > 1:
        try:
            _, p = wilcoxon(proto_f1, ft_f1)
            logger.info(f"Wilcoxon signed-rank test p-value = {p:.6f}")
        except Exception as exc:
            logger.warning(f"Could not compute Wilcoxon signed-rank test: {exc}")

    np.save(os.path.join(PROCESSED_DIR, "ft_f1_scores.npy"), ft_f1)
    np.save(os.path.join(PROCESSED_DIR, "proto_f1_scores.npy"), proto_f1)
    np.save(os.path.join(PROCESSED_DIR, "global_f1_scores.npy"), global_f1)
    save_evaluation_artifacts("global", all_global_preds, all_global_trues, PROCESSED_DIR)
    save_evaluation_artifacts("fine_tune", all_ft_preds, all_ft_trues, PROCESSED_DIR)
    save_evaluation_artifacts("prototype", all_proto_preds, all_proto_trues, PROCESSED_DIR)
    return ft_f1_list, proto_f1_list, global_f1_list


if __name__ == "__main__":
    run_loso()
