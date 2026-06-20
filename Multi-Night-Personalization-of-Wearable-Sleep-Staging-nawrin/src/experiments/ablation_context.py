import numpy as np

from src.data.data_loader import create_temporal_context_blocks, load_preprocessed_data
from src.evaluation.evaluate_prototype import evaluate_prototype
from src.features.prototypes import compute_fair_global_prototypes
from src.lib.config import CONTEXT_SIZES, ensure_output_dirs
from src.lib.utils import bootstrap_ci
from src.training.train_global import train_global_encoder


def _rewindow_nights(nights, context_size):
    rewindowed = []
    for night in nights:
        X_flat, y_flat = night["flat"]
        X_seq, y_seq = create_temporal_context_blocks(X_flat, y_flat, context_size=context_size)
        rewindowed.append({"windowed": (X_seq, y_seq), "flat": (X_flat, y_flat)})
    return rewindowed


def run_context_ablation(num_subjects_for_ablation=10):
    ensure_output_dirs()
    all_subjects = load_preprocessed_data()
    subject_ids = list(all_subjects.keys())[:num_subjects_for_ablation]
    results = {cs: [] for cs in CONTEXT_SIZES}

    for context_size in CONTEXT_SIZES:
        print(f"\n=== Context size +/-{context_size} (window length {2 * context_size + 1}) ===")
        for test_subj in subject_ids:
            train_subj_ids = [s for s in subject_ids if s != test_subj]
            train_subjects = {
                subject_id: _rewindow_nights(all_subjects[subject_id], context_size)
                for subject_id in train_subj_ids
            }
            train_nights = [night for subject_nights in train_subjects.values() for night in subject_nights]

            encoder, _, global_mean, global_std = train_global_encoder(train_nights, verbose=False)
            global_protos = compute_fair_global_prototypes(
                encoder, train_subjects, global_mean, global_std
            )

            test_nights = all_subjects[test_subj]
            if len(test_nights) < 4:
                continue

            calib_re = _rewindow_nights(test_nights[1:3], context_size)
            test_re = _rewindow_nights(test_nights[3:], context_size)
            if not test_re:
                continue

            _, _, macro_f1, _, _ = evaluate_prototype(
                encoder, global_protos, calib_re, test_re, global_mean, global_std
            )
            results[context_size].append(macro_f1)
            print(f"Test subject {test_subj} - macro F1: {macro_f1:.4f}")

    print("\n=== Context Window Ablation Results ===")
    for context_size, scores in results.items():
        if scores:
            mean_f1 = np.mean(scores)
            ci = bootstrap_ci(scores)
            print(f"Context +/-{context_size}: mean F1 = {mean_f1:.4f}, 95% CI = {ci}")
    return results


if __name__ == "__main__":
    run_context_ablation()
