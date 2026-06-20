import numpy as np
from src.lib.config import CALIB_NIGHTS_LIST, PROCESSED_DIR, ensure_output_dirs
from src.data.data_loader import load_preprocessed_data
from src.training.train_global import train_global_encoder
from src.evaluation.evaluate_prototype import evaluate_prototype
from src.lib.utils import compute_metrics, bootstrap_ci
from src.features.prototypes import compute_fair_global_prototypes


def run_calibration_ablation(num_subjects_for_ablation=10):
    ensure_output_dirs()
    all_subjects = load_preprocessed_data()
    subject_ids = list(all_subjects.keys())[:num_subjects_for_ablation]
    results = {n: [] for n in CALIB_NIGHTS_LIST}

    for calib_len in CALIB_NIGHTS_LIST:
        print(f"\n=== Calibration nights: {calib_len} ===")
        for test_subj in subject_ids:
            # Build training subjects (all others in this subset)
            train_subj_ids = [s for s in subject_ids if s != test_subj]
            train_nights = []
            for s in train_subj_ids:
                for night in all_subjects[s]:
                    train_nights.append(night)

            # Train global encoder
            encoder, _, global_mean, global_std = train_global_encoder(train_nights, verbose=False)

            train_subjects = {s: all_subjects[s] for s in train_subj_ids}
            global_protos = compute_fair_global_prototypes(encoder, train_subjects, global_mean, global_std)

            # Test subject nights: skip night 0 
            test_nights = all_subjects[test_subj]
            if len(test_nights) < calib_len + 1:
                continue
            # Calibration: 
            calib_nights = test_nights[1:1 + calib_len]
            
            test_eval = test_nights[1 + calib_len:]
            if len(test_eval) == 0:
                continue

            _, _, macro_f1, _, _ = evaluate_prototype(
                encoder, global_protos, calib_nights, test_eval, global_mean, global_std
            )
            results[calib_len].append(macro_f1)
            print(f"Test subject {test_subj} – macro F1: {macro_f1:.4f}")

    print("\n=== Calibration Length Ablation Results ===")
    for cl in CALIB_NIGHTS_LIST:
        scores = results[cl]
        if scores:
            mean_f1 = np.mean(scores)
            ci = bootstrap_ci(scores)
            print(f"{cl} calibration night(s): mean F1 = {mean_f1:.4f}, 95% CI = {ci}")
    return results


if __name__ == "__main__":
    run_calibration_ablation()
