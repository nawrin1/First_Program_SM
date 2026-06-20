# Multi-Night Personalization of Wearable Sleep Staging

Code for: **Multi-Night Personalization of Wearable Sleep Staging Using Instantaneous Heart Rate and Accelerometry: A Prototypical Adaptation Framework**.

## Dataset

Download the BIDSleep dataset from [PhysioNet](https://physionet.org/content/bidsleep-dataset/1.0.0/) and place it so the raw data folder contains subject directories such as:

```text
Data/raw/dataset/Bidslab00/1/motion.csv
Data/raw/dataset/Bidslab00/1/hr.csv
Data/raw/dataset/Bidslab00/1/labels.mat
```

Each night folder should contain `motion.csv`, `hr.csv`, and `labels.mat`.

## Setup

```bash
pip install -r requirements.txt
```

## Main Experiment

Run leave-one-subject-out evaluation:

```bash
python loso_pipeline.py
```

The pipeline:

- preprocesses raw BIDSleep files on the first run;
- trains a global CNN-BiLSTM encoder on all non-test subjects;
- evaluates the global classifier, fine-tuning baseline, and prototypical personalization;
- saves fold-level F1 arrays, aggregate predictions, per-class F1 tables, classification reports, and confusion matrices.

Key outputs:

```text
data/processed/global_f1_scores.npy
data/processed/ft_f1_scores.npy
data/processed/proto_f1_scores.npy
data/processed/global_preds.npy
data/processed/fine_tune_preds.npy
data/processed/prototype_preds.npy
results/tables/*_summary_metrics.csv
results/tables/*_per_class_f1.csv
results/tables/*_classification_report.txt
results/tables/*_confusion_matrix.csv
results/figures/*_confusion_matrix.png
```

## Ablations

```bash
python -m src.experiments.ablation_context
python -m src.experiments.ablation_calib_len
```

Outputs are saved under `results/` and `data/processed/` depending on the script.
