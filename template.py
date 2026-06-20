import os
from pathlib import Path

project_name = "./"

list_of_files = [

    # -----------------------------
    # Data
    # -----------------------------
    f"{project_name}/Data/raw/.gitkeep",

    # -----------------------------
    # Source Code
    # -----------------------------
    f"{project_name}/src/__init__.py",

    f"{project_name}/src/config.py",
    f"{project_name}/src/data_loader.py",
    f"{project_name}/src/dataset.py",
    f"{project_name}/src/models.py",

    f"{project_name}/src/train_global.py",
    f"{project_name}/src/extract_embeddings.py",

    f"{project_name}/src/evaluate_prototype.py",
    f"{project_name}/src/evaluate_finetune.py",

    f"{project_name}/src/ablation_context.py",
    f"{project_name}/src/ablation_calib_len.py",

    f"{project_name}/src/utils.py",

    # -----------------------------
    # Logger
    # -----------------------------
    f"{project_name}/src/logger/__init__.py",
    
    
    # -----------------------------
    # Exception
    # -----------------------------
    f"{project_name}/src/exception/__init__.py",
    
    
    # -----------------------------
    # Notebooks
    # -----------------------------
    f"{project_name}/notebooks/exploratory.ipynb",

    # -----------------------------
    # Results
    # -----------------------------
    f"{project_name}/results/figures/.gitkeep",
    f"{project_name}/results/tables/.gitkeep",
    f"{project_name}/results/models/.gitkeep",

    # -----------------------------
    # Root Files
    # -----------------------------
    f"{project_name}/README.md",
    f"{project_name}/requirements.txt",
]


for filepath in list_of_files:

    path_to_file = Path(filepath)

    filedir, filename = os.path.split(path_to_file)

    # Create directories if missing
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)

    # Create file only if:
    # 1. file does not exist
    # 2. OR file exists but empty
    if (
        not os.path.exists(path_to_file)
        or os.path.getsize(path_to_file) == 0
    ):

        with open(path_to_file, "w", encoding="utf-8") as f:
            pass

        print(f"Created: {path_to_file}")

    else:
        print(f"Already exists: {path_to_file}")