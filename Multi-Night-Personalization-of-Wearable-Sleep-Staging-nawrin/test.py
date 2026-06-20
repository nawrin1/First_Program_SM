# import matplotlib.pyplot as plt
# from src.data.data_loader import load_preprocessed_data
# import numpy as np

# data = load_preprocessed_data()
# subj = 'Bidslab00'
# night_idx = 0  # first night
# X_seq, y_seq = data[subj][night_idx]['windowed']  # shape (epochs, 9, 9)

# # Take first 60 epochs (30 minutes)
# epochs_to_plot = 60
# X_plot = X_seq[:epochs_to_plot]  # (60, 9, 9)

# # Plot one representative feature (e.g., motion mean, feature 0)
# motion_mean = X_plot[:, 4, 0]  # center epoch? Actually X_plot[i, center_idx, feature] – careful
# # Better: extract center epoch (index 4) for each window
# center_epoch_features = X_plot[:, 4, :]  # (60, 9)
# motion_mean = center_epoch_features[:, 0]
# hr_mean = center_epoch_features[:, 4]

# fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
# ax[0].plot(motion_mean, label='Motion mean')
# ax[0].set_ylabel('Motion')
# ax[1].plot(hr_mean, label='HR mean')
# ax[1].set_ylabel('HR (BPM)')
# ax[2].plot(y_seq[:epochs_to_plot], drawstyle='steps-mid', label='Sleep stage')
# ax[2].set_yticks([0,1,2,3,4])
# ax[2].set_yticklabels(['Wake','N1','N2','N3','REM'])
# ax[2].set_xlabel('Epoch (30s)')
# ax[2].legend()
# plt.tight_layout()
# plt.show()

import os
import pandas as pd
import scipy.io as sio
from datetime import datetime
import pytz
from src.data.data_loader import convert_to_unix   # reuse your conversion function

# Paths (adjust if your raw data is elsewhere)
subj = 'Bidslab00'
night_idx = 1  # night number (1-indexed)
raw_subj_path = f"data/raw/dataset/{subj}/{night_idx}"

# Load raw files as strings, then convert timestamps
motion_raw = pd.read_csv(os.path.join(raw_subj_path, 'motion.csv'), 
                         skiprows=1,  # ← ADD THIS
                         names=['timestamp','x','y','z'], 
                         dtype={'timestamp': str})

hr_raw = pd.read_csv(os.path.join(raw_subj_path, 'hr.csv'), 
                     skiprows=1,  # ← ADD THIS
                     names=['timestamp','heart_rate'], 
                     dtype={'timestamp': str})
mat = sio.loadmat(os.path.join(raw_subj_path, 'labels.mat'))

# Convert timestamps to Unix seconds using your function
motion_raw['timestamp'] = motion_raw['timestamp'].apply(convert_to_unix)
hr_raw['timestamp'] = hr_raw['timestamp'].apply(convert_to_unix)

# Get rec_start and convert to Unix UTC
rec_start_str = mat['recStart'].item()
naive_dt = datetime.strptime(rec_start_str, '%Y-%m-%d %H:%M:%S')
eastern = pytz.timezone('US/Eastern')
eastern_dt = eastern.localize(naive_dt)
utc_dt = eastern_dt.astimezone(pytz.UTC)
rec_start = utc_dt.timestamp()

print(f"rec_start = {rec_start}")
print(f"First motion timestamp = {motion_raw['timestamp'].iloc[0]}")
print(f"First HR timestamp = {hr_raw['timestamp'].iloc[0]}")
print(f"Difference motion - rec_start = {motion_raw['timestamp'].iloc[0] - rec_start:.2f} seconds")
print(f"Difference HR - rec_start = {hr_raw['timestamp'].iloc[0] - rec_start:.2f} seconds")