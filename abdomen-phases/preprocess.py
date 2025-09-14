import pandas as pd
import os
import numpy as np
from extract_slices import extract_slices

# Load labels
labels_df = pd.read_csv('labels.csv')

# Create output directories
os.makedirs('Split_data/images', exist_ok=True)
os.makedirs('Split_data/data', exist_ok=True)

# Map phases to numeric labels (assuming: non-contrast:0, arterial:1, venous:2, other:3)
phase_map = {'non-contrast': 0, 'arterial': 1, 'aterial': 1, 'venous': 2, 'other': 3}

all_rows = []
for _, row in labels_df.iterrows():
    nifti_path = row['orig_volume_path']
    study_id = row['StudyInstanceUID']
    series_id = row['SeriesInstanceUID']
    phase = row['Phase']
    label = phase_map.get(phase, 3)  # Default to 3 if unknown
    
    output_dir = os.path.join('Split_data/images', study_id)
    csv_rows = extract_slices(nifti_path, output_dir, study_id, series_id, label)
    all_rows.extend(csv_rows)

# Create DataFrame
df = pd.DataFrame(all_rows)

# Split into train/valid/test (80/10/10)
np.random.seed(42)
indices = np.arange(len(df))
np.random.shuffle(indices)
train_end = int(0.8 * len(df))
valid_end = int(0.9 * len(df))

train_df = df.iloc[indices[:train_end]]
valid_df = df.iloc[indices[train_end:valid_end]]
test_df = df.iloc[indices[valid_end:]]

# Save CSVs
train_df.to_csv('Split_data/data/train.csv', index=False)
valid_df.to_csv('Split_data/data/valid.csv', index=False)
test_df.to_csv('Split_data/data/test.csv', index=False)
print('Preprocessing completed. CSVs and PNG files generated in Split_data/')