import pandas as pd

# Path to your original CSV
input_csv = '../../ncct_cect/vindr_ds/vindr_nifti_metadata.csv'  # <-- change this to your actual file
output_csv = '../../ncct_cect/vindr_ds/original_volumes/Abdomen/label/Abdomen_label.csv'

# Read the original CSV
# It must have columns: SeriesInstanceUID, ct_phase, orig_volume_path, Phase

df = pd.read_csv(input_csv)

# Create the new columns
# ID: SeriesInstanceUID
# Contrast: 1 if ct_phase > 0 else 0
# path: orig_volume_path
# phase: ct_phase
# phase_name: Phase

df_new = pd.DataFrame({
    'ID': df['SeriesInstanceUID'],
    'Contrast': df['ct_phase'].apply(lambda x: 1 if pd.to_numeric(x, errors='coerce') > 0 else 0),
    'phase': df['ct_phase'],
    'phase_name': df['Phase']
})

# Save the new CSV
# You can change output_csv to your desired output path

df_new.to_csv(output_csv, index=False)

print(f'Wrote new label file: {output_csv}') 