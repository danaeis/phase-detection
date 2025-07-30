CT Phase Classifier

Overview
The CT Phase Classifier processes CT images to determine their phase and outputs the results in a CSV file.

Installation
1 Install the latest version of PyTorch and TotalSegmentator:
pip install TotalSegmentator
pip install scikit-learn == 1.3.0

2 Obtain a license for TotalSegmentator:

 * Visit TotalSegmentator Academic License (https://backend.totalsegmentator.com/license-academic/)
 * Fill in your personal information
 * Select at least the "heartchambers_highres" task
 
3 Activate the virtual environment for CT Phase Classifier:
   conda activate ct_phase_classifier

Usage
Run the following command based on your input type:

python inference_CT_phase.py -i file_list.txt -o OUTPUT_FILE
python inference_CT_phase.py -i file.nii.gz -o OUTPUT_FILE
python inference_CT_phase.py -i file_DIR -o OUTPUT_FILE

Command-line Arguments

usage: inference_CT_phase.py [-h] [-i INPUT_DATA] -o OUTPUT_FOLDER 
                             [--imputation_method] [--model_type] 
                             [--n_neighbors] [--TS_speed]

Argument	Description
-h, --help	Show the help message and exit
-i INPUT_DATA	Input data (choose one of the following):
1. file_list.txt – a text file containing paths to NIFTI files
2. file.nii.gz – a single NIFTI file
3. file_DIR – a directory containing NIFTI files
-o, --output	The output CSV file containing the predicted CT phase and phase probability
-v, --verbose	Enable verbose output
--imputation_method	Feature imputation method: mean (default, mean intensity) or knn (K-Nearest Neighbors)
--model_type	Classifier type: rf (Random Forest, default) or gb (Gradient Boosting)
--n_neighbors	Number of neighbors for KNN (default: 5)
--TS_speed	Speed setting for TotalSegmentator: normal, fast, or fastest (default: fast)

Examples
1 Processing a list of CT files:

python inference_CT_phase.py -i fn_fat_patients.txt -o res.csv
This command predicts the CT phases for files listed in "fn_fat_patients.txt" and saves the results in "res.csv".

2 Processing a single CT volume with custom settings:

python inference_CT_phase.py -i ct_vol.nii.gz -o res.csv --imputation_method knn --model_type gb
Uses KNN for feature imputation and Gradient Boosting for classification.

3 Processing all NIFTI files in a directory with adjusted settings:

python inference_CT_phase.py -i fn_dir -o res.csv --model_type gb --TS_speed normal
Uses the Gradient Boosting classifier and TotalSegmentator at normal speed to process all files in "fn_dir".