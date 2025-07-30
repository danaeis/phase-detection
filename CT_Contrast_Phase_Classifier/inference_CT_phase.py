import argparse
import os
from tqdm import tqdm
import pandas as pd
import random
import uuid
import shutil
import subprocess
import warnings
import pathlib
from feature_imputation import (mean_imputation, knn_imputation,
                                knn_imputation_within_class,
                                create_intensity_feature_from_file)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load

def recursive_glob(rootdir, suffix):
    suffix_list = []
    if isinstance(suffix, str):
        suffix_list.append(suffix)
    else:
        suffix_list = suffix

    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        for s in suffix_list
        if filename.endswith(s)
    ]



def extract_key_organs_by_TS(input_file, speed, seg_dir):
    file_basename = os.path.basename(input_file)
    output_path = os.path.join(seg_dir, 'total', 'seg_vol.nii.gz')

    if speed == 'normal':
        command = f"TotalSegmentator -i {input_file} -o {output_path} --statistics --ml"
    elif speed == 'fast':
        command = f"TotalSegmentator -i {input_file} -o {output_path} --statistics --ml -f"
    else:
        command = f"TotalSegmentator -i {input_file} -o {output_path} --statistics --ml -ff"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    output, error = process.communicate()
    if error:
        print(f"Error in total segmentation of {file_basename}: {error}")
    else:
        print(f"Total segmentation of {file_basename} successfully.")

    output_path = os.path.join(seg_dir, 'heart', 'seg_vol.nii.gz')
    command = f"TotalSegmentator -i {input_file} -o {output_path} --statistics --ml -ta heartchambers_highres"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    output, error = process.communicate()
    if error:
        print(f"Error in heart segmentation of {file_basename}: {error}")
    else:
        print(f"Heart segmentation of {file_basename} successfully.")

def extract_image_features(seg_dir, imputation_method='mean', n_neighbors=5):
    # Imputation method initialization
    train_feature_info_df = pd.read_csv("model_files/train_set_respacing113_radiomics.csv")
    # create initial image feature for the current input CT scan
    initial_feature_df = create_intensity_feature_from_file(
            os.path.join(seg_dir, 'total', 'statistics.json'),
            os.path.join(seg_dir, 'heart', 'statistics.json'))
    initial_feature_names = initial_feature_df.columns[:]

    if imputation_method == 'mean':
        modfied_train_feature_df, mean_filled_val \
            = mean_imputation(train_feature_info_df, train_feature_info_df.columns[1:])
        imputed_feature_df, _ = mean_imputation(
            initial_feature_df, initial_feature_names, mean_filled_val)
    elif imputation_method == 'knn':
        modfied_train_feature_df, knn_imputer \
            = knn_imputation(train_feature_info_df, train_feature_info_df.columns[1:],
                             n_neighbors=n_neighbors)
        imputed_feature_df, _ = knn_imputation(
            initial_feature_df, initial_feature_names, imputer=knn_imputer,
            n_neighbors=n_neighbors)
    else:
        raise ValueError("invalid imputation method {}".format(imputation_method))

    return imputed_feature_df

def predict_contrast_phase(img_feature, model_type, imputation_method):
    if model_type == 'rf' and imputation_method == 'mean':
        model_path = 'model_files/rf_model_mean.joblib'
    elif model_type == 'rf' and imputation_method == 'knn':
        model_path = 'model_files/rf_model_knn_imputed_data.joblib'
    elif model_type == 'gb' and imputation_method == 'mean':
        model_path = 'model_files/gb_model_mean.joblib'
    else:
        model_path = 'model_files/gb_model_knn_imputed_data.joblib'

    if not os.path.exists(model_path):
        raise ValueError("{} is not existed!".format(model_path))

    prediction_model = load(model_path)
    predictions = prediction_model.predict(img_feature.values)
    prediction_proba = prediction_model.predict_proba(img_feature.values)
    return predictions, prediction_proba


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_data', help='This program accepts three types of '
                                                   'inputs: 1) a single file in .nii or .nii.gz format; '
                                                   '2) a file fold with all or parts of files in .nii.gz format '
                                                   'its subfolds with these file format also selected; '
                                                   '3) a txt file list a set of .nii or .nii.gz files',
                        required=True)
    parser.add_argument('--imputation_method', type=str, default="mean", choices=['mean', 'knn'],
                        help="Specify the imputation method. Default: 'mean'.")
    parser.add_argument('--model_type', type=str, default="rf", choices=['rf', 'gb'],
                        help="Specify the imputation method. Default: 'rf'.")
    parser.add_argument('--n_neighbors', type=int, default=5,
                        help="Specify the number of neighbors for KNN feature imputation")
    parser.add_argument('--TS_speed', type=str, default="fast", choices=['normal', 'fast', 'fastest'],
                        help="Specify the TS computation speed")
    parser.add_argument('-o', "--output_file", help="The output file with phase information",
                        required=True)
    args = parser.parse_args()

    if args.imputation_method not in ['mean', 'knn'] or args.model_type not in ['rf', 'gb']:
        raise ValueError("Either {} or {} is not input properly!".format(
            args.imputation_method, args.model_type))

    if os.path.isdir(args.input_data):
        input_file_list = recursive_glob(args.input_data, ['.nii.gz', '.nii'])
    else:
        if args.input_data.endswith('.txt'):
            with open(args.input_data) as f:
                tmp_image_list = f.read().splitlines()
        else:
            tmp_image_list = []
            tmp_image_list.append(args.input_data)
        input_file_list = [f for f in tmp_image_list if os.path.isfile(f) and os.path.exists(f)]

    # create temp fold
    tmp_fold = 'tmp_fold'
    tmp_fold = os.path.join(os.path.dirname(args.output_file), tmp_fold)
    pathlib.Path(tmp_fold).mkdir(parents=True, exist_ok=True)

    res_dict = {}
    res_dict['filename'] = []
    res_dict['CT phase'] = []
    res_dict['Non-contrast phase probability'] = []
    res_dict['Arterial phase probability'] = []
    res_dict['Portal venous phase probability'] = []
    res_dict['Nephrographic phase probability'] = []
    res_dict['Delayed phase probability'] = []

    for filename in tqdm(input_file_list, desc="Processing files"):
        tqdm.write(f"Processing: {filename}")
        if filename.find('.nii') == -1:
            warnings("invalid file type: must have .nii in {}".format(filename))
            continue

        extract_key_organs_by_TS(filename, args.TS_speed, tmp_fold)
        img_feature_vector = extract_image_features(tmp_fold, args.imputation_method, args.n_neighbors)
        pred, pred_prob = predict_contrast_phase(img_feature_vector, args.model_type, args.imputation_method)
        res_dict['filename'].append(os.path.basename(filename))
        if pred[0] == 0:
            res_dict['CT phase'].append('non-contrast')
        elif pred[0] == 1:
            res_dict['CT phase'].append('arterial')
        elif pred[0] == 2:
            res_dict['CT phase'].append('portal venous')
        elif pred[0] == 3:
            res_dict['CT phase'].append('nephrographic')
        else:
            res_dict['CT phase'].append('delayed')

        res_dict['Non-contrast phase probability'].append(pred_prob[0][0])
        res_dict['Arterial phase probability'].append(pred_prob[0][1])
        res_dict['Portal venous phase probability'].append(pred_prob[0][2])
        res_dict['Nephrographic phase probability'].append(pred_prob[0][3])
        res_dict['Delayed phase probability'].append(pred_prob[0][4])

    res_df = pd.DataFrame.from_dict(res_dict)
    res_df.to_csv(args.output_file, index=False)

    if os.path.exists(tmp_fold):
        shutil.rmtree(tmp_fold)
        print(f"Folder '{tmp_fold}' and its contents removed successfully.")
    else:
        print(f"Folder '{tmp_fold}' does not exist.")


if __name__ == '__main__':
    main()
