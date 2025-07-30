import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from typing import List, Optional, Dict, Tuple
import json

def load_and_merge_labels(df, labels_filepath):
    """
    Load class labels from a separate file and merge them into the main DataFrame.

    Parameters:
    - df: pandas DataFrame, the main dataset with filenames as the first column.
    - labels_filepath: str, the path to the labels file, which has filenames and labels.

    Returns:
    - pandas DataFrame with class labels merged.
    """
    # Load labels
    # Assuming filenames and labels are the first two columns
    labels_df = pd.read_csv(labels_filepath, usecols=[0, 1])
    # Rename columns for clarity
    labels_df.columns = ['fullpath', 'GT']

    # Merge the class labels into the main DataFrame based on filenames
    merged_df = pd.merge(df, labels_df, on='fullpath', how='left')

    return merged_df

def mean_imputation(df, columns, means=None):
    """
    Perform mean imputation on specified columns of a DataFrame.
    Assumes missing values are represented as 0, and replaces these with NaN before imputation.

    Parameters:
    df (pd.DataFrame): The DataFrame on which to perform mean imputation.
    columns (list): A list of column names on which to perform mean imputation.

    Returns:
    pd.DataFrame: The DataFrame after mean imputation.
    """
    # if 'GT' in df.columns:
    #     df = df.drop(columns=['GT'])
    df[columns] = df[columns].replace(0, np.nan)

    if means is None:
        filled_df = df[columns].fillna(df[columns].mean())
        return filled_df, df[columns].mean()
    if means is not None:
        return df.fillna(means), means

def knn_imputation(df, columns, n_neighbors=5, within_class=False, imputer=None):
    """
    Perform k-nearest neighbors imputation on specified columns of a DataFrame.

    Args:
        df (_type_): The DataFrame on which to perform knn imputation.
        columns (_type_): A list of column names on which to perform knn imputation.
        n_neighbors (int, optional): The number of neighbors to form as well as the number of centroids to generate. Defaults to 3.
    """
    df[columns] = df[columns].replace(0, np.nan)

    # kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(df[columns].dropna())
    # df[columns] = df[columns].fillna(kmeans.cluster_centers_[kmeans.predict(df[columns])])

    if imputer is None:
        if within_class == False:
            imputer = KNNImputer(n_neighbors=n_neighbors)
            imputer.set_output(transform="pandas")
            imputer.fit(df[columns[0:-1]])
            filled_df = imputer.fit_transform(df[columns[0:-1]])
        else:
            imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
            imputer.set_output(transform="pandas")
            imputer.fit(df[columns[0:-1]])
            filled_df = imputer.fit_transform(df[columns[0:-1]])
    else:
        filled_df = imputer.transform(df[columns])
    # print(df.iloc[:, 0])
    # filled_df = pd.concat([df.iloc[:, 0], filled_df], axis=1)
    return filled_df, imputer

def knn_imputation_within_class(df: pd.DataFrame, columns: List[str],
                                train_labels_filepath: str,
                                n_neighbors: int = 5,
                                df_imputed_train: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    # def knn_imputation_within_class(df, columns, labels_filepath, n_neighbors=5, df_imputed_train=None):
    """
    Perform KNN imputation within each class of entries, with class labels coming from a separate file.

    Parameters:
    - df: pandas DataFrame, the dataset with rows as entries and columns as features. The first column is 'filename'.
    - labels_filepath: str, the path to the labels file, which has filenames and class labels.
    - n_neighbors: int, number of neighbors to use for KNN imputation.

    Returns:
    - pandas DataFrame with imputed values, imputation performed within each class.
    """
    df[columns] = df[columns].replace(0, np.nan)

    # Initialize the KNNImputer
    imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')

    # Empty DataFrame to hold the imputed data, preserving the fullpath and class label columns
    imputed_data = pd.DataFrame(
        columns=['fullpath'] + list(df.columns[1:]) + ['GT'])

    if df_imputed_train is None:
        # Load and merge class labels
        df_with_labels = load_and_merge_labels(df, train_labels_filepath)

       # Separate features and class labels
        X = df_with_labels.drop(columns=['fullpath', 'GT'])
        y = df_with_labels['GT']

        # Iterate over each class, perform KNN imputation, and concatenate the results
        for GT in y.unique():
            # Subset the data for the current class
            subset_X = X[y == GT]
            # Preserve fullpaths for merging
            subset_fullpaths = df_with_labels['fullpath'][y == GT]

            # Perform KNN imputation on the subset
            imputed_subset = imputer.fit_transform(subset_X)

            # Convert the imputed numpy array back to a DataFrame
            imputed_subset_df = pd.DataFrame(imputed_subset, columns=X.columns)
            # Add fullpaths back
            imputed_subset_df.insert(0, 'fullpath', subset_fullpaths)
            # imputed_subset_df['fullpath'] = subset_fullpaths.values
            imputed_subset_df['GT'] = GT  # Add class labels back

            # Concatenate the imputed subset back to the full DataFrame
            imputed_data = pd.concat(
                [imputed_data, imputed_subset_df], ignore_index=True)
    else:
        imputer.fit(df_imputed_train[columns])
        imputed_data = imputer.transform(df[columns])
        imputed_data = pd.DataFrame(
            imputed_data, columns=columns)

    # Ensure the class label column is in the same type as in the original DataFrame
    if 'fullpath' not in imputed_data.columns:
        imputed_data.insert(0, 'fullpath', df['fullpath'])
    else:
        print("fullpath already exists in the DataFrame")
    # imputed_data['fullpath'] = df['fullpath']
    # imputed_data['GT'] = df['GT'].astype(y.dtype)

    return imputed_data

def create_intensity_feature_from_file(total_json_path, heart_json_path):
    # Load the JSON file
    with open(total_json_path) as file:
        total_data = json.load(file)

    with open(heart_json_path) as file:
        heart_data = json.load(file)

    # Define the keys
    keys = [
        "iliac_artery_left",
        "iliac_artery_right",
        "pulmonary_artery",
        "aorta",
        "kidney_left",
        "kidney_right",
        "heart_myocardium",
        "heart_atrium_left",
        "heart_atrium_right",
        "heart_ventricle_left",
        "heart_ventricle_right",
        "portal_vein_and_splenic_vein",
        "liver",
        "urinary_bladder",
        "inferior_vena_cava",
        "iliac_vena_left",
        "iliac_vena_right",
    ]

    # Get the value of the "intensity" subkey for each key
    intensity_values = {}
    for key in keys:
        if key in total_data:
            if "intensity" in total_data[key]:
                intensity_values[key] = total_data[key]['intensity']
            else:
                intensity_values[key] = None
        elif key in heart_data:
            if "intensity" in heart_data[key]:
                intensity_values[key] = heart_data[key]['intensity']
            else:
                intensity_values[key] = None
        else:
            continue
    # intensity_values = {key: data[key]["intensity"]
    #                     if "intensity" in data[key] else None for key in keys}

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(intensity_values, index=[0])
    return df