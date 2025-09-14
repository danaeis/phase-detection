import glob
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def get_pat_dataset(data_dir, pre_data_dir, label_dir, label_file, pro_data_dir):
    """
    Traverse pre_data_dir for NIfTI files, extract ID, join with label_file on ID, and split.
    """
    # print(data_dir, pre_data_dir, label_dir, label_file, pro_data_dir)
    # # Recursively find all .nii.gz and .nii files
    # fns = [fn for fn in sorted(glob.glob(os.path.join(pre_data_dir, '*.nii.gz'), recursive=True))]
    # fns += [fn for fn in sorted(glob.glob(os.path.join(pre_data_dir, '*.nii'), recursive=True))]
    # # Extract ID from filename (without extension)
    # file_df = pd.DataFrame({
    #     'file': fns,
    #     'ID': [os.path.splitext(os.path.basename(fn))[0].replace('.nii', '') for fn in fns]
    # })
    # # Load label file (must have ID, Contrast, phase columns)
    # label_df = pd.read_csv(os.path.join(label_dir, label_file))
    # # Merge on ID
    # merged = pd.merge(file_df, label_df, on='ID', how='inner')
    # print('Merged shape:', merged.shape)
    # # Check required columns
    # assert 'Contrast' in merged.columns, 'Contrast column missing in label file!'
    # assert 'phase' in merged.columns, 'phase column missing in label file!'
    # # Train/val/test split
    # data = merged['file']
    # label = merged['phase']
    # ID = merged['ID']
    # data_train_, data_test, label_train_, label_test, ID_train_, ID_test = train_test_split(
    #     data, label, ID, stratify=label, test_size=0.3, random_state=42)
    # data_train, data_val, label_train, label_val, ID_train, ID_val = train_test_split(
    #     data_train_, label_train_, ID_train_, stratify=label_train_, test_size=0.3, random_state=42)
    # # Save DataFrames
    # train_pat_df = pd.DataFrame({'ID': ID_train, 'file': data_train, 'label': label_train})
    # val_pat_df = pd.DataFrame({'ID': ID_val, 'file': data_val, 'label': label_val})
    # test_pat_df = pd.DataFrame({'ID': ID_test, 'file': data_test, 'label': label_test})
    # train_pat_df.to_csv(os.path.join(pro_data_dir, 'train_pat_df.csv'))
    # val_pat_df.to_csv(os.path.join(pro_data_dir, 'val_pat_df.csv'))
    # test_pat_df.to_csv(os.path.join(pro_data_dir, 'test_pat_df.csv'))
    # # Return lists for each split
    # data_tot = [data_train, data_val, data_test]
    # label_tot = [label_train, label_val, label_test]
    # ID_tot = [ID_train, ID_val, ID_test]
    # return data_tot, label_tot, ID_tot

    fns = [fn for fn in sorted(glob.glob(pre_data_dir + '/*nrrd'))]
    file_df = pd.DataFrame({
        'file': fns,
        'ID': [os.path.splitext(os.path.basename(fn))[0].replace('.nrrd', '') for fn in fns]
    })
    # Load label file (must have ID, Contrast, phase columns)
    label_df = pd.read_csv(os.path.join(label_dir, label_file))
    # Merge on ID
    merged = pd.merge(file_df, label_df, on='ID', how='inner')
    print('Merged shape:', merged.shape)
    ## labels
    # df_label = pd.read_csv(os.path.join(label_dir, label_file))
    # df_label['Contrast'] = df_label['Contrast'].map({'Yes': 1, 'No': 0})
    # labels = df_label['Contrast'].to_list()
    ## data
    # fns = [fn for fn in sorted(glob.glob(pre_data_dir + '/*nrrd'))]
    ## patient ID
    # IDs = []
    # for fn in fns:
    #     ID = fn.split('/')[-1].split('.')[0].strip()
    #     IDs.append(ID)

    ## create dataframe
    print('ID:', len(merged))
    print('file:', len(fns))
    print('label:', len(label_df))
    df = pd.DataFrame({'ID': merged['ID'], 'file': merged['file'], 'label': merged['Contrast']})
    data = df['file']
    label = df['label']
    ID = df['ID']
    data_train_, data_test, label_train_, label_test, ID_train_, ID_test = train_test_split(
        data,
        label,
        ID,
        stratify=label,
        test_size=0.3,
        random_state=42)
    data_train, data_val, label_train, label_val, ID_train, ID_val = train_test_split(
        data_train_,
        label_train_,
        ID_train_,
        stratify=label_train_,
        test_size=0.3,
        random_state=42)
    ## save train, val, test df on patient level
    train_pat_df = pd.DataFrame({'ID': ID_train, 'file': data_train, 'label': label_train})
    val_pat_df = pd.DataFrame({'ID': ID_val, 'file': data_val, 'label': label_val})
    test_pat_df = pd.DataFrame({'ID': ID_test, 'file': data_test, 'label': label_test})
    train_pat_df.to_csv(os.path.join(pro_data_dir, 'train_pat_df.csv'))
    val_pat_df.to_csv(os.path.join(pro_data_dir, 'val_pat_df.csv'))
    test_pat_df.to_csv(os.path.join(pro_data_dir, 'test_pat_df.csv'))
    ## save data, label and ID as list
    data_tot = [data_train, data_val, data_test]
    label_tot = [label_train, label_val, label_test]
    ID_tot = [ID_train, ID_val, ID_test]

    return data_tot, label_tot, ID_tot