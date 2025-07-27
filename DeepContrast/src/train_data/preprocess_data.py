import glob
import shutil
import os
import pandas as pd
#import nrrd  # No longer needed for NIfTI
import re
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from time import gmtime, strftime
from datetime import datetime
import timeit
from utils.respacing import respacing
from utils.nrrd_reg import nrrd_reg_rigid_ref
from utils.crop_image import crop_image
from utils.resize_3d import resize_3d
from utils.crop_image import crop_image
import SimpleITK as sitk


def preprocess_data(data_dir, pre_data_dir, new_spacing, data_exclude=None,
                    crop_shape=[192, 192, 10], interp_type='linear'):
    """
    Preprocess data including: respacing, registration, cropping;

    Args:
        data_dir {path} -- path to CT data;
        out_dir {path} -- path to result outputs;
    Keyword args:
        new_spacing {tuple} -- respacing size;
        return_type {str} -- image data format after preprocessing, default: 'nrrd';
        data_exclude {str} -- exclude patient data due to data issue, default: None;
        crop_shape {np.array} -- numpy array size afer cropping;
        interp_type {str} -- interpolation type for respacing, default: 'linear';
    Return:
        save nifti image data;
    """

    # Recursively find all .nii.gz files in nested folders
    fns = [fn for fn in sorted(glob.glob(os.path.join(data_dir, '*', '*.nii.gz')))]
    # If you also want to support .nii (uncompressed), add:
    fns += [fn for fn in sorted(glob.glob(os.path.join(data_dir, '*', '*.nii')))]

    # Use the filename (without extension) as the ID
    IDs = [os.path.splitext(os.path.basename(fn))[0].replace('.nii', '') for fn in fns]

    # PMH dataframe
    df = pd.DataFrame({'ID': IDs, 'file': fns})
    for fn, ID in zip(df['file'], df['ID']):
        print(ID)
        # respacing
        img_nifti = respacing(
            nrrd_dir=fn,  # param name kept for compatibility, but it's now a NIfTI file
            interp_type=interp_type,
            new_spacing=new_spacing,
            patient_id=ID,
            return_type='nifti',  # change to 'nifti' for clarity
            save_dir=None)
        # registration
        # For registration, you need a fixed image. You may want to select a reference NIfTI from your dataset.
        # For now, we'll skip registration or use the first file as reference.
        reg_temp_img = fns[0]  # Use the first file as reference
        img_reg = nrrd_reg_rigid_ref(
            img_nrrd=img_nifti,
            fixed_img_dir=reg_temp_img,
            patient_id=ID,
            save_dir=None)
        # img_reg = img_nifti  # If skipping registration for now
        # crop image
        img_crop = crop_image(
            nrrd_file=img_reg,
            patient_id=ID,
            crop_shape=crop_shape,
            return_type='nifti',
            save_dir=pre_data_dir)


