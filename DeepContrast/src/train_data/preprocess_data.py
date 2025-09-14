import glob
import shutil
import os
import pandas as pd
import nrrd
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
        save nrrd image data;
    """

    # reg_temp_img = os.path.join(data_dir, '1.2.392.200036.9116.2.5.1.37.2418751871.1574048534.842006.nrrd')
    # print(sorted(glob.glob(os.path.join(data_dir, '*', '*.nii.gz'))))
    fns = [fn for fn in sorted(glob.glob(os.path.join(data_dir, '*', '*.nii.gz')))]
    ## patient ID
    IDs = []
    pats = []
    for fn in fns:
        ID = fn.split('/')[-1].replace('.nii.gz', '').strip()
        if 'registered' in ID:
            IDs.append(ID.replace('registered_', ''))
            pats.append(fn)
    ## PMH dataframe
    df = pd.DataFrame({'ID': IDs, 'file': pats})
    print("DF",df[:10])
    # Recursively find all .nii.gz files in nested folders
    # fns = [fn for fn in sorted(glob.glob(os.path.join(data_dir, '*', '*.nii.gz')))]

    # Use the filename (without extension) as the ID
    # IDs = [os.path.splitext(os.path.basename(fn))[0].replace('.nii', '') for fn in fns]

    ## PMH dataframe
    reg_temp_img = pats[0]  # Use the first file as reference
    print(reg_temp_img)
    for fn, ID in zip(df['file'], df['ID']):
        print(ID)
        # respacing
        img_nrrd = respacing(
            nrrd_dir=fn,
            interp_type=interp_type,
            new_spacing=new_spacing,
            patient_id=ID,
            return_type='nrrd',
            save_dir=None)
        print("img_nrrd",img_nrrd)
        # registration
        img_reg = nrrd_reg_rigid_ref(
            img_nrrd=img_nrrd,
            fixed_img_dir=reg_temp_img,
            patient_id=ID,
            save_dir=pre_data_dir)
        # registration
       
        img_crop = crop_image(
            nrrd_file=img_reg,
            patient_id=ID,
            crop_shape=crop_shape,
            return_type='nrrd',
            save_dir=pre_data_dir)
        print('After respacing:', np.min(img_nrrd), np.max(img_nrrd))
        print('After registration:', np.min(img_reg), np.max(img_reg))
        # print('After cropping:', np.min(img_crop), np.max(img_crop))
