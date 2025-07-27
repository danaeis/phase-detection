import os
import itertools
import operator
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import nrrd
from PIL import Image
from totalsegmentator.python_api import totalsegmentator
import tempfile


#--------------------------------------------------------------------------------------
# crop image
#-------------------------------------------------------------------------------------
def crop_image(nrrd_file, patient_id, crop_shape, return_type, save_dir, region='abdomen', mode='final'):
    
    img_sitk = nrrd_file
    img_arr = sitk.GetArrayFromImage(img_sitk)
    c, y, x = img_arr.shape
    
    if mode == 'roi':
        # Save to temporary nifti for TotalSegmentator
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = os.path.join(tmp_dir, 'input.nii.gz')
            sitk.WriteImage(img_sitk, input_path)
            output_path = os.path.join(tmp_dir, 'segmentations.nii.gz')
            
            # Run TotalSegmentator
            totalsegmentator(input_path, output_path, ml=True, task='total')
            
            # Load segmentation
            seg_sitk = sitk.ReadImage(output_path)
            seg_arr = sitk.GetArrayFromImage(seg_sitk)
        
        # Accurate class indices from TotalSegmentator (based on standard mapping)
        # Full map abbreviated; key regions:
        relevant_labels = {
            'abdomen': [1,2,3,4,5,6,7,8,9,10,11,12,18,19,20,21,22,23,24],  # spleen, kidneys, gallbladder, liver, stomach, pancreas, adrenals, veins, bowel, duodenum, colon, bladder, prostate, cysts
            'chest': [13,14,15,16,17,18,19,20,21,22,50,51,52],  # lungs (5 classes), esophagus, trachea, heart, aorta, pulmonary artery (adjust as per exact indices)
            'head_neck': [17,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,57]  # thyroid, vertebrae C1-C7 and others, face, etc.
        }.get(region, [])
        if not relevant_labels:
            raise ValueError('Unknown region')
        
        # Create mask
        mask = np.isin(seg_arr, relevant_labels)
        
        # Find bounding box
        z, yy, xx = np.where(mask)
        if len(z) == 0:
            raise ValueError('No relevant segmentation found')
        min_z, max_z = z.min(), z.max()
        min_y, max_y = yy.min(), yy.max()
        min_x, max_x = xx.min(), xx.max()
        
        # Add padding (20% for safety)
        pad = 0.2
        dz = int((max_z - min_z) * pad)
        dy = int((max_y - min_y) * pad)
        dx = int((max_x - min_x) * pad)
        startz = max(0, min_z - dz)
        endz = min(c, max_z + dz + 1)
        starty = max(0, min_y - dy)
        endy = min(y, max_y + dy + 1)
        startx = max(0, min_x - dx)
        endx = min(x, max_x + dx + 1)
        
        # Crop
        img_crop_arr = img_arr[startz:endz, starty:endy, startx:endx]
        
    elif mode == 'final':
        # Original center of mass based crop to exact shape
        mask_arr = np.copy(img_arr)
        mask_arr[mask_arr > -500] = 1
        mask_arr[mask_arr <= -500] = 0
        centermass = ndimage.center_of_mass(mask_arr)
        cpoint = c - crop_shape[2]//2
        centermass = ndimage.center_of_mass(mask_arr[cpoint, :, :])
        startx = int(centermass[0] - crop_shape[0]//2)
        starty = int(centermass[1] - crop_shape[1]//2)
        startz = int(c - crop_shape[2])
        
        if startz < 0:
            img_crop_arr = np.pad(img_arr, ((abs(startz)//2, abs(startz)//2), (0, 0), (0, 0)), 'constant', constant_values=-1024)
            img_crop_arr = img_arr[0:crop_shape[2], starty:starty + crop_shape[1], startx:startx + crop_shape[0]]
        else:
            img_crop_arr = img_arr[startz:startz + crop_shape[2], starty:starty + crop_shape[1], startx:startx + crop_shape[0]]
        
        if img_crop_arr.shape[0] < crop_shape[2]:
            img_crop_arr = np.pad(img_crop_arr, ((crop_shape[2] - img_crop_arr.shape[0], 0), (0, 0), (0, 0)), 'constant', constant_values=-1024)
    else:
        raise ValueError('Invalid mode')
    
    # For roi mode, we don't resize yet; for final, it's already to shape
    if mode == 'roi':
        # Optionally pad if needed, but keep original size for registration
        pass
    
    img_crop_nifti = sitk.GetImageFromArray(img_crop_arr)
    img_crop_nifti.SetSpacing(img_sitk.GetSpacing())
    img_crop_nifti.SetOrigin(img_sitk.GetOrigin())

    if save_dir is not None:
        fn = str(patient_id) + '_' + mode + '.nii.gz'
        writer = sitk.ImageFileWriter()
        writer.SetFileName(os.path.join(save_dir, fn))
        writer.SetUseCompression(True)
        writer.Execute(img_crop_nifti)

    if return_type == 'nifti':
        return img_crop_nifti
    elif return_type == 'npy':
        return img_crop_arr
 
#-----------------------------------------------------------------------
# run codes to test
#-----------------------------------------------------------------------
if __name__ == '__main__':

#    output_dir = '/media/bhkann/HN_RES1/HN_CONTRAST/output'
    output_dir = '/mnt/aertslab/USERS/Zezhong/constrast_detection/test'
    file_dir = '/media/bhkann/HN_RES1/HN_CONTRAST/0_image_raw_PMH'
    test_file = 'PMH_OPC-00050_CT-SIM_raw_raw_raw_xx.nrrd'
    return_type = 'sitk'
    crop_shape = [192, 192, 110]

    train_file = os.path.join(file_dir, test_file)

    img_crop = crop_image(
        nrrd_file=train_file,
        crop_shape=crop_shape,
        return_type=return_type,
        output_dir=output_dir
        )
    print('crop arr shape:', img_crop)
#    arr_crop = image_arr_crop[0, :, :]
#    img_dir = os.path.join(output_dir, 'arr_crop.jpg')
#    plt.imsave(img_dir, arr_crop, cmap='gray')
    print('successfully save image!!!')
    

