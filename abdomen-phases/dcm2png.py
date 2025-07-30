import pydicom
import cv2
import numpy as np
import nibabel as nib
import os

def extract_slices(nifti_path, output_dir, study_id, series_id, label):
    # Load NIfTI
    img = nib.load(nifti_path)
    data = img.get_fdata()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    csv_rows = []
    for i in range(data.shape[2]):
        slice_img = data[:, :, i]
        slice_img = ((slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-6) * 255).astype('uint8')
        slice_path = os.path.join(output_dir, f'slice_{i:04d}.png')
        cv2.imwrite(slice_path, slice_img)
        csv_rows.append({'image': slice_path, 'label': label, 'study': study_id, 'series': series_id})
    
    return csv_rows

def dicom_to_png(dicom_path, output_path):
    # Read DICOM file
    dicom = pydicom.dcmread(dicom_path)
    
    # Get pixel array and rescale to 0-255
    image = dicom.pixel_array
    image = image.astype(float)
    image = ((image - image.min()) * (255.0 / (image.max() - image.min()))).astype(np.uint8)
    
    # Save as PNG
    cv2.imwrite(output_path, image)

