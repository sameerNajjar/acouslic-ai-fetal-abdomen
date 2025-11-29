import os
import SimpleITK as sitk
import numpy as np
import pandas as pd


images_folder = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/acouslic-ai-train-set/images/stacked_fetal_ultrasound'
masks_folder = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/acouslic-ai-train-set/masks/stacked_fetal_abdomen'

# List all MHA files (assuming the files are .mha format)
mha_files = [f for f in os.listdir(images_folder) if f.endswith('.mha')]

# Initialize a list to store labels
labels = []

# Function to load the MHA file (ultrasound scan)
def load_mha_file(file_path):
    image = sitk.ReadImage(file_path)
    image_array = sitk.GetArrayFromImage(image)
    return image_array

# Function to load the corresponding mask
def load_mask(mask_path):
    mask = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(mask)
    return mask_array

# Loop through each MHA file and process the frames and masks
for mha_file in mha_files:
    # Load the ultrasound volume (3D image: [frames, height, width])
    image_array = load_mha_file(os.path.join(images_folder, mha_file))

    # Load the corresponding mask (assuming mask is also in .mha format)
    mask_file = mha_file  # Assuming mask has the same name as the image file
    mask_array = load_mask(os.path.join(masks_folder, mask_file))

    # Get the number of frames (assumed as the first dimension)
    num_frames = image_array.shape[0]  # Number of frames

    # Iterate through each frame
    for frame_idx in range(num_frames):
        # Get the corresponding mask frame (2D mask for segmentation)
        mask_frame = mask_array[frame_idx, :, :]

        # Assign label based on mask contents
        if np.any(mask_frame == 2):
            label = 2
        elif np.any(mask_frame == 1):
            label = 1
        else:
            label = 0

        # Append the filename, frame, and label
        labels.append((mha_file, frame_idx, label))

# Define output directory
output_dir = "/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output"

# Check if the directory exists, if not, create it
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the labels to a CSV file for future use
labels_df = pd.DataFrame(labels, columns=["Filename", "Frame", "Label"])
labels_df.to_csv(os.path.join(output_dir, 'labels.csv'), index=False)

print("3-class labeling completed and saved to 'labels.csv'")
