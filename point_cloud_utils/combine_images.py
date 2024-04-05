## Code for combining the test and val images together
## Run Colmap on the combined images
# We will get a point cloud on all the images. We save this point cloud
# We then run the script to change the images.bin, cameras.bin and point3D into the new NeRF compatible frame.

import os
import shutil

basedir = '../nerf_synthetic/chair/'
# Define the source and destination directories
source_dirs = [basedir + 'train', basedir + 'val']
destination_dir = basedir + 'combined_data'

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Iterate through each source directory
for source_dir in source_dirs:
    # Get the list of files in the source directory
    files = os.listdir(source_dir)
    
    # Iterate through each file in the source directory
    for file in files:
        # Construct the full path of the file
        file_path = os.path.join(source_dir, file)
        
        # Get the file extension
        _, extension = os.path.splitext(file)
        
        # Construct the new file name with prefix
        new_file_name = f"{source_dir}_{file}"
        
        # Construct the full path of the destination file
        destination_file_path = os.path.join(destination_dir, new_file_name)
        
        # Copy the file to the destination directory with the new name
        shutil.copy(file_path, destination_file_path)