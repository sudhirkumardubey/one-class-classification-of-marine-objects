import os
import shutil
import glob

# Path to the directory with the files you want to copy
source_dir = './002_FathomNet_Dataset_Min_ImgCnt/'

# Path to the directory you want to copy the files to
destination_dir = "003_FathomNet_Dataset_Min_ImgCnt_Train"
os.mkdir(destination_dir)

# Iterate through all the files in the source directory
for folder in os.listdir(source_dir):
  # Check if the file is a png or jpg
  src = os.path.join(f'{source_dir}', folder)
  destination_path = os.path.join(destination_dir, folder)
  os.mkdir(destination_path)
  for file in glob.iglob(f'{src}/*'):
    if file.endswith('.png') or file.endswith('.jpg'):  
        shutil.copy(file, destination_path)
    
