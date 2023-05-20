import shutil
import os
import numpy as np
import argparse

"""
To split data in tain, test and val folder
"""


import splitfolders #split_folders # or import splitfolders

input_folder1 = "./003_FathomNet_Dataset_Min_ImgCnt_Train/"
input_folder2 = "./004_FathomNet_Dataset_Min_ImgCnt_Cropped_Train/"
input_folder3 = "./005_FathomNet_Dataset_Augmented_Train/"
input_folder4 = "./005_FathomNet_Dataset_Cropped_Augmented_Train/"

Baseline_data_1 = "./006_Baseline_dataset_1" #where you want the split datasets saved. one will be created if it does not exist or none is set
Baseline_data_2 = "./006_Baseline_dataset_2" #where you want the split datasets saved. one will be created if it does not exist or none is set
Baseline_data_3 = "./006_Baseline_dataset_3" #where you want the split datasets saved. one will be created if it does not exist or none is set
Baseline_data_4 = "./006_Baseline_dataset_4" #where you want the split datasets saved. one will be created if it does not exist or none is set

splitfolders.ratio(input_folder1, output=Baseline_data_1, seed=42, ratio=(.8, .1, .1)) # ratio of split are in order of train/val/test
splitfolders.ratio(input_folder2, output=Baseline_data_2, seed=42, ratio=(.8, .1, .1)) # ratio of split are in order of train/val/test
splitfolders.ratio(input_folder3, output=Baseline_data_3, seed=42, ratio=(.8, .1, .1)) # ratio of split are in order of train/val/test
splitfolders.ratio(input_folder4, output=Baseline_data_4, seed=42, ratio=(.8, .1, .1)) # ratio of split are in order of train/val/test
