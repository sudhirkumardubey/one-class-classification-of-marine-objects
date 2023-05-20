# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/bin/bash

# Fathomnet dataset:

#In our study we want to see how well Fathomnet dataset work for deep learning one class classification problem. 

# <----------- Introduction of Dataset types --------------> Refer Report for more information
# Following test cases are created for analysis with respect to dataset. 
#    1. Training model with Fathomnet data that have been randomly downloaded.
#	(a) Fathomnet random dataset with augmentation
#	(b) Fathomnet random dataset with cropped bounding boxes and augmentation performed on them

#    2. Training model with Fathomnet data that have been randomly downloaded with minimum x images per concept. (x = 100, for our setup)
#	(a) Fathomnet random dataset with minimum 100 images
#	(b) Fathomnet random dataset with cropped bounding boxes


# Run DownloadData_Min_ImgCnt.py to download from FathomNet dataset. It will create '002_FathomNet_Dataset_Min_ImgCnt' folder in the present directory. Minimum number of images per concept will be 100
python3 DownloadData_Min_ImgCnt.py
 

