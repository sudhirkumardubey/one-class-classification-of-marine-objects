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
 
# run Augment.py it will make folder our 1(a) case folder with name '005_FathomNet_Dataset_Augmented_Train' and 1(b) case folder with name
#                      '005_FathomNet_Dataset_Cropped_Augmented_Train'
python3 Augment.py


