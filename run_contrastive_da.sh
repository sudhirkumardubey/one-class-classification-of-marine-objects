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
# Contrastive learning with distribution augmentation
# DATA in {cifar10ood, fathomnetood}
# CATEGORY in
#   {0..9..1} for cifar10ood
#   {0..9..1} for fathomnetood

# Path to different data sets

#   './Dataset/003_FathomNet_Dataset_Min_ImgCnt_Train/' (Randomly downloaded images with condition Per class images > 100, no prior augmentation)
# DATA = 'fathomnetood1'

#   './Dataset/004_FathomNet_Dataset_Min_ImgCnt_Cropped_Train/' (Cropped bounding box of class for Randomly downloaded images with condition per class images >100, no prior augmentation)
# DATA = 'fathomnetood2'

#   './Dataset/005_FathomNet_Dataset_Augmented_Train/' (Random downloaded images, no prior condition, here images in a class is minimum 1, no prior augmentation)
# DATA = 'fathomnetood3'

#   './Dataset/005_FathomNet_Dataset_Cropped_Augmented_Train/' (Random downloaded images, no prior condition, here images in a class is minimum 1, Augmentation performed on images to increase data set)

# DATA = 'fathomnetood4'

# For Cifar please check README file
DATA=fathomnetood2
METHOD=Contrastive
SEED=1
CATEGORY=0
DISTAUG_TYPE in {1,2,3,4,5,6}
python3 train_and_eval_loop.py \
  --method=${METHOD} \
  --file_path="${DATA}_${PREFIX}_s${SEED}_c${CATEGORY}" \
  --dataset=${DATA} \
  --category=${CATEGORY} \
  --seed=${SEED} \
  --root='./Dataset/004_FathomNet_Dataset_Min_ImgCnt_Cropped_Train/' \
  --net_type=ResNet18 \
  --net_width=1 \
  --latent_dim=0 \
  --aug_list="cnr0.5+hflip+jitter_b0.4_c0.4_s0.4_h0.4+gray0.2+blur0.5,+" \
  --aug_list_for_test="x" \
  --input_shape="64,64,3" \
  --optim_type=sgd \
  --sched_type=cos \
  --learning_rate=0.01 \
  --momentum=0.9 \
  --weight_decay=0.0003 \
  --num_epoch=1 \
  --batch_size=32 \
  --temperature=0.2 \
  --distaug_type "${DISTAUG_TYPE}"

 
 

  


