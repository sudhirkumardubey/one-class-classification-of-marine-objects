import numpy as np
import os

from os.path import join, exists
from PIL import ImageFile

import tensorflow

from keras.utils import load_img, img_to_array
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, classification_report, confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import re
import glob
from csv import writer
import json
from sklearn.metrics import f1_score

#path of target class folder for ocsvm
# train_img_paths ="/home/sudhir/SUDHIR/ROSTOCK_COURSE_WORK/SEMESTER_2/Software_Lab_Project/8_Jan/DROCC_fathom_RuningModel/Dataset/Baseline_data_2/train/Actiniaria"  

#path to test and val folder
# all_test_paths = "/home/sudhir/SUDHIR/ROSTOCK_COURSE_WORK/SEMESTER_2/Software_Lab_Project/8_Jan/DROCC_fathom_RuningModel/Dataset/Baseline_data_2/test"
# all_val_paths ="./Dataset/006_Baseline_dataset_2/val"
train_img_paths ="/home/sudhir/SUDHIR/ROSTOCK_COURSE_WORK/SEMESTER_2/Software_Lab_Project/8_Jan/DROCC_fathom_RuningModel/Dataset/FathomNet_Dataset_Min_ImgCnt_Cropped_Train/Actiniidae sp 1"  #path of target class folder for ocsvm
all_test_paths = "/home/sudhir/SUDHIR/ROSTOCK_COURSE_WORK/SEMESTER_2/Software_Lab_Project/8_Jan/DROCC_fathom_RuningModel/Dataset/Test_model_11"


# output_path1 = ".Output/Baseline1_Auc"
# output_path2 = ".Output/Baseline2_Auc"
# output_path3 = ".Output/Baseline3_Auc"

#output_path4 = ".Output/Baseline4_Auc"
image_size = 156

category = os.path.split(train_img_paths)
#print("{}".format(category[-1]))

train_array = []
test_array =[]
val_array = []

def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    for file in glob.iglob(f'{img_paths}/*'):
        imgs = load_img(file, target_size=(img_height, img_width))
        img_array = np.array([img_to_array(imgs)])
        train_array.append(img_array)
    
    return train_array


def read_and_prep_images_test(img_paths, img_height=image_size, img_width=image_size):
    for file in glob.iglob(f'{img_paths}/*'):
        
        for img in os.listdir(file):
            imgs = load_img(os.path.join(file, img), target_size=(img_height, img_width))
            img_array = np.array([img_to_array(imgs)])
            test_array.append(img_array)
    
    return test_array

def read_and_prep_images_val(img_paths, img_height=image_size, img_width=image_size):
    for file in glob.iglob(f'{img_paths}/*'):
        
        for img in os.listdir(file):
            imgs = load_img(os.path.join(file, img), target_size=(img_height, img_width))
            img_array = np.array([img_to_array(imgs)])
            val_array.append(img_array)
    
    return val_array


X_train = read_and_prep_images(train_img_paths)
X_test = read_and_prep_images_test(all_test_paths)
# X_val = read_and_prep_images_val(all_val_paths)

X_train = np.array(X_train).reshape((len(X_train), 3*image_size **2))
X_test = np.array(X_test).reshape((len(X_test), 3*image_size **2))
# X_val = np.array(X_val).reshape((len(X_val), 3*image_size **2))
#print(X_test.shape)



labels = [1 if folder == "{}".format(category[-1]) else -1 for folder in os.listdir(all_test_paths) for files in glob.iglob(os.path.join(all_test_paths, f'{folder}/*'))]
labels = np.array(labels) # y_true

oc_svm_clf = svm.OneClassSVM(kernel='linear', gamma='auto').fit(X_train)
#print("oc_svm_clf  and fit done")
score = oc_svm_clf.score_samples(X_test)

roc = roc_auc_score(labels, score)

oc_svm_preds = oc_svm_clf.predict(X_test) # y_pred
f = f1_score(labels, oc_svm_preds, labels=[-1,1])
print(f)


# Uncomment these lines to make .json files
dictionary = {"linear roc auc score :": roc*100}
print(dictionary)
# json_object = json.dumps(dictionary)

# if os.path.exists(output_path4):
#     pass
# else:
#     os.mkdir(output_path4)

# # change jason file name for each category

# with open("./Baseline_data_4_9.json", "w") as outfile:
#     outfile.write(json_object)

tn, fp, fn, tp= confusion_matrix(labels, oc_svm_preds, labels=[-1,1]).ravel()
print(tn, ',', fp, ',',fn, ',',tp)
classification_accuracy = (tp + tn) / float(tp + tn + fp + fn)
print('classification_accuracy-OCSVM',100*classification_accuracy)

ConfusionMatrixDisplay.from_predictions(labels, oc_svm_preds, labels=[-1,1])
plt.show()
