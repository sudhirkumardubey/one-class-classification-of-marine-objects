import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join
from keras.utils import load_img, img_to_array 
import glob
from sklearn import svm, metrics
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd
from sklearn.metrics import f1_score

net = keras.models.load_model("/home/sudhir/SUDHIR/ROSTOCK_COURSE_WORK/SEMESTER_2/Software_Lab_Project/8_Jan/DROCC_fathom_RuningModel/saved_model/my_model.h5")

#restore the pretrained model's weights and variables from the checkpoint for the target class
ckpt = tf.train.Checkpoint(model=net)

#Path to weights need to be changed class wise
ckpt.restore(tf.train.latest_checkpoint('/home/sudhir/SUDHIR/ROSTOCK_COURSE_WORK/SEMESTER_2/Software_Lab_Project/8_Jan/DROCC_fathom_RuningModel/Old/fathomnetood2_s1_c4/raw')).expect_partial()

# change class name as per requirement
# train_img_paths ="./Dataset/006_Baseline_dataset_2/train/Paelopatides confundens"  #path of target class folder for ocsvm
# all_test_paths = "./Dataset/006_Baseline_dataset_2/test"
# all_val_paths ="./Dataset/006_Baseline_dataset_2/val"

train_img_paths ="/home/sudhir/SUDHIR/ROSTOCK_COURSE_WORK/SEMESTER_2/Software_Lab_Project/8_Jan/DROCC_fathom_RuningModel/Dataset/FathomNet_Dataset_Min_ImgCnt_Cropped_Train/Actiniidae sp 1"  #path of target class folder for ocsvm
all_test_paths = "/home/sudhir/SUDHIR/ROSTOCK_COURSE_WORK/SEMESTER_2/Software_Lab_Project/8_Jan/DROCC_fathom_RuningModel/Dataset/Test_model_11"
# all_val_paths ="./Dataset/006_Baseline_dataset_2/val"

category = os.path.split(train_img_paths)

image_size = 156

train_array = []
test_array = []
val_array = []

def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    for file in glob.iglob(f'{img_paths}/*'):
        imgs = load_img(file, target_size=(img_height, img_width))
        img_array = img_to_array(imgs)
        train_array.append(img_array)
    
    return train_array


def read_and_prep_images_test(img_paths, img_height=image_size, img_width=image_size):
    for file in glob.iglob(f'{img_paths}/*'):        
        for img in os.listdir(file):
            imgs = load_img(os.path.join(file, img), target_size=(img_height, img_width))
            img_array = np.array([img_to_array(imgs)])
            test_array.append(img_array)
    
    print("Done")
    return test_array

def read_and_prep_images_val(img_paths, img_height=image_size, img_width=image_size):
    for file in glob.iglob(f'{img_paths}/*'):        
        for img in os.listdir(file):
            imgs = load_img(os.path.join(file, img), target_size=(img_height, img_width))
            img_array = np.array([img_to_array(imgs)])
            val_array.append(img_array)
    
    print("Done")
    return val_array

labels = [1 if folder == "{}".format(category[-1]) else -1 for folder in os.listdir(all_test_paths) for files in glob.iglob(os.path.join(all_test_paths, f'{folder}/*'))]
labels = np.array(labels)



X_train = read_and_prep_images(train_img_paths)
X_test = read_and_prep_images_test(all_test_paths)
# X_val = read_and_prep_images_val(all_val_paths)

print(len(X_train))
print(len(X_test))

# generate ResNet representations for train/test/val images which will be given to ocsvm for prediction

pred1 = net.predict(np.array(X_train).squeeze())
pred2 = net.predict(np.array(X_test).squeeze())
im = pred2['embeds']

# #linear OC-SVM
oc_svm_clf = svm.OneClassSVM(kernel='linear', gamma='auto').fit(pred1['embeds'])

# #Generate predicted class (-1 or 1)
oc_svm_preds = oc_svm_clf.predict(im) 
#print('prediction', oc_svm_preds.T)

# #generate confusion matrix for test set
tn, fp, fn, tp= metrics.confusion_matrix(labels, oc_svm_preds, labels=[-1,1]).ravel()
f = f1_score(labels, oc_svm_preds, labels=[-1,1])
# # print(f)
print(tn, ',', fp, ',',fn, ',',tp)
classification_accuracy = (tp + tn) / float(tp + tn + fp + fn)
print('classification_accuracy-Deep OCC',100*classification_accuracy)
ConfusionMatrixDisplay.from_predictions(labels, oc_svm_preds, labels=[-1,1])
plt.show()
