import os
import xml.etree.ElementTree as ET
import torch
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import cv2
import glob

#Path to Image folder
Path1="./002_FathomNet_Dataset_Min_ImgCnt"
Path2="./001_FathomNet_Dataset"

#Path to Target folder
Target1 = "./004_FathomNet_Dataset_Min_ImgCnt_Cropped_Train"
Target2 = "./004_FathomNet_Dataset_Cropped"

"""
27dec
Cropping image for a given species with given Bounding box
"""

def Mutilple_bbx(Path, TargetPath, species):

    for files in os.listdir(Path):
        if files.endswith('.png') or files.endswith('.jpg'):
            img = Image.open(os.path.join(Path,files))
            split_tuple = os.path.splitext(files)
            annotation_filename = os.path.join(split_tuple[0]+'.xml')
            # print(annotation_filename)
            
             # Get bounding box
            tree = ET.parse(os.path.join(Path,annotation_filename))
            root = tree.getroot()
            boxes = list()
            target = ".//*[name='{}']".format(species)
            for o in root.findall(target):
                bndbox = o.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                if(xmin<xmax and ymin < ymax):
                    bbox = (xmin, ymin, xmax, ymax)
                    boxes.append(bbox)
                elif(xmin<xmax and ymin > ymax):
                    bbox = (xmin, ymax, xmax, ymin)
                    boxes.append(bbox)
                elif(xmin>xmax and ymin < ymax):
                    bbox = (xmax, ymin, xmin, ymax)
                    boxes.append(bbox)
                else:
                    bbox = (xmax, ymax, xmin, ymin)
                    boxes.append(bbox)  

            # print("Number of bounding boxes for given species : ",len(boxes))
            # Crop image and save in Target folder
            for i in range(len(boxes)):
                img2 = img.crop(boxes[i])
                name = str(split_tuple[0])+ "_croped_"+ str(i) + "_.png"
                img2.save(TargetPath+name)


# End of function

for folder in os.listdir(Path1):
    P = os.path.join(f'{Path1}', folder)
    directory = "{}".format(folder) # Also gets target species
    T = os.path.join(Target1, directory)
    os.makedirs(T)
    Mutilple_bbx(P,f'{T}/',folder)

for folder in os.listdir(Path2):
    P = os.path.join(f'{Path2}', folder)
    directory = "{}".format(folder) # Also gets target species
    T = os.path.join(Target2, directory)
    os.makedirs(T)
    Mutilple_bbx(P,f'{T}/',folder)

print("Finished cropping")
