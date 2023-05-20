# Imports
import imgaug as ia
ia.seed(1)
# imgaug uses matplotlib backend for displaying images
import matplotlib
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa 
# imageio library will be used for image input/output
import imageio
import pandas as pd
import numpy as np
import re
import os
import glob
# this library is needed to read XML files for converting it into CSV
import xml.etree.ElementTree as ET
import shutil


#Path to cropped images, To be created before
Path1 = "./001_FathomNet_Dataset/"
Path2 = "./004_FathomNet_Dataset_Cropped/"

#Path to save resiged and augmented images, To be created before
Target1 = "./005_FathomNet_Dataset_Augmented_Train"
Target2 = "./005_FathomNet_Dataset_Cropped_Augmented_Train"


def image(Path,Target):
    for file in os.listdir(Path):
        if file.endswith('.png') or file.endswith('.jpg'):
            image_main = imageio.imread(os.path.join(Path,file))
            resize = iaa.Sequential([ iaa.Resize({"height": 512, "width": 512})])
            image = resize(image=image_main)
            imageio.imwrite(f'{Target}/'+'Resized_'+file, image)

        #Rotate image by -5 degree
            rotate1 = iaa.Affine(rotate=(-5))
            image_aug = rotate1(image=image)
            imageio.imwrite(f'{Target}/'+'Rotate_-5_'+file, image_aug)
            
        #Rotate image by -10 degree
            rotate2 = iaa.Affine(rotate=(-10))
            image_aug = rotate2(image=image)
            imageio.imwrite(f'{Target}/'+'Rotate_-10_'+file, image_aug)

        #Rotate image by -15 degree
            rotate3 = iaa.Affine(rotate=(-15))
            image_aug = rotate3(image=image)
            imageio.imwrite(f'{Target}/'+'Rotate_-15_'+file, image_aug)
            
        #Rotate image by -20 degree
            rotate4 = iaa.Affine(rotate=(-20))
            image_aug = rotate4(image=image)
            imageio.imwrite(f'{Target}/'+'Rotate_-20_'+file, image_aug)
            
        #Rotate image by 5 degree
            rotate5 = iaa.Affine(rotate=(5))
            image_aug = rotate5(image=image)
            imageio.imwrite(f'{Target}/'+'Rotate_+5_'+file, image_aug)
            
        #Rotate image by 10 degree
            rotate6 = iaa.Affine(rotate=(10))
            image_aug = rotate6(image=image)
            imageio.imwrite(f'{Target}/'+'Rotate_+10_'+file, image_aug)

        #Rotate image by 15 degree
            rotate7 = iaa.Affine(rotate=(15))
            image_aug = rotate7(image=image)
            imageio.imwrite(f'{Target}/'+'Rotate_+15_'+file, image_aug)
            
        #Rotate image by 20 degree
            rotate8 = iaa.Affine(rotate=(20))
            image_aug = rotate8(image=image)
            imageio.imwrite(f'{Target}/'+'Rotate_+20_'+file, image_aug)

        #Rotate image by -45 degree
            rotate9 = iaa.Affine(rotate=(-45))
            image_aug = rotate9(image=image)
            imageio.imwrite(f'{Target}/'+'Rotate_-45_'+file, image_aug)
            
        #Rotate image by +45 degree
            rotate10 = iaa.Affine(rotate=(45))
            image_aug = rotate10(image=image)
            imageio.imwrite(f'{Target}/'+'Rotate_+45_'+file, image_aug)

        #Rotate image by -90 degree
            rotate11 = iaa.Affine(rotate=(-90))
            image_aug = rotate11(image=image)
            imageio.imwrite(f'{Target}/'+'Rotate_-90_'+file, image_aug)
            
        #Rotate image by 180 degree
            rotate12 = iaa.Affine(rotate=(180))
            image_aug = rotate12(image=image)
            imageio.imwrite(f'{Target}/'+'Rotate_180_'+file, image_aug)

        #Gaussian blur
            gaussian = iaa.GaussianBlur(sigma=(1.0,3.0))
            image_aug = gaussian(image=image)
            imageio.imwrite(f'{Target}/'+'GaussianBlurr_'+file, image_aug)

        #AdditiveGaussianNoise(scale=(10, 60))
            GaussianNoise1 = iaa.AdditiveGaussianNoise(scale=(20))
            image_aug = GaussianNoise1(image=image)
            imageio.imwrite(f'{Target}/'+'GaussianNoise_Scale20_'+file, image_aug)
            
        #AdditiveGaussianNoise(scale=(10, 60))
            GaussianNoise2 = iaa.AdditiveGaussianNoise(scale=(40))
            image_aug = GaussianNoise2(image=image)
            imageio.imwrite(f'{Target}/'+'GaussianNoise_Scale40_'+file, image_aug)

        #AdditiveGaussianNoise(scale=(10, 60))
            GaussianNoise3 = iaa.AdditiveGaussianNoise(scale=(60))
            image_aug = GaussianNoise3(image=image)
            imageio.imwrite(f'{Target}/'+'GaussianNoise_Scale60_'+file, image_aug)
            
        #iaa.AddToHueAndSaturation((-60, 60))
            HueandSat1 = iaa.AddToHueAndSaturation((-30))
            image_aug = HueandSat1(image=image)
            imageio.imwrite(f'{Target}/'+'HueAndSat_Scale_-30'+file, image_aug)

        #iaa.AddToHueAndSaturation((-60, 60))
            HueandSat2 = iaa.AddToHueAndSaturation((30))
            image_aug = HueandSat2(image=image)
            imageio.imwrite(f'{Target}/'+'HueAndSat_Scale_+30'+file, image_aug)
        
        #Incease Brightness
            Brightness = iaa.Multiply((0.5, 1.5), per_channel=0.5)
            image_aug = Brightness(image=image)
            imageio.imwrite(f'{Target}/'+'Brightness_'+file, image_aug)

        #Incease Constrast
            Constrast = iaa.LinearContrast((0.5, 2.0), per_channel=0.5)
            image_aug = Brightness(image=image)
            imageio.imwrite(f'{Target}/'+'Brightness_'+file, image_aug)

        #Grayscale
            Grayscale = iaa.Grayscale(alpha=(0.0, 1.0))
            image_aug = Grayscale(image=image)
            imageio.imwrite(f'{Target}/'+'Brightness_'+file, image_aug)
        
    
for folder in os.listdir(Path2):
    P = os.path.join(f'{Path2}', folder)
    directory = "{}".format(folder) # Also gets target species
    T = os.path.join(Target2, directory)
    os.makedirs(T)
    print(T)
    image(P,T)

for folder in os.listdir(Path1):
    P = os.path.join(f'{Path1}', folder)
    directory = "{}".format(folder) # Also gets target species
    T = os.path.join(Target1, directory)
    os.makedirs(T)
    print(T)
    image(P,T)
    
