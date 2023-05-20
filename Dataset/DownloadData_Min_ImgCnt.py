"""
Download the fathomnet data set with respective to each 
class folders as per their UUID, 
Each class is having atleast 100 images
"""


# import ipywidgets as widgets                      # Provides embedded widgets
# import ipyleaflet                                 # Provides map widgets
import requests                                   # Manages HTTP requests
import numpy as np                                # Facilitates array/matrix operations
import plotly.express as px                       # Generates nice plots
import random                                     # Generates pseudo-random numbers
from PIL import Image, ImageFont, ImageDraw       # Facilitates image operations
from io import BytesIO                            # Interfaces byte data
from fathomnet.api import images, boundingboxes   # Generates bounding box
from urllib.request import urlretrieve            # Download images
from fathomnet.api import images, boundingboxes
from pascal_voc_writer import Writer              # Writes XML files
from fathomnet.api import firebase
import os
import shutil
# from pathlib import Path


# print(boundingboxes.count_all())

all_concepts = boundingboxes.find_concepts()

# Make temporary list of classes with bounding boxes greater than 100
temp =[]
concept_counts = boundingboxes.count_total_by_concept()
for cc in concept_counts[:]:   
    if cc.count > 100:
        temp.append(cc.concept)

# print(temp)
concept_list = []
i = 1
while(i <=10):
    concept = random.choice(temp)
    if len(concept_list) == 0:
        concept_list.append(concept)
        i += 1
    elif concept not in concept_list:
        concept_list.append(concept)
        i += 1

# print(concept_list)
def download_image(image_record):
    """To download images from FathomNet data set"""
    url = image_record.url                     # Extract the URL
    extension = os.path.splitext(url)[-1]
    uuid = image_record.uuid
    image_filename = os.path.join(Path, image_record.uuid + extension)
    urlretrieve(url, image_filename)        # Download the image
    return image_filename

def write_annotation(image_record, image_filename):
    """ To write XML file for Images with bounding box"""
    writer = Writer(image_filename, image_record.width, image_record.height,database='FathomNet')
    for box in image_record.boundingBoxes:
                concept = box.concept
                if box.altConcept is not None:
                    box.concept += box.altConcept
                writer.addObject(box.concept,box.x,box.y,box.x + box.width, box.y + box.height)
                xml_filename = os.path.join(Path, image_record.uuid + '.xml')
                writer.save(xml_filename)            # Write the annotation


for i in concept_list:
    
    directory = "./002_FathomNet_Dataset_Min_ImgCnt"
    random_concept = i
    random_concept_images = images.find_by_concept(random_concept)
    #print(len(random_concept_images))
    data_directory = random_concept
    Path = os.path.join(directory,data_directory)
    os.makedirs(Path, exist_ok=True)

    # print("Started for concept: ", random_concept)
    for image_record in random_concept_images:
        image_filename = download_image(image_record)
        id = image_record.id
        url = image_record.url                     # Extract the URL
        extension = os.path.splitext(url)[-1]
        uuid = image_record.uuid
        image_filename = os.path.join(Path, image_record.uuid + extension)
        image_filename = download_image(image_record)
        write_annotation(image_record, image_filename)

print("Images Downloaded")

