"""
Download the fathomnet data set with respective to each class folders as per their UUID
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

concept_counts = boundingboxes.count_total_by_concept()
# Print how many there are
print('FathomNet has', len(concept_counts), 'localized concepts!')

#print("Number of concept to see in bar chart, N : ")

N = int(input("Number of concept to see in bar chart, N : "))

""" Make a bar chart of the top N concepts by bounding boxes """
concept_counts = boundingboxes.count_total_by_concept()
concept_counts.sort(key=lambda cc: cc.count, reverse=True)
concepts, counts = zip(*((cc.concept, cc.count) for cc in concept_counts[:N]))

# Make a bar chart
fig = px.bar(
x=concepts, y=counts, 
labels={'x': 'Concept', 'y': 'Bounding box count'}, 
title=f'Top {N} concepts', 
text_auto=True
)
fig.show()

concept_list = []
print(concept_list)
i = 1
while(i <=10):
    concept = random.choice(all_concepts)
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
    directory = "./001_FathomNet_Dataset"
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
