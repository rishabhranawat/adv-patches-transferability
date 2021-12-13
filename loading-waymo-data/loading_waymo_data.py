import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools
from PIL import Image 
from os import listdir

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

DIR_TO_STORE = '/home/rdr2143/waymo-adv-dataset/train-large-v2/'

TYPE_UNKNOWN = 0
TYPE_VEHICLE = 1
TYPE_PEDESTRIAN = 2
TYPE_SIGN = 3
TYPE_CYCLIST = 4

def convert_to_dataset_loader_format(frame, camera_image, camera_labels, layout, counter=0):
    if (camera_image.name != 2):
        return False
    
    tf_image = tf.image.decode_jpeg(camera_image.image)
    h, w, _ = tf_image.shape
    for camera_labels in frame.camera_labels:
        if camera_labels.name != camera_image.name:
            continue
        labels_for_image = []
        for label in camera_labels.labels:
            if (label.type == TYPE_PEDESTRIAN):
                width = label.box.length/w
                height = label.box.width/h
                x = label.box.center_x/w
                y = label.box.center_y/h
                labels_for_image.append([0, x, y, width, height])
    
    if (len(labels_for_image) == 0):
        return False
    
    file_name = 'training_img_'+str(counter)
    labels_file_name = DIR_TO_STORE+'labels/'+file_name+'.txt'
    label_file = open(labels_file_name, 'w')
    for each in labels_for_image:
        value_to_write = " ".join([str(x) for x in each])
        label_file.write(f'{value_to_write} \n')
    label_file.close()
    
    image_name = DIR_TO_STORE+file_name+'.jpg'
    decodeit = open(image_name, 'wb')
    decodeit.write(camera_image.image)
    decodeit.close()
    
    return True

def run():
    try:
        TF_RECORDS_DIR = '/home/rdr2143/data/waymotfrecord/training/'
        counter = 0
        processed = []
        for FILENAME in listdir(TF_RECORDS_DIR):
            processed.append(FILENAME)
            dataset = tf.data.TFRecordDataset(TF_RECORDS_DIR+FILENAME, compression_type='')
            for data in dataset:
                if counter % 50 == 0:
                    print(f'images: {counter}')
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                for index, image in enumerate(frame.images):
                    wrote = convert_to_dataset_loader_format(frame, image, frame.camera_labels, [3, 3, index+1], counter)
                    if wrote:
                        counter += 1
                        break
        print(processed)
    except:
        print('There was an error, processed dirs:\n')
        print(processed)

if __name__ == '__main__':
    print('Starting Creating Waymo Traininable Dataset...')
    run()