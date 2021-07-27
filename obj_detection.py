import os
import pathlib

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
from PIL import Image, ImageFont, ImageDraw, ImageEnhance


from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import datetime
import cv2
import time
import csv

import linecache
from scipy.spatial import distance

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

# Model preparation 

class ModelLoader:
    def load_model(self, model_name):
        #If we want to download a new model
        #base_url = 'http://download.tensorflow.org/models/object_detection/'
        model_file = model_name + '.tar.gz'
        #model_dir = tf.keras.utils.get_file(fname=model_name, origin=base_url + model_file,untar=True)
        model_dir = "saved_models"/pathlib.Path(model_name)/"saved_model"

        model = tf.saved_model.load(str(model_dir))
        model = model.signatures['serving_default']

        return model


PATH_TO_LABELS = 'F:/Machine_learning/Online-study/computer-vision/person_detection_work/models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

#### Loading the saved object detection model

model_name = 'ssd_mobilenet_v2_coco_2018_03_29'
detection_model = ModelLoader().load_model(model_name)

class ObjectDetection:
    def run_inference_for_single_image(self, model, image):
        image = np.asarray(image)
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis,...]
        # Run inference
        output_dict = model(input_tensor)
        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key:value[0, :num_detections].numpy() 
                     for key,value in output_dict.items()}
        output_dict['num_detections'] = num_detections

        # detection_classes should be ints.
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
        # Handle models with masks:

        if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                      output_dict['detection_masks'], output_dict['detection_boxes'],
                       image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                               tf.uint8)
            output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
        return output_dict
    
    def show_inference(self, model, image_path):
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = image_path
        # Actual detection.
        output_dict = self.run_inference_for_single_image(model, image_np)
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          output_dict['detection_boxes'],
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          category_index,
          instance_masks=output_dict.get('detection_masks_reframed', None),
          use_normalized_coordinates=True,
          line_thickness=8)

        return image_np, output_dict


class GetClassAndBoundingBox:
    def get_boxes(self, op_dict):
        boxes = []
        classes = []
        for i in range(op_dict["num_detections"]):
            if op_dict["detection_scores"][i] > 0.5:
                boxes.append(op_dict["detection_boxes"][i])
                classes.append(op_dict["detection_classes"][i])
        return boxes, classes



def video_upload(vid):
    #if not vid:
    #    vid = "/videos/football_min1.mp4"

    person=0
    ball=0
    othors=0


    cap = cv2.VideoCapture(vid)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    size = (frame_width, frame_height)
    
    # Below VideoWriter object will create
    # a frame of above defined The output 
    # is stored in 'filename.avi' file.
    path = os.path.join("static", "detected", "detected.avi")

    fps = cap.get(cv2.CAP_PROP_FPS)      
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps

    font = ImageFont.truetype('arial.ttf', 30)
    while(cap.isOpened()):
        #success is boolean and image contains frame of the video
        try:
            success, vimg = cap.read()
            if success:
                img, op_dict = ObjectDetection().show_inference(detection_model, vimg)
                frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                seek_time = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                seek_time = round(seek_time, 2)
                
                # Fetching the bounding box coordinates
                boxes, classes = GetClassAndBoundingBox().get_boxes(op_dict)
                if len(classes) and (1 in classes):
                    person += classes.count(1)
                if len(classes) and (37 in classes):
                    ball += classes.count(37)

                vimg = cv2.resize(img, (1200,690))
                vimg = cv2.putText(vimg,"Time : " + str(seek_time), (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                if frame_count == frame_count - 1:
                    vimg = cv2.putText(vimg,"Inferences written in csv file.", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Frame",vimg)
                key = cv2.waitKey(1)
                if key == 27:
                    break

            else:
                break
        except Exception as e:
            exc_type, exc_obj, tb = sys.exc_info()
            f = tb.tb_frame
            lineno = tb.tb_lineno
            print('EXCEPTION IN LINE ', lineno , " exception : " , exc_obj)

    cap.release()
    cv2.destroyAllWindows()

    
    print("Number of times person was detected in the video: ", person)
    print("Number of times ball was detected in the video  : ", ball)
    res = "Number of times person detected: " + str(person) + ", Number of times ball detected: " +str(ball)
    
    return path, res