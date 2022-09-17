import os
import glob
import numpy as np
from typing import Any
import cv2 as cv
from xml.etree import ElementTree as ET
import tensorflow as tf

from data_representation import Image, BoundBox


class TfrecordReader:
    """This class loads a tfrecord dataset.
    
    To get the dataset iterate over the instance with a for loop or use the next() function.
    """
    
    def __init__(self, record_path: str) -> None:
        """Open the dataset and save the iterable because the dataset itself is not needed.

        Args:
            record_path (str): Path to the .record file
        """
        
        dataset = self.open_tfrecord(record_path)
        self.iterable = iter(dataset)
        
    def open_tfrecord(self, record_path: str) -> Any:
        """Opends a .record file and decodes it.

        Args:
            record_path (str): Path to the record file

        Returns:
            tf.python.data.ops.dataset_ops.MapDataset: Parsed dataset
        """
        
        assert isinstance(record_path, str), 'record path needs to be string'
        
        raw_image_dataset = tf.data.TFRecordDataset(record_path)

        # the features that are not needed later are commented out
        image_feature_description = {
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/width' : tf.io.FixedLenFeature([], tf.int64),
            'image/height' : tf.io.FixedLenFeature([], tf.int64),
            'image/object/bbox/xmax' : tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'image/object/bbox/ymax' : tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'image/object/bbox/xmin' : tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'image/object/bbox/ymin' : tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'image/filename' : tf.io.FixedLenFeature([], tf.string),
            # 'image/format' : tf.io.FixedLenFeature([], tf.string),
            # 'image/object/class/label' : tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
            'image/object/class/text' : tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True),
            # 'image/source_id' : tf.io.FixedLenFeature([], tf.string),
        }

        def _parse_image_function(example_proto):
            # Parse the input tf.train.Example proto using the dictionary above.
            return tf.io.parse_single_example(example_proto, image_feature_description)
        
        return raw_image_dataset.map(_parse_image_function)

    def preprocess_tfrecord(self, element: dict) -> Image:
        """Extracts the information from the dict the record file returns and stores it as an Image instance.

        Args:
            element (dict): Dict the record file gives

        Returns:
            Image: Image with the image and the annotations
        """
        
        image_np = tf.io.decode_jpeg(element['image/encoded'].numpy()).numpy()

        height = element['image/height'].numpy()
        width = element['image/width'].numpy()
        width, height = int(width), int(height)

        image = Image(image_np, str(element['image/filename'].numpy()))
        image.set_size(width, height)
        
        # get label names
        labels = [a.decode('ascii') for a in element['image/object/class/text'].numpy()]    

        # build bboxes
        xmins = element['image/object/bbox/xmin'].numpy()
        ymins = element['image/object/bbox/ymin'].numpy()
        xmaxs = element['image/object/bbox/xmax'].numpy()
        ymaxs = element['image/object/bbox/ymax'].numpy()

        for i in range(len(labels)):
            ymin, xmin, ymax, xmax = (ymins[i] * height, xmins[i] * width, ymaxs[i] * height, xmaxs[i] * width)
            # for whatever reason when loading the values they have errors in the 1e-6 range
            # This is why round is needed
            ymin, xmin, ymax, xmax = round(ymin), round(xmin), round(ymax), round(xmax)
            bbox = BoundBox(xmin, ymin, xmax, ymax, labels[i])
            image.add_annotation(bbox)

        return image
    
    def __iter__(self):
        """Retrurns the iterable class for the for fucntion.
        
        since we want to iterate over the class itself return self

        Returns:
            TfrecordReader: The class itself
        """
        
        return self
    
    def __next__(self) -> Image:
        """Returns the next value of the loop

        Returns:
            Image: The image with all the information loaded
        """
        
        element = next(self.iterable)
        return self.preprocess_tfrecord(element)

class ImageReader:
    def __init__(self, input_path: str, file_extention: str = 'jpg') -> None:
        
        assert isinstance(input_path, str), 'input path needs to be str'
        
        if os.path.isdir(input_path):
            images = glob.glob(input_path + f'/*.{file_extention}')
            
            assert len(images) > 0, 'No images found. If the extention is not jpg specify it in the arguments.'

            self.image_paths = images
            
        if os.path.isfile(input_path):
            assert input_path.split('.')[-1] == file_extention, 'file needs to end in file extention'
            
            self.image_paths = [input_path]

        # get the iterator so we can use next function on it
        self.image_paths = iter(self.image_paths)
    
    def preprocess_image(self, image_path: str) -> Image:
        """Loads an image from path and returns Image class instance

        Args:
            image_path (str): Path to the image to load

        Returns:
            Image: Image instance with the image array loaded
        """
        
        assert isinstance(image_path, str), 'image path needs to be string'
        
        img = cv.imread(image_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        image_np = np.array(img)
        image = Image(image_np)
        
        return image
    
    def load_xml_annotations(self, image: Image, xml_path: str) -> None:
        """Loads the information of the xml and adds it to the image

        Args:
            image (Image): Image instance to load the annotations onto
            xml_path (str): Path to the xml file
        """
        
        assert isinstance(image, Image), 'image needs to be instance of Image'
        assert isinstance(xml_path, str), 'xml path needs to be string'
        
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        image.set_size(width, height)

        objs = root.findall('object')
        for obj in objs:
            bbox = obj.find('bndbox')
            xmin = round(bbox.find('xmin').text)
            ymin = round(bbox.find('ymin').text)
            xmax = round(bbox.find('xmax').text)
            ymax = round(bbox.find('ymax').text)
            class_id = obj.find('name').text
            
            annotaion = BoundBox(xmin, ymin, xmax, ymax, class_id)
            image.add_annotation(annotaion)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        image_path = next(self.image_paths)
        
        image = self.preprocess_image(image_path)
        
        # convert image path to xml path
        elems = image_path.split('.')
        elems[-1] = 'xml'
        xml_path = '.'.join(elems)
        
        self.load_xml_annotations(image, xml_path)
        
        return image
