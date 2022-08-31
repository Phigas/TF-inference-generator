import os
import glob
import cv2 as cv
import numpy as np
from typing import Any, Union
from xml.etree import ElementTree as ET

import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder


class BoundBox:
    """This class is able to represent one bounding box.
    
    The bounding box can be a ground truth without confidence or a prediction with confidence.
    """
    
    def __init__(self, xmin: int, ymin: int, xmax: int, ymax: int, class_name: str, confidence: str = None) -> None:
        """Initializes the class with all given inputs.

        The origin of the image is on the top left square with x in the width direction.
        
        Args:
            xmin (int): left boundary of the box in pixels
            ymin (int): top boundary of the box in pixels
            xmax (int): right boundary of the box in pixels
            ymax (int): bottom boundary of the box in pixels
            class_name (str): name of the class of the box
            confidence (float, optional): Confidence of the prediction. Defaults to None.
        """
        
        self.set_box(xmin, ymin, xmax, ymax)
        self.set_class(class_name)

        if confidence is not None:
            self.set_confidence(confidence)

    def set_box(self, xmin: int, ymin: int, xmax: int, ymax: int) -> None:
        """Sets the box edge attributes.

        Args:
            xmin (int): left boundary of the box in pixels
            ymin (int): top boundary of the box in pixels
            xmax (int): right boundary of the box in pixels
            ymax (int): bottom boundary of the box in pixels
        """
        
        assert all(isinstance(x, int) for x in [xmin, ymin, xmax, ymax]), 'Values need to be int'
        assert all(x >= 0 for x in [xmin, ymin, xmax, ymax]), 'Values given to box have to be bigger than or zero.'
        assert xmin != xmax, f'xmin and xmax need to be different ({xmin=} {xmax=})'
        assert ymin != ymax, f'ymin and ymax need to be different ({ymin=} {ymax=})'
        
        # check and fix inverted annotations
        if xmax < xmin:
            xmax, xmin = xmin, xmax
        if ymax < ymin:
            ymax, ymin = ymin, ymax
                
        self.__xmin = xmin
        self.__ymin = ymin
        self.__xmax = xmax
        self.__ymax = ymax

    def get_box(self) -> tuple[int, int, int, int]:
        """Returns the box edge locations.

        Returns:
            tuple[int, int, int, int]: Tuple containing xmin, ymin, xmax and ymax in that order.
        """

        return self.__xmin, self.__ymin, self.__xmax, self.__ymax

    def set_class(self, class_name: str) -> None:
        """Sets the class id of the box.

        Args:
            class_name (str): Id of the class
        """
        
        assert isinstance(class_name, str), 'class name needs to be string'
        
        self.__class_name = class_name

    def get_class(self) -> str:
        """Returns the class name.

        Returns:
            str: class id of the box
        """
        
        return self.__class_name

    def set_confidence(self, confidence: float) -> None:
        """Sets the detection confidence of the box

        Args:
            confidence (float): Confidence between 0 and 1
        """
        
        # type casting
        confidence = float(confidence)
        
        assert 0 <= confidence <= 1, 'confidence needs to be between 0 and 1'
        self.__confidence = confidence

    def get_confidence(self) -> float:
        """Returns the confidence of the box

        Returns:
            float: Confidence of the box
        """
    
        return self.__confidence

    def __str__(self) -> str:
        """Returns a short description of the box

        Returns:
            str: Description of the box
        """
        
        return f'BoundBox object of class {self.__class_name} with conficence {self.__confidence}'
    
class Image:
    """This class can store and image and it's annotations.
    
    It also has a useful function to draw annotation to the file.
    """
    
    def __init__(self, image_np: np.ndarray, filename: str = '') -> None:
        """Initailizes the instance with empty annotations.

        Height and width are uniquely used for saving the size of the image the annotations were scaled to.
        
        Args:
            image_np (np.ndarray): numpy array of the image
            filename (str): filename of the file
        """
        
        self.set_filename(filename)
        self.set_image(image_np)

        # This contains the ground truth annotations of the image
        self.__annotations = []

    def get_filename(self) -> str:
        """Returns the filename

        Returns:
            str: filename
        """
        
        return self.__filename
    
    def set_filename(self, filename: str) -> None:
        """Sets the filename

        Args:
            filename (str): filename
        """
        
        assert isinstance(filename, str), 'filename needs to be str'
        
        self.__filename = filename
    
    def set_image(self, image_np: np.ndarray) -> None:
        """This functions sets the image array.

        Args:
            image_np (np.ndarray): 3D array with last dimension as channels (either RGB or grayscale)
        """
        
        assert isinstance(image_np, np.ndarray), 'The image needs to be a numpy ndarray' # This should be avoided and replaced with duck typing
        assert np.all(0 <= image_np) and np.all(image_np <= 255), 'The image values need to be between 0 and 255'
        assert np.issubdtype(image_np.dtype, np.integer), 'The image dtype needs to be int. Image might be normalized to 1.'
        image_np = image_np.astype(np.uint8)
        
        assert len(image_np.shape) == 3, 'The image needs to be a 3 dimensional array'
        assert image_np.shape[-1] in {1,3}, 'The image needs to have 1 or 3 colour channels and the colour channels need to be the last dimension'
        
        # convert to 3 channels
        if image_np.shape[-1] == 1:
            np.concatenate([image_np]*3, -1)
        
        self.__image_np = image_np

    def get_image(self) -> np.ndarray:
        """Rerturn the image.

        Returns:
            np.ndarray: Image array
        """
        
        return self.__image_np

    def set_size(self, annotation_width: int, annotation_height: int) -> None:
        """Saves the size that the annotations were scaled to. This is needed if the image is rescaled to scale the annotations accordingly.

        Args:
            width (int): Width of the image the annotations were scaled to
            height (int): Height of the image the annotations were scaled to
        """
        
        annotation_width, annotation_height = int(annotation_width), int(annotation_height)
        assert annotation_width > 0 and annotation_height > 0, 'Width and height need to be bigger than zero.'
        
        self.__annotation_width = annotation_width
        self.__annotation_height = annotation_height

    def get_size(self) -> tuple[int, int]:
        """Return the size of the image that the annotations were scaled to.

        Returns:
            tuple[int, int]: tuple with width and height in that order
        """
        
        return self.__annotation_width, self.__annotation_height

    def add_annotation(self, bbox: BoundBox) -> None:
        """Add one bbox to the annotations

        Args:
            bbox (BoundBox): One BoundBox instance
        """
        
        assert isinstance(bbox, BoundBox), 'bbox needs to be an instance of BoundBox'
        
        self.__annotations.append(bbox)

    def get_annotations(self) -> tuple[BoundBox, ...]:
        """Returns the ground truth annotations

        Returns:
            tuple[BoundBox, ...]: ground truth annotations
        """
        
        # returned as tuple because ground truth is definitive
        return tuple(self.__annotations)

    def draw_annotations(self, predictions: list[BoundBox] = [], show: bool = True, save_name: str = None, show_time: Union[int, float] = 5.0):
        """Draw the annotations and predictions on the image

        Args:
            predictions (list[BoundBox], optional): A list of BoundBoxes to also draw. Defaults to [].
            show (bool, optional): Show the image in a window? Defaults to True.
            save_name (str, optional): Save the image with the given name. Defaults to None.
            show_time (float, optional): Time in seconds to show the image. Set to 0 for keypress wait. Defaults to 5.
        """
        
        assert isinstance(predictions, list) # bad since other similar datatypes can also work
        assert isinstance(show_time, (int, float)) and show_time >= 0, 'show time needs to be 0 or bigger and float or int'
        assert all(isinstance(x, BoundBox) for x in predictions), 'The elements in predictions need to be BoundBox instances'
        assert isinstance(show, bool), 'Show needs to be a bool'
        assert isinstance(save_name, str) or save_name == None, 'save_name needs to be string or None'
        image_bgr = cv.cvtColor(self.__image_np, cv.COLOR_RGB2BGR)

        RECTANGLE_WIDTH = 1
        FONT_SIZE = .5
        FONT_THICKNESS = 1
        
        RECTANGLE_WIDTH = 5
        FONT_SIZE = 2.5
        FONT_THICKNESS = 5
        
        # draw ground truth
        ground_truth_colour = (0, 255, 0)
        for annotation in self.__annotations:
            xmin, ymin, xmax, ymax = annotation.get_box()
            image_bgr = cv.rectangle(image_bgr, (xmin, ymin), (xmax, ymax), ground_truth_colour, RECTANGLE_WIDTH)
            clss = annotation.get_class()
            image_bgr = cv.putText(image_bgr, clss, (xmin, ymin), cv.FONT_HERSHEY_SIMPLEX, FONT_SIZE, ground_truth_colour, FONT_THICKNESS, cv.LINE_AA)

        # draw predictions (if any)
        predictions_colour = (0, 0, 255)
        for prediction in predictions:
            xmin, ymin, xmax, ymax = prediction.get_box()
            image_bgr = cv.rectangle(image_bgr, (xmin, ymin), (xmax, ymax), predictions_colour, RECTANGLE_WIDTH)
            string = prediction.get_class() + ' ' + f'{prediction.get_confidence()*100:.0f}'
            image_bgr = cv.putText(image_bgr, string, (xmin, ymin), cv.FONT_HERSHEY_SIMPLEX, FONT_SIZE, predictions_colour, FONT_THICKNESS, cv.LINE_AA)
        
        if save_name != None:
            print('Saving image')
            cv.imwrite(f'{save_name}.png', image_bgr)
        if show:
            print('Showing image')
            cv.imshow(save_name, image_bgr)
            # wait for 10 seconds or keypress
            cv.waitKey(int(1000*show_time))

    def __str__(self) -> str:
        """Returns a short description of the image

        Returns:
            str: Description of the image
        """
        
        return f'Image object with {len(self.__annotations)} annotations and size {self.get_size()}'
    
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
            
class LabelMapUtil:
    def parse_label_map(self, label_map_path: str) -> dict[str, int]:
        """Reads the label map protobuf text file and returns as dict

        Raises:
            Exception: Raised if label map file incorrectly formatted
            Exception: Raised if label map file incorrectly formatted

        Returns:
            dict[str, int]: Dict mapping item_name to item_id
        """
        
        assert isinstance(label_map_path, str), 'Label map path needs to be string'
        
        item_id = None
        item_name = None
        items = {}

        with open(label_map_path, "r") as file:
            for line in file:
                line.replace(" ", "")
                if line == "item{":
                    pass
                elif line == "}":
                    pass
                elif "id" in line:
                    if item_id == None:
                        item_id = int(line.split(":", 1)[1].strip())
                    else: raise Exception('Found id twice before finding name in label map.')
                elif "name" in line:
                    if item_name == None:
                        item_name = line.split(":", 1)[1].replace("'", "").strip()
                    else: raise Exception('Found name twice before finding id in label map.')

                if item_id is not None and item_name is not None:
                    items[item_name] = item_id
                    item_id = None
                    item_name = None

        return items
    
    def write_label_map(self, label_map_path: str, items: dict[str, int]) -> None:
        """Writes the label map to a file.

        Args:
            label_map_path (str): Path to the lable map file
            items (dict[str, int]): Dict mappint the label name to the label id
        """
        
        assert isinstance(label_map_path, str), 'Label map path needs to be string'
        
        # create label map     
        with open(label_map_path, 'w') as f:
            for name in items:
                f.write('item { \n')
                f.write(f'\tname:\'{name}\'\n')
                f.write(f'\tid:{items[name]}\n')
                f.write('}\n')  
    
    def parse_label_map_from_config(self, configs: Union[str, dict], from_path: bool = False) -> dict[str, int]:
        """Either reads the label map from the loaded configs or loads configs.

        Args:
            configs (Union[str, dict]): Either the path to the config file or the config dict itself
            from_path (bool, optional): True if a path was provided. Defaults to False.

        Returns:
            dict[str, int]: Dict mapping label_name to label_id
        """
        
        assert isinstance(configs, str) and from_path == True or isinstance(configs, dict) and from_path == False, 'configs needs to be a path if from_path is true. Otherwise is needs to be a dict.'
        
        if from_path:
            configs = config_util.get_configs_from_pipeline_file(configs)
        
        label_map_path = configs['eval_input_config'].label_map_path

        label_map = self.parse_label_map(label_map_path)
        return label_map
    
    def dict_from_list(self, in_list: list[str]) -> dict[str, int]:
        """Creates a label_map dict from a list.

        Args:
            in_list (list[str]): List of the label names in order

        Returns:
            dict[str, int]: Dict mapping label_name to label_id
        """
        
        assert isinstance(in_list, list) and all(isinstance(x, str) for x in in_list), 'in_list has to be a list of strings'
        
        out_dict = {}
        label_id = 1
        for label_name in list:
            out_dict[label_name] = label_id
            label_id += 1
        return out_dict
     
class Detector:
    """This class contains all the code to load a model and evaluate it on inputs.
    
    It can evaluate on .record files on files in the Pascal Voc format and multiple Voc style files.
    """
    
    def __init__(self, model_dir: str, checkpoint_nr: int = -1) -> None:
        """Loads the configs and loads the model.

        Args:
            model_dir (str): Path to the model folder
            checkpoint_nr (int, optional): Number of the checkpoint to load. Defaults to -1.
        """
        
        assert isinstance(model_dir, str), 'model_dir needs to be string'
        
        self.model_dir = model_dir
        
        # read the cofig file 
        config_file = os.path.join(self.model_dir, 'pipeline.config')
        self.configs = config_util.get_configs_from_pipeline_file(config_file)

        self.model, self.step = self.load_model(self.model_dir, checkpoint_nr)
    
    def load_model(self, model_dir: str, checkpoint_nr: int) -> tuple[Any, int]:
        """Loads the specified model and checkpoint

        Args:
            model_dir (str): path to the model direcotry
            checkpoint_nr (int): number of the checkpoint to load (-1 for latest)

        Returns:
            tuple[Any, int]: the loaded model and the global step
        """
        
        assert isinstance(checkpoint_nr, int), 'checkpoint number needs to be int'
        assert checkpoint_nr >= -1, 'checkpoint number needs to be bigger or equal to -1'
        assert isinstance(model_dir, str), 'model_dir needs to be string'
        
        # Load pipeline config and build a detection model
        detection_model = model_builder.build(model_config=self.configs['model'], is_training=False)

        if checkpoint_nr == -1:
            # get latest checkpoint
            get_ckpt_number = lambda ckpt_str : int(ckpt_str[:-6].split('-')[-1])

            checkpoints = []
            for checkpoint in glob.glob(model_dir + '/ckpt-*.index'):
                checkpoints.append(get_ckpt_number(checkpoint))

            assert checkpoints, 'No checkpoints found'
            
            checkpoints.sort()
            ckpt_nr = checkpoints[-1]
        else:
            ckpt_nr = checkpoint_nr

        global_step = tf.compat.v2.Variable(0, trainable=False, dtype=tf.compat.v2.dtypes.int64)
        
        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model, step=global_step)
        ckpt.restore(os.path.join(model_dir, f'ckpt-{ckpt_nr}')).expect_partial()

        return detection_model, int(global_step)
    
    def load_iterable_dataset(self, input_path: str = 'default') -> list[tuple[list[BoundBox], Image]]:
        """Automatically detects if input is a record file or Pascal Voc files.

        Args:
            input_path (str): Path to the items to be evaluated

        Returns:
            list[tuple[list[BoundBox], Image]]: list with the predictions and images. 'default' loads file from config. Defaults to 'default'
        """
        
        assert isinstance(input_path, str), 'input_path needs to be str'
        
        if input_path == 'default':
            input_path = self.configs['eval_input_config'].tf_record_input_reader.input_path[0]
        
        if os.path.isfile(input_path):
            file_extention = input_path.split('.')[-1]
            if file_extention == 'record':
                self.iterable_dataset = TfrecordReader(input_path)
        else:
            self.iterable_dataset = ImageReader(input_path)

    def get_predictions(self, image: Image) -> list[BoundBox]:
        """Runs the image through the model and returns predictions

        Args:
            image (Image): instance of Image to run the model on

        Returns:
            list[BoundBox]: list with all predictions
        """
        
        assert isinstance(image, Image), 'image needs to be instance of Image'
        
        label_map = LabelMapUtil().parse_label_map_from_config(self.configs)
        
        input_tensor = tf.convert_to_tensor(np.expand_dims(image.get_image(), 0), dtype=tf.float32)

        image_preprocessed, shapes = self.model.preprocess(input_tensor)
        prediction_dict = self.model.predict(image_preprocessed, shapes)
        detections = self.model.postprocess(prediction_dict, shapes)
        
        detection_boxes = detections['detection_boxes'].numpy()[0]
        detection_scores = detections['detection_scores'].numpy()[0]
        detection_classes = detections['detection_classes'].numpy()[0]
        
        width, height = image.get_size()
        
        detections = []
        for box, score, clss in zip(detection_boxes, detection_scores, detection_classes):
            ymin, xmin, ymax, xmax = box
            xmin, xmax = round(xmin*width), round(xmax*width)
            ymin, ymax = round(ymin*height), round(ymax*height)

            # sometimes predictions are zero size
            if xmin == xmax or ymin == ymax:
                continue
            
            # convert clss to id (+1 because label map starts at 1 and clss starts at 0)
            for label_name in label_map:
                if label_map[label_name] == clss + 1:
                    class_name = label_name
            
            bbox = BoundBox(xmin, ymin, xmax, ymax, class_name, confidence=float(score))
            detections.append(bbox)      
        
        return detections
    
    def calculate_iou(self, box1: BoundBox, box2: BoundBox) -> float:
        """Calculates the intersection over union of two boxes

        Args:
            box1 (BoundBox): First box
            box2 (BoundBox): Second box

        Returns:
            float: iou value
        """
        
        assert isinstance(box1, BoundBox) and isinstance(box2, BoundBox), 'boxes need to be BoundBox'
        
        xmin_t, ymin_t, xmax_t, ymax_t = box1.get_box()
        area_t = (xmax_t - xmin_t) * (ymax_t - ymin_t)
        xmin_p, ymin_p, xmax_p, ymax_p = box2.get_box()
        area_p = (xmax_p - xmin_p) * (ymax_p - ymin_p)

        xmin_i = max(xmin_t, xmin_p)
        ymin_i = max(ymin_t, ymin_p)
        xmax_i = min(xmax_t, xmax_p)
        ymax_i = min(ymax_t, ymax_p)
        
        if area_t == 0 or area_p == 0:
            return 0

        if xmax_i - xmin_i > 0 and ymax_i - ymin_i > 0:
            area_i = (xmax_i - xmin_i) * (ymax_i - ymin_i)
        else:
            area_i = 0
        
        iou = area_i / (area_p + area_t - area_i)
        return iou

    def non_max_suppression(self, boxes: list[BoundBox], overlap_threshold: float = 0.5, use_soft_nms: bool = False) -> list[BoundBox]:
        """Removes all boxes that overlap too much

        Args:
            boxes (list[BoundBox]): list with the input boxes
            overlap_threshold (float, optional): Boxes that overlap more than this are eliminated. Defaults to 0.5.
            use_soft_nms (bool, optional): soft nms reduces the confidence if overlap instead of removing the box. Defaults to False.

        Returns:
            list[BoundBox]: List with the remaining boxes
        """
        
        assert isinstance(overlap_threshold, float), 'overlap threshold needs to be an int'
        assert 0 < overlap_threshold < 1, 'overlap threshold needs to be between 0 and 1'
        assert isinstance(use_soft_nms, bool), 'use soft nms needs to be bool'
        assert isinstance(boxes, list), 'boxes needs to be a list'
        assert all(isinstance(x, BoundBox) for x in boxes), 'elements of boxes need to be BoundBox'
        
        return_boxes = []
        while boxes:
            # find box with highest iou
            best_box = max(boxes, key=lambda box: box.get_confidence())
            
            # move it to return boxes and delete it from boxes
            boxes.remove(best_box)
            
            for box in boxes:
                # if iou too big -> remove
                iou = self.calculate_iou(box, best_box)
                
                if iou > overlap_threshold:
                    if use_soft_nms:
                        conf = box.get_confidence()
                        conf *= 1 - iou
                        box.set_confidence(conf)
                    else:
                        boxes.remove(box)                    
            
            return_boxes.append(best_box)
            
        return return_boxes
    
    def __iter__(self):
        return self
    
    def __next__(self):
        image = next(self.iterable_dataset)

        predictions = self.get_predictions(image)
        predictions = self.non_max_suppression(predictions)
        
        return predictions, image
        
if __name__ == '__main__':
    # det = Detector('./Tensorflow/workspace/models/complete_efficientdet_d0')
    rec = TfrecordReader('./Tensorflow/workspace/annotations/test_synthetic.record')
    
    for i in rec:
        print(i)
        i.draw_annotations()
        break