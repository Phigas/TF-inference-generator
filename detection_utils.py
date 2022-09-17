import os
import glob
import numpy as np
from typing import Any, Union

import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder

from data_representation import Image, BoundBox
from data_readers import ImageReader, TfrecordReader

            
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
    pass