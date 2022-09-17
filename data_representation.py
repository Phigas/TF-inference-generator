import numpy as np
import cv2 as cv
from typing import Union


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
            str: class name of the box
        """
        
        return self.__class_name

    def set_confidence(self, confidence: float) -> None:
        """Sets the detection confidence of the box

        Args:
            confidence (float): Confidence between 0 and 1
        """
        
        assert isinstance(confidence, float)
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
    """This class can store and image and it's annotations and predictions.
    
    It also has a useful function to draw annotation onto the image.
    """
    
    def __init__(self, image_np: np.ndarray, filename: str = '') -> None:
        """Initailizes the instance with empty annotations.

        Height and width are uniquely used for saving the size of the image the annotations were scaled to.
        
        Args:
            image_np (np.ndarray): numpy array of the image
            filename (str, optional): filename of the file. Defaults to ''.
        """
        
        self.set_filename(filename)
        self.set_image(image_np)

        # This contains the ground truth annotations of the image
        self.__annotations = []
        self.__predictions = []

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
            image_np (np.ndarray): 3D array dimensions: height, width, channels(1 or 3).
        """
        
        assert isinstance(image_np, np.ndarray), 'The image needs to be a numpy ndarray'
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

    # TODO: The entire size feature is currently pretty useless. It may be smart to save normalized boundingboxes instead of pixels
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

    def get_size_from_image(self) -> None:
        """This function will read the width and height of the loaded image.
        """
        
        assert isinstance(self.__image_np, np.ndarray), 'An image needs to be loaded to read the size from it.'
        
        self.set_size(self.__image_np.shape[1], self.__image_np.shape[0])
    
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

    def set_predictions(self, predictions: list[BoundBox]) -> None:
        """Set the predictions to the given list

        Args:
            predictions (list[BoundBox]): List with all predictions
        """
        
        assert isinstance(predictions, list), 'bbox needs to be an instance of BoundBox'
        assert all(isinstance(x, BoundBox) for x in predictions), 'predictions needs to contain BoundBox objects'
        
        self.__predictions = predictions
    
    def get_predictions(self) -> tuple[BoundBox, ...]:
        """Returns the predictions.

        Returns:
            tuple[BoundBox, ...]: prediction annotations
        """
        
        return tuple(self.__predictions)
    
    def draw_annotations(self, show: bool = True, save_name: str = None, show_time: Union[int, float] = 5.0):
        """Draw the annotations and predictions on the image

        Args:
            show (bool, optional): Show the image in a window? Defaults to True.
            save_name (str, optional): Save the image with the given name. Defaults to None.
            show_time (float, optional): Time in seconds to show the image. Set to 0 for keypress wait. Defaults to 5.
        """
        
        assert isinstance(show_time, (int, float)) and show_time >= 0, 'show time needs to be 0 or bigger and float or int'
        assert isinstance(show, bool), 'Show needs to be a bool'
        assert isinstance(save_name, str) or save_name == None, 'save_name needs to be string or None'
        image_bgr = cv.cvtColor(self.__image_np, cv.COLOR_RGB2BGR)

        # TODO: set as function of image size
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
        for prediction in self.__predictions:
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
            # wait for X seconds or keypress
            cv.waitKey(int(1000*show_time))

    def __str__(self) -> str:
        """Returns a short description of the image

        Returns:
            str: Description of the image
        """
        
        return f'Image object with {len(self.__annotations)} annotations, {len(self.__predictions)} predictions and size {self.get_size()}'