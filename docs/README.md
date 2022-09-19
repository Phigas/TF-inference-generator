# Use TFOD models and datasets in a super easy way

## What does this code do?

This codebase contains a number of utility classes for the [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) (TFOD). The utility classes help with:

- Loading TFOD models and using them for inference
- Loading datasets in .record of Pascal Voc format
- Storing Object Detection related data in a unified way

One important feature of this codebase is that the classes act as python generators. This means that once the model/dataset is loaded the data can be accessed in a for loop which is very convenient. The example code below demonstrates this. It also shows that this codebase was designed to be as easy to use as possible. Only 5 lines of code are necessary to load a model and dataset and to show the inference results on the screen for each image in the dataset (see Example 1).

This table summarizes all classes, what they do and their locations:

|Class name|location|What does it do?|
|---|---|---|
|BoundBox|data_representation.py|Saves a BoundingBox in a unified format|
|Image|data_representation.py|Saves one image with it's ground truth annotations and it's predictions. Can also draw the ground truth and predictions on the image|
|TfrecordReader|data_readers.py|Opens a TFrecord file (.record extension). The contents can also be read in a for loop|
|ImageReader|data_readers.py|Opens one or multiple images in the Pascal Voc format. The contents can also be read in a for loop|
|LabelMapUtil|detection_utils.py|Primitive class for reading and writing TFOD label map files in the .pbtxt format|
|Detector|detection_utils.py|Loads a TFOD model and a dataset to run inference. Outputs are read in a for loop|

Since all the code is the repository has type hinting and docstrings you can take a look at the classes and functions for a better understanding of the codebase.

## Examples

### 1. Load a model, run inference and show results

```python
from detection_utils import Detector

# loads the TFOD model
detector = Detector('path/to/model/dir')

# loads the dataset specified in the models config file
detector.load_iterable_dataset()

# in each passage through the loop 'image' contains the next loaded image, the ground truth and the predictions from the model
for image in detector:
    # shows the image with ground truth and predictions
    image.draw_annotations()
```

### 2. Load and read a .record file

```python
from data_readers import TfrecordReader

counter = 0

# create an instance of the class and iterate over the dataset
for image in TfrecordReader('dataset.record'):

    # save the results to the drive
    image.draw_annotations(show=False, save_name=f'image_{counter}')
    
    counter += 1
```

## Potential improvements

- Only one image is passed through the neural net at a time. Implementing batches could increase execution times.
- The bounding boxes are saved in pixels. Saving them as normalized values between 0 and 1 is more stable since.
- Adding a functionality for writing tfrecord files would be useful.
- The Detection class currently uses fixed postprocessing steps (non max suppression with threshold 0,5). What steps to use and their parameters needs to be changeable with arguments.
