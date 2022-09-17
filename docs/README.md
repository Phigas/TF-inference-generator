# TFOD inference generator

## Code functions overview

The main objective when writing this code was to simplify the process of loading a [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) (TFOD) model and running it on a dataset as much as possible. After the model and the dataset are loaded the outputs can very conveniently be iterated over with a for loop like so:

**Example**  
This code loads a TFOD model and a dataset in the .record format. Then it runs all images in the dataset through the model and shows each output on screen.

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

Even though this was the main objective, along the way many very useful utility classes were built that can help with TFOD usage:

|Class name|What does it do?|
|---|---|
|BoundBox|Saves a BoundingBox in a unified format|
|Image|Saves one image with it's ground truth annotations and it's predictions|
|TfrecordReader|Opens a TFrecord file (.record extension). The contents can also be read in a for loop|
|ImageReader|Opens one or multiple images in the Pascal Voc format. The contents can also be read in a for loop|
|LabelMapUtil|Primitive class for reading and writing TFOD label map files in the .pbtxt format|
|Detector|Loads a TFOD model and a dataset to run inference. Outputs are read in a for loop|
