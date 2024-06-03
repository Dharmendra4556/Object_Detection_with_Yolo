# AI Image Object Detection using Streamlit and YOLO

This repository contains a Streamlit app for AI-based object detection in images using the YOLO (You Only Look Once) algorithm. The app allows users to upload an image and detects objects within the image, displaying the results.

## Features

- Upload an image through the Streamlit interface.
- Detect objects in the uploaded image using the YOLO algorithm.
- Display the image with detected objects highlighted and labeled.

## Requirements

- Python 3.x
- Streamlit
- OpenCV
- numpy
- Pillow
- matplotlib


1. Download the YOLO model configuration and weights files:

- [yolov3-tiny.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg)
- [yolov3-tiny.weights](https://pjreddie.com/media/files/yolov3-tiny.weights)
- [coco.names](https://github.com/pjreddie/darknet/blob/master/data/coco.names)

Save these files in the root directory of the project.

## Running the App

1. Start the Streamlit app:

```sh
streamlit run Yolo_Image_Object_Detection.py
```

2. Open your web browser and navigate to `http://localhost:8501`.

3. Upload an image and see the detected objects.

## Acknowledgments

- The YOLO algorithm was developed by Joseph Redmon and Ali Farhadi.
- Streamlit library for creating the web app interface.
- OpenCV library for image processing.

## Contact

For more information, visit my [LinkedIn profile](https://www.linkedin.com/in/dharmendra-behara-230388239/).
