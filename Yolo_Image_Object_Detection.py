# import required packages
import streamlit as st
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Streamlit app
st.title("AI Image Object Detection")
st.write("Upload an image, and the model will detect objects within the image.")

def post_process(frame, outs, img, classes):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    boxes = []
    confidences = []
    classIDs = []
    for out in outs:
        for detection in out:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0.7:
                centerX = int(detection[0] * frameWidth)
                centerY = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(centerX - width / 2)
                top = int(centerY - height / 2)
                classIDs.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
                
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 0, 0)
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[classIDs[i]])
            confi = str(round(confidences[i], 2))
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label + " " + confi, (x, y - 10), font, 0.5, (255, 255, 255), 2)
    
    return img

def yolo_out(modelConf, modelWeights, classesFile, image_path):
    net = cv2.dnn.readNetFromDarknet(modelConf, modelWeights)
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    inpWidth = 416
    inpHeight = 416

    frame = cv2.imread(image_path)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (inpWidth, inpHeight), swapRB=True, crop=False)
    net.setInput(blob)
    yolo_layers = net.getUnconnectedOutLayersNames()
    outs = net.forward(yolo_layers)
    
    result_img = post_process(frame, outs, img, classes)
    return result_img

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file:
    image = Image.open(uploaded_file)
    image_path = "temp_image.jpg"
    image.save(image_path)
    
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # YOLO model configuration
    modelConf = "yolov3-tiny.cfg"
    modelWeights = "yolov3-tiny.weights"
    classesFile = "coco.names"

    result_img = yolo_out(modelConf, modelWeights, classesFile, image_path)
    plt.imshow(result_img)
    plt.axis('off')
    st.pyplot(plt)
