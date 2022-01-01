# Obj. Det. here...
import yolov5
import numpy as np
import cv2 as cv
from warnings import filterwarnings

filterwarnings('ignore')

# load model
model = yolov5.load('static/models/best.pt')

# set image
img = "static/models/test.jpg"
# set confidence level
model.conf = 0.4

# inference with custom input size
results = model(img)

# save image
results.save("static/models/")