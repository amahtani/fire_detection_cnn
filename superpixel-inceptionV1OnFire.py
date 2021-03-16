################################################################################

# Example : perform live fire detection in video using superpixel localization
# and the superpixel trained version of the InceptionV1-OnFire CNN

# Copyright (c) 2017/18 - Andrew Dunnings / Toby Breckon, Durham University, UK

# License : https://github.com/tobybreckon/fire-detection-cnn/blob/master/LICENSE

################################################################################

import cv2
import os
import sys
import math
import numpy as np

################################################################################

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression

################################################################################

from inceptionV1OnFire import construct_inceptionv1onfire

################################################################################

# construct and display model

model = construct_inceptionv1onfire (224, 224, training=False)
print("Constructed SP-InceptionV1-OnFire ...")

model.load(os.path.join("models/SP-InceptionV1-OnFire", "sp-inceptiononv1onfire"),weights_only=True)
print("CNN network weights loaded...")

################################################################################

# network input sizes

rows = 224
cols = 224

# display and loop settings
keepProcessing = True;

################################################################################

if len(sys.argv) == 2:

    # load video file from first command line argument
    video = cv2.VideoCapture(sys.argv[1])
    print("Loading video ...")

    # get video properties

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH));
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    res = cv2.VideoWriter('res.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (224,224))

    while (keepProcessing):

        # get video frame from file, handle end of file
        ret, frame = video.read()
        if not ret:
            print("... end of video file reached");
            break;

        # re-size image to network input size and perform prediction
        small_frame = cv2.resize(frame, (rows, cols), cv2.INTER_AREA);

        # OpenCV imgproc SLIC superpixels implementation below
        slic = cv2.ximgproc.createSuperpixelSLIC(small_frame, region_size=22)
        slic.iterate(10)

        # getLabels method returns the different superpixel segments
        segments = slic.getLabels()

        # loop over the unique segment values
        for (i, segVal) in enumerate(np.unique(segments)):

            # Construct a mask for the segment
            mask = np.zeros(small_frame.shape[:2], dtype = "uint8")
            mask[segments == segVal] = 255

            # get contours (first checking if OPENCV >= 4.x)
            if (int(cv2.__version__.split(".")[0]) >= 4):
                contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            else:
                im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # create the superpixel by applying the mask
            superpixel = cv2.bitwise_and(small_frame, small_frame, mask = mask)

            # use loaded model to make prediction on given superpixel segments
            output = model.predict([superpixel])
            
            # summarize the shape of the list of arrays
            print([a.shape for a in output])

            if round(output[0][0]) == 1:
                # if prediction for FIRE was TRUE (round to 1), draw GREEN contour for superpixel
                cv2.drawContours(small_frame, contours, -1, (0,255,0), 1)
            else:
                # if prediction for FIRE was FALSE, draw RED contour for superpixel
                cv2.drawContours(small_frame, contours, -1, (0,0,255), 1)

        # image display and key handling
        res.write(small_frame)
        #cv2.imwrite('res.jpg',mask)

        # wait fps time or less depending on processing time taken (e.g. 1000ms / 25 fps = 40 ms)
        key = cv2.waitKey(1)
        if (key == 27):
            keepProcessing = False;
            
else:
    print("usage: python superpixel-inceptionV1-OnFire.py videofile.ext");

################################################################################
# When everything done, release the capture
video.release()
#res.release()
cv2.destroyAllWindows()