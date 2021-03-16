################################################################################

# Example : perform live fire detection in video using FireNet CNN

# Copyright (c) 2017/18 - Andrew Dunnings / Toby Breckon, Durham University, UK

# License : https://github.com/tobybreckon/fire-detection-cnn/blob/master/LICENSE

################################################################################

import cv2
import os
import sys
import math
import imutils
from imutils.video import FPS
from imutils import contours
from skimage import measure

################################################################################

import tflearn
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.layers.normalization import *
from tflearn.layers.estimator import regression

################################################################################

def construct_firenet (x,y, training=False):
    
    network = tflearn.input_data(shape=[None, y, x, 3], dtype=tf.float32)

    network = conv_2d(network, 64, 5, strides=4, activation='relu')

    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = conv_2d(network, 128, 4, activation='relu')

    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = conv_2d(network, 256, 1, activation='relu')

    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)

    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)

    network = fully_connected(network, 2, activation='softmax')

    # if training then add training hyperparameters
    if(training):
        network = regression(network, optimizer='momentum',
                            loss='categorical_crossentropy',
                            learning_rate=0.001)

    # constuct final model
    model = tflearn.DNN(network, checkpoint_path='firenet',
                        max_checkpoints=1, tensorboard_verbose=2)

    return model

################################################################################

def localize_fire (frame):
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
    
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
    
    labels = measure.label(thresh, neighbors=8, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")
 
    frame_copy = frame.copy()

    # loop over the unique components
    for label in np.unique(labels):
    # if this is the background label, ignore it
        if label == 0:
            continue
            
        # otherwise, construct the label mask and count the number of pixels 
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
 
        # if the number of pixels in the component is sufficiently large, then add it to our mask of "large blobs"
        if numPixels > 300:
            mask = cv2.add(mask, labelMask)
    
    # find the contours in the mask, then sort them from left to right
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) > 0:
        cnts = contours.sort_contours(cnts)[0]
        
        # draw in blue the contours that were founded
        #cv2.drawContours(frame_copy, contours, -1, 255, 3)
        
        # find the biggest area
        #c = max(contours, key = cv2.contourArea)
        #(x, y, w, h) = cv2.boundingRect(c)
        #((cX, cY), radius) = cv2.minEnclosingCircle(c)
        #cv2.circle(frame, (int(cX), int(cY)), int(radius),(0, 0, 255), 3)

    # loop over the contours
    for (i, c) in enumerate(cnts):
        # draw the bright spot on the frame
        big = max(contours, key = cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(big)
        ((cX, cY), radius) = cv2.minEnclosingCircle(big)
        cv2.circle(frame, (int(cX), int(cY)), int(radius),(0, 0, 255), 3)
        cv2.putText(frame, "#{}".format(i + 1), (x, y - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    return frame_copy

################################################################################

if __name__ == '__main__':

################################################################################

    # construct and display model
    model = construct_firenet (224, 224, training=False)
    print("Constructed FireNet ...")

    model.load(os.path.join("models/FireNet", "firenet"),weights_only=True)
    print("Loading CNN network weights ...")

################################################################################

    # network input sizes
    rows = 224
    cols = 224

    # display and loop settings
    keepProcessing = True;

################################################################################

    if len(sys.argv) == 2:

        fps = FPS().start()
        
        # load video file from first command line argument
        video = cv2.VideoCapture(sys.argv[1])
        print("Loaded video ...")

        # get video properties
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH));
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        res = cv2.VideoWriter('res.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (width,height))

        
        while (keepProcessing):

            # get video frame from file, handle end of file
            ret, frame = video.read()
            if not ret:
                print("... end of video file reached");
                break;

            # fire localisation function    
            frame_copy = localize_fire(frame)    
                
            # re-size image to network input size and perform prediction
            small_frame = cv2.resize(frame_copy, (rows, cols), cv2.INTER_AREA)
            output = model.predict([small_frame])
            
            # label image based on prediction
            if round(output[0][0]) == 1:
                #cv2.rectangle(frame, (0,0), (width,height), (0,0,255), 50)
                cv2.putText(frame_copy,'FIRE',(int(width/16),int(height/4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),10,cv2.LINE_AA);
            else:
                #cv2.rectangle(frame, (0,0), (width,height), (0,255,0), 50)
                cv2.putText(frame_copy,'CLEAR',(int(width/16),int(height/4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),10,cv2.LINE_AA);
            
            # image display and key handling
            res.write(frame_copy)

            # wait fps time or less depending on processing time taken (e.g. 1000ms / 25 fps = 40 ms)
            fps.update()
            
            key = cv2.waitKey(1)
            if (key == 27):
                keepProcessing = False
            
            # stop the timer and display FPS information
            fps.stop()
            print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
            print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
            
    else:
        print("usage: python firenet.py videofile.ext");

print("=========================================================")
print("End of execution")
print("[INFO] Total elasped time: {:.2f}".format(fps.elapsed()/60))
        
################################################################################
# When everything done, release the capture
video.release()
res.release()
cv2.destroyAllWindows()