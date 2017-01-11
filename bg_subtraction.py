# Implement bg subtraction along the lines of what is being performed in bgslibrary c++ application

import numpy as np
import cv2
import copy

import socket
import sys
import subprocess
import shlex
import time

from os import listdir
from os.path import isfile, join

#from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import argparse
import imutils

#class mysocket:
#    """ demonstration class only
#      - coded for clarity, not efficiency"""
#    
#    def __init__(self, sock=None):
#        if sock is None:
#            self.sock = socket.socket(
#                socket.AF_INET, socket.SOCK_STREAM)
#        else:
#            self.sock = sock
#
#    def connect(self, host, port):
#        self.sock.connect((host, port))
#
#    def mysend(self, msg):
#        totalsent = 0
#        while totalsent < MSGLEN:
#            sent = self.sock.send(msg[totalsent:])
#            if sent == 0:
#                raise RuntimeError("socket connection broken")
#            totalsent = totalsent + sent
#
#    def myreceive(self):
#        chunks = []
#        bytes_recd = 0
#        while bytes_recd < MSGLEN:
#            chunk = self.sock.recv(min(MSGLEN - bytes_recd, 2048))
#            if chunk == '':
#                raise RuntimeError("socket connection broken")
#            chunks.append(chunk)
#            bytes_recd = bytes_recd + len(chunk)
#        return ''.join(chunks)


# Create a TCP/IP socket (client is this application)
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
SOURCE_PORT, DESTINATION_PORT = 31415, 25001
sock.bind(('0.0.0.0', SOURCE_PORT))

# Connect the socket to the port where the server is listening
server_address = ('localhost', DESTINATION_PORT)
print >>sys.stderr, 'connecting to %s port %s' % server_address
sock.connect(server_address)

# Create Capture Video  Object
#cap = cv2.VideoCapture(0) 

#Video capture for Nestcam stream
#Start the child process for rtmpdump
#command_line = "rtmpdump -v -r rtmps://stream-ire-charlie.dropcam.com/nexus/ce2d2428c4fc4aa5abd9935c323665b5 -o nest.avi"
#args = shlex.split(command_line)
#p = subprocess.Popen(args)
#time.sleep(20)

prev_frame = None
prev_frame_gray = None
prev_frame_binary = None
intensity = -100
#TODO Need to do something about these thresholds - make it work on all cameras
# Need adaptive technique
threshold_sony_laptop_camera = 1500
threshold_nestcam = 100
count = 5

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

try:
    #cap=cv2.VideoCapture("nest.avi")
    while(True):
        time.sleep(1)
        count+=1
        # Read the pictures/ Read the latest (minus 1) picture in the directory
        # TODO Need to modify the mypath variable to make it generic / machine agnostic
        mypath = "/home/smavnet/Akshat/bgslibrary2/images/"
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        # TODO Perhaps there is an faster way to obtain the "latest"
        latest = max([int(f.split('capImage')[1].split('.')[0]) for f in onlyfiles])
        print latest - 1
        #frame = cv2.imread("images/capImage"+str(count)+".jpg")
        orig_frame = cv2.imread("images/capImage"+str(latest-1)+".jpg")
        current_frame = cv2.resize(orig_frame, (0,0), fx=0.5, fy=0.5) 
        # Our operations on the frame come here
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        current_frame_gray = cv2.GaussianBlur(current_gray,(5,5),5)
        current_frame_binary = cv2.adaptiveThreshold(current_frame_gray,\
                                                          1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

        # Reading the previous frame from disk like this because it is possible that processing was not real time and during proc other frames were obtained
        orig_prev_frame = cv2.imread("images/capImage"+str(latest-2)+".jpg")
        prev_frame = cv2.resize(orig_prev_frame, (0,0), fx=0.5, fy=0.5) 
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_frame_gray = cv2.GaussianBlur(prev_gray,(5,5),5)
        prev_frame_binary = cv2.adaptiveThreshold(prev_frame_gray,\
                                                          1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        #ret, current_frame_binary = cv2.threshold(current_frame_gray, 127,255,cv2.THRESH_BINARY)

        #Subtract the current frame from the prev_frame
        if prev_frame is not None:
            bg = cv2.absdiff(current_frame_binary, prev_frame_binary)
            bg_1 = cv2.absdiff(current_frame_gray, prev_frame_gray)
        else:
            bg = copy.copy(current_frame_binary)
            bg_1 = copy.copy(current_frame_gray)
        if prev_frame is not None:
            intensity = np.sum(bg)
            intensity_gray = np.sum(bg_1)
        # Find the contours and the bounding rectangle for the portion of the image where motion was observed
        contours,hierarchy = cv2.findContours(bg, 1, 2)
        cnt = contours[0]
        M = cv2.moments(cnt)
        try:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        cv2.rectangle(bg,(cx,cy),(cx+2,cy+2),(255,0,0),thickness=5)
               
        if intensity > 0 and intensity > threshold_nestcam: 
            message = "Intruder Detected"
            print message
            print intensity
            sock.sendall(message)
            # Writing the image on the disk to be shown/sent to the client application
            cv2.imwrite("/usr/share/openhab/webapps/images/suspicious_activity.png",current_frame)

            ## The pedestrain detection section begins
            #image = current_frame.copy()
            ##image = imutils.resize(image, width=min(400, image.shape[1]))
            #orig = image.copy()

            ## detect people in the image
            #(rects, weights) = hog.detectMultiScale(image, winStride=(1, 1),padding=(8, 8), scale=1.1)

            ## draw the original bounding boxes
            #i = 0
            #for (x, y, w, h) in rects:
            #    cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
            #    cv2.putText(orig,str(weights[i]), (25,25), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 5)
            #    i += 1

            ## apply non-maxima suppression to the bounding boxes using a
            ## fairly large overlap threshold to try to maintain overlapping
            ## boxes that are still people
            #rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
            #pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

            ## Apply some motion criterion (The bounding box must surround a high motion region)
            #pick_new = []
            #for (xA, yA, xB, yB) in pick:
            #    size_of_bounding_box = (xB - xA)*(yB - yA)
            #    #print "Size of Bounding box ", size_of_bounding_box
            #    size_of_bg = bg.shape[0]*bg.shape[1]
            #    #size_of_image = image.shape[0]*image.shape[1]
            #    #print bg.shape
            #    #print xA, yA, xB, yB
            #    #print "Size of bg image ", size_of_bg
            #    #print "Size of image ", size_of_image
            #    intensity_bounding_box = np.sum(bg[xA:xB,yA:yB])
            #    #print "Intensity of binary ", intensity
            #    #print "Intensity of bounding box ", intensity_bounding_box
            #    #print "ratio of intensities", intensity_bounding_box/float(intensity) 
            #    if (intensity_bounding_box/float(intensity)) > ((1.5) * (size_of_bounding_box/float(size_of_bg))) :
            #        pick_new.append((xA, yA, xB, yB)) 
            ## draw the final bounding boxes
            #for (xA, yA, xB, yB) in pick_new:
            #    cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
            #    print "ratio of intensities", intensity_bounding_box/float(intensity) 

            ##cv2.imshow("Before NMS", orig)
            #cv2.imshow("After NMS", image)
            #cv2.waitKey(2)
        # TODO update the prev_frame (the previous is the last frame processed but in priciple it should be the second last frame captured)
        #prev_frame = copy.copy(current_frame)
        #prev_frame_gray = copy.copy(current_frame_gray)
        #prev_frame_binary = copy.copy(current_frame_binary)
        # Display the resulting frame
        #cv2.imshow('frame',current_frame_binary)
        #cv2.imshow('frame',current_gray_small)
        cv2.imshow('frame',bg_1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # When everything done, release the capture
    #cap.release()
    cv2.destroyAllWindows()

except(KeyboardInterrupt, SystemExit):
    #p.terminate()
    raise
except:
    pass
