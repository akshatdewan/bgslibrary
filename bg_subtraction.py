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

from __future__ import print_function
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


# Create a TCP/IP socket
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
intensity = -100
#TODO Need to do something about these thresholds - make it work on all cameras
# Need adaptive technique
threshold_sony_laptop_camera = 1500
threshold_nestcam = 4000
count = 5

# ~/bin/ffmpeg -i "rtmps://stream-ire-charlie.dropcam.com/nexus/ce2d2428c4fc4aa5abd9935c323665b5" -r 1 images/capImage%01d.jpg

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

try:
    #cap=cv2.VideoCapture("nest.avi")
    while(True):
        time.sleep(1)
        count+=1
        # Read the pictures/ Read the latest (minus 1) picture in the directory
        mypath = "/home/smavnet/Akshat/bgslibrary2/images/"
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        latest = max([int(f.split('capImage')[1].split('.')[0]) for f in onlyfiles])
        print latest - 1
        #frame = cv2.imread("images/capImage"+str(count)+".jpg")
        frame = cv2.imread("images/capImage"+str(latest-1)+".jpg")
        # Capture the stream frame by frame
        #ret, frame = cap.read()
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, (0,0), fx=0.5, fy=0.5) 
        current_frame_gray = cv2.GaussianBlur(gray_small,(5,5),5)
        current_frame_binary = cv2.adaptiveThreshold(current_frame_gray,\
                                                          1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        #ret, current_frame_binary = cv2.threshold(current_frame_gray, 127,255,cv2.THRESH_BINARY)
        
        # The pedestrain detection section begins
        image = frame.copy()
        image = imutils.resize(image, width=min(400, image.shape[1]))
        orig = image.copy()

        # detect people in the image
        (rects, weights) = hog.detectMultiScale(image, winStride=(1, 1),padding=(8, 8), scale=1.01)

        # draw the original bounding boxes
        i = 0
        for (x, y, w, h) in rects:
            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(orig,str(weights[i]), (25,25), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 5)
            i += 1

        # apply non-maxima suppression to the bounding boxes using a
        # fairly large overlap threshold to try to maintain overlapping
        # boxes that are still people
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

        #cv2.imshow("Before NMS", orig)
        cv2.imshow("After NMS", image)
        cv2.waitKey(2)



        #Subtract the current frame from the prev_frame
        if prev_frame is not None:
            bg = cv2.absdiff(current_frame_binary, prev_frame)
        else:
            bg = copy.copy(current_frame_binary)
        if prev_frame is not None:
            intensity = np.sum(bg)
                
        if intensity > 0 and intensity > threshold_nestcam: 
            message = "Intruder Detected"
            print message
            print intensity
            sock.sendall(message)
            # Writing the image on the disk to be shown/sent to the client application
            cv2.imwrite("/usr/share/openhab/webapps/images/suspicious_activity.png",frame)
        # update the prev_frame
        prev_frame = copy.copy(current_frame_binary)
        # Display the resulting frame
        cv2.imshow('frame',gray_small)
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
