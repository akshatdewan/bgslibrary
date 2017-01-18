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
#count = 5

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
latest = 2
suspicious_cntr = 0
try:
    #cap=cv2.VideoCapture("nest.avi")
    while(True):
        time.sleep(0.1)
        # Read the pictures/ Read the latest (minus 1) picture in the directory
        # TODO Need to modify the mypath variable to make it generic / machine agnostic
        mypath = "/home/smavnet/Akshat/bgslibrary2/images/"
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        # TODO Perhaps there is an faster way to obtain the "latest"
        latest = max([int(f.split('capImage')[1].split('.')[0]) for f in onlyfiles])
        latest += 1
        #frame = cv2.imread("images/capImage"+str(count)+".jpg")
        #print "images/capImage"+str(latest - 1)+".jpg"
        #print "images/capImage"+str(latest - 2)+".jpg"
        # Our operations on the frame come here
        orig_frame = cv2.imread(mypath+"/capImage"+str(latest-1)+".jpg")
        current_frame = cv2.resize(orig_frame, (0,0), fx=0.5, fy=0.5) 
        #current_frame = imutils.resize(orig_frame, width=min(400,orig_frame.shape[1])) 
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        current_frame_gray = cv2.GaussianBlur(current_gray,(5,5),5)
        current_frame_binary = cv2.adaptiveThreshold(current_frame_gray,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

        # Reading the previous frame from disk like this because it is possible that processing was not real time and during proc other frames were obtained
        orig_prev_frame = cv2.imread(mypath+"/capImage"+str(latest-2)+".jpg")
        prev_frame = cv2.resize(orig_prev_frame, (0,0), fx=0.5, fy=0.5) 
        #prev_frame = imutils.resize(orig_frame, width=min(400,orig_prev_frame.shape[1])) 
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_frame_gray = cv2.GaussianBlur(prev_gray,(5,5),5)
        prev_frame_binary = cv2.adaptiveThreshold(prev_frame_gray,1,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        #ret, current_frame_binary = cv2.threshold(current_frame_gray, 127,255,cv2.THRESH_BINARY)

        #Subtract the current frame from the prev_frame
        if prev_frame is not None:
            bg = cv2.absdiff(current_frame_gray, prev_frame_gray)
        else:
            bg = copy.copy(current_frame_gray)
        intensity = np.sum(bg)
        #print intensity
        
        # Find the moments of the bg image and then estimate its centroid
        M = cv2.moments(bg)
        try:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            #cv2.rectangle(bg,(cx,cy),(cx+2,cy+2),(255,0,0),thickness=5)
        except: 
            pass
               
        if intensity > 0 and intensity > threshold_nestcam: 
            message = "Motion Detected"
            message1 = "Intruder Detected"
            #print message
            sock.sendall(message)
            #sock.sendall(message1)
            # Writing the image on the disk to be shown/sent to the client application
            cv2.imwrite("/usr/share/openhab/webapps/images/suspicious_activity.png",current_frame)

            # The pedestrain detection section begins
            orig = current_frame.copy()

            # detect people in the image
            (rects, weights) = hog.detectMultiScale(current_frame, winStride=(4, 4),padding=(8, 8), scale=1.1)

            # apply non-maxima suppression to the bounding boxes using a
            # fairly large overlap threshold to try to maintain overlapping
            # boxes that are still people
            rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
            pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

            # Apply the criterion that bg centroid must lie inside a box(The bounding box must surround a high motion region)
            pick_new = []
            for (xA, yA, xB, yB) in pick:
                xA_new = xA + 0.25*(xB - xA)
                yA_new = yA + 0.25*(yB - yA)
                xB_new = xB - 0.25*(xB - xA)
                yB_new = yB - 0.25*(yB - yA)
                if(xA_new < cx and cx < xB_new and yA_new < cy and cy < yB_new ):
                    pick_new.append((xA, yA, xB, yB))
            if len(pick_new) > 0:
                print suspicious_cntr
                if (suspicious_cntr == 0):
                    suspected_frame = latest
                if (latest - suspected_frame < 50):
                    suspicious_cntr += 1
                else: 
                    suspicious_cntr = 0
                if (suspicious_cntr == 1):
                    suspicious_cntr = 0
                    cv2.imwrite("/usr/share/openhab/webapps/images/suspicious_person.png",current_frame)
                    print message1
                    print (mypath+"/capImage"+str(latest-1)+".jpg") 
                    sock.sendall(message1)

            # draw the final bounding boxes
            #for (xA, yA, xB, yB) in pick_new:
            #    cv2.rectangle(current_frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

        # TODO update the prev_frame (the previous is the last frame processed but in priciple it should be the second last frame captured)
        #prev_frame = copy.copy(current_frame)
        #prev_frame_gray = copy.copy(current_frame_gray)
        #prev_frame_binary = copy.copy(current_frame_binary)
        #cv2.imshow('frame', current_frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
        # When everything done, release the capture
    #cap.release()
    #cv2.destroyAllWindows()

except(KeyboardInterrupt, SystemExit):
    #p.terminate()
    raise
except:
    pass
