'''
Class: CIS 581
Assignment: 3B - Optical Flow
Authors: Barrows, Fisher, Woc
Date: Nov 13, 2019
'''

from objectTracking import *
import cv2

# Import Raw Video
rawVideo = cv2.VideoCapture('Easy.MP4')

# Create Tracking Video
trackedVideo = objectTracking(rawVideo)

