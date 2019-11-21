'''
Class: CIS 581
Assignment: 3B - Optical Flow
Authors: Barrows, Fisher, Woc
Date: Nov 13, 2019
'''

from objectTracking import *
import cv2

# Pick Video
while True:
    choice = input("Choose difficulty... Type 'e' for easy, 'm' for medium, 'h' for hard: ").lower()
    if choice == 'e':
        vid = 'Easy.MP4'
        n_boxes = 2
        break
    elif choice == 'm':
        vid = 'Medium.MP4'
        n_boxes = 1
        break
    elif choice == 'h':
        vid = 'hard.MP4'
        n_boxes = 1
        break
    else:
        print('INVALID INPUT. Try again.')

# Import Raw Video
rawVideo = cv2.VideoCapture(vid)

# Create Tracking Video
trackedVideo = objectTracking(rawVideo,n_boxes)