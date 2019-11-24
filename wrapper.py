'''
Class: CIS 581
Assignment: 3B - Optical Flow
Authors: Barrows, Fisher, Woc
Date: Nov 24, 2019
'''
# Has user pick a video and the number of bounding boxes desired
# (INPUT) rawVideo: Raw video provided for difficulty level chosen
# (OUTPUT) trackedVideo: Final video with bounding box(es) and feature points

from objectTracking import objectTracking
import cv2

# Pick Video
while True:
    choice = input("Choose difficulty... Type 'e' for easy, 'm' for medium, 'h' for hard: ").lower()
    if choice == 'e':
        vid = 'Easy.MP4'
        break
    elif choice == 'm':
        vid = 'Medium.MP4'
        break
    elif choice == 'h':
        vid = 'hard.MP4'
        break
    else:
        print('INVALID INPUT. Try again.')
        
# Choose number of objects to track
while True:
    choice = input("Would you like to track 1, 2, or 3 objects? ")
    if choice == '1' or choice == '2' or choice == '3':
        n_box = int(choice)
        break
    else:
        print('INVALID INPUT. Try again.')

# Import Raw Video
rawVideo = cv2.VideoCapture(vid)

# Create Tracking Video
trackedVideo = objectTracking(rawVideo,n_box,max_pts=20,sigma=1,window_size=25,choice)
trackedVideo.release()