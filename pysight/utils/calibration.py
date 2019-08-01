#! /usr/bin/python3
# calibration.py -- Webcam calibration using OpenCV and chessboards
# https://docs.opencv.org/4.1.0/dc/dbb/tutorial_py_calibration.html
# https://kushalvyas.github.io/calib.html
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_pose/py_pose.html


import os
import glob
import sys
import argparse
import time
import pprint
import numpy as np
import cv2
from scipy import optimize as opt

np.set_printoptions(suppress=True)
puts = pprint.pprint

# Chessboard params
PATTERN_SIZE = (7, 5)
SQUARE_SIZE = 1.0

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def show_image(string, image):
    cv2.imshow(string, image)
    while True:
        key_press = cv2.waitKey(0)
        if key_press == 27:
            cv2.destroyAllWindows()
            break
    return

def get_webcam_frame():
    wbc = cv2.VideoCapture(0)
    # while not wbc.isOpened(): pass  # Wait for webcam

    time.sleep(5)
    for i in range(10):
        ok, frame = wbc.read()

    if ok: return frame
    else: raise Exception('error reading cam')

def get_chess_corners(frame):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ok, corners = cv2.findChessboardCorners(gray_frame, PATTERN_SIZE, None)
    print(ok)
    if ok:
        objpoints.append(objp)

        sub_corners = cv2.cornerSubPix(
                gray_frame, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(sub_corners)
        return (objpoints, imgpoints)
    else:
        show_image("Error", frame)
        #raise Exception('Chessboard not found')

def get_webcam_matrix():
    frame = get_webcam_frame()
    objpoints, imgpoints = get_chess_corners(frame)
    ok, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, frame.shape[1:], None, None)

    if ok:
        puts(mtx)
        puts(dist)
        puts(rvecs)
        puts(tvecs)



if __name__ == "__main__":
    get_webcam_matrix()
