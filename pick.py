from imutils import paths
import os
import cv2
import pickle
import argparse
import face_recognition

with open('encodings.pickle', 'rb') as handle:
    b = pickle.load(handle)

print(b)