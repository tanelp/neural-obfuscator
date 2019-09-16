# TODO: change and handle model paths and downloading

import os
from collections import OrderedDict

import dlib
import numpy as np

from .utils import download_file, unpack_bz2

class DlibFaceDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def detect(self, img):
        dets = self.detector(img, 1)
        return dets

class Landmarks:
    FACIAL_LANDMARKS_68_IDXS = OrderedDict([
        ("mouth", (48, 68)),
        ("inner_mouth", (60, 68)),
        ("right_eyebrow", (17, 22)),
        ("left_eyebrow", (22, 27)),
        ("right_eye", (36, 42)),
        ("left_eye", (42, 48)),
        ("nose", (27, 36)),
        ("jaw", (0, 17))
    ])

    def __init__(self, coords):
        self.coords = coords

    def get_left_eye_center(self):
        left_eye_start, left_eye_end = Landmarks.FACIAL_LANDMARKS_68_IDXS["left_eye"]
        left_eye_points = self.coords[left_eye_start:left_eye_end]
        left_eye_center = left_eye_points.mean(axis=0).astype("int")
        return left_eye_center

    def get_right_eye_center(self):
        right_eye_start, right_eye_end = Landmarks.FACIAL_LANDMARKS_68_IDXS["right_eye"]
        right_eye_points = self.coords[right_eye_start:right_eye_end]
        right_eye_center = right_eye_points.mean(axis=0).astype("int")
        return right_eye_center

class FacialLandmarksModel:
    def __init__(self):
        self.model = self.load_model("shape_predictor_68_face_landmarks.dat")

    def load_model(self, name):
        path = os.path.expanduser(os.path.join("~", ".neural_obfuscator", name))
        if not os.path.exists(path) and name == "shape_predictor_68_face_landmarks.dat":
            landmarks_model_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            download_file(landmarks_model_url, name + ".bz2")
            unpack_bz2(path + ".bz2")
        model = dlib.shape_predictor(path)
        return model

    def predict(self, img, rect):
        landmarks = self.model(img, rect)
        coords = [(x.x, x.y) for x in landmarks.parts()]
        coords = np.array(coords, dtype="int")
        return Landmarks(coords)
