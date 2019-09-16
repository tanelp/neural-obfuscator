import cv2

from .detector import DlibFaceDetector
from .aligner import FaceAligner

class Obfuscator:
    def __init__(self, detector_name = "dlib", aligner_name="default", method="pixelate"):
        self.detector = self._init_detector(detector_name)
        self.aligner = self._init_aligner(aligner_name)

    def _init_detector(self, name):
        if name == "dlib":
            return DlibFaceDetector()
        else:
            raise NotImplementedError

    def _init_aligner(self, name):
        if name == "default":
            return FaceAligner()
        else:
            raise NotImplementedError

    @staticmethod
    def pixelate(img, w=16, h=16):
        out_width, out_height, _ = img.shape
        img_small = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        img_out = cv2.resize(img_small, (out_width, out_height), interpolation=cv2.INTER_NEAREST)
        return img_out

    @staticmethod
    def merge(img1, img2, mask):
        out = img1 * (1 - mask) + img2 * mask
        return out

    def obfuscate(self, img):
        faces = self.detector.detect(img)
        for face in faces:
            landmarks = self.aligner.get_landmarks(img, face)
            aligned_face, params = self.aligner.align(img, face, landmarks)
            obfuscated_face = self.pixelate(aligned_face)
            obfuscated_face_back = self.aligner.backproject(obfuscated_face, img, params)
            obfuscated_face_canvas = self.aligner.append_face_to_black_canvas_by_eyes(obfuscated_face, img, params["center"])
            face_mask = self.aligner.get_mask(landmarks, img.shape[0], img.shape[1])
            img = self.merge(img, obfuscated_face_canvas, face_mask)
        return img
