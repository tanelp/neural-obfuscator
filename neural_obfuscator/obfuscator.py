import cv2
import numpy as np
import torch

from .detector import DlibFaceDetector
from .aligner import FaceAligner
from .stylegan import StyleGAN
from .stylegan_encoder import StyleGANEncoder

class Obfuscator:
    def __init__(self, detector_name = "dlib", aligner_name="default", method="pixelate"):
        self.detector = self._init_detector(detector_name)
        self.aligner = self._init_aligner(aligner_name)
        self.method = method
        if method == "swap":
            self.encoder = StyleGANEncoder()
            self.decoder = StyleGAN(weights="ffhq")

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
            if self.method == "pixelate":
                aligned_face, params = self.aligner.align(img, landmarks)
                obfuscated_face = self.pixelate(aligned_face)
                obfuscated_face_back = self.aligner.backproject(obfuscated_face, params)
                obfuscated_face_canvas = self.aligner.append_face_to_black_canvas(obfuscated_face, img, params["center"])
                face_mask = self.aligner.get_mask(landmarks, img.shape[0], img.shape[1])
                img = self.merge(img, obfuscated_face_canvas, face_mask)
            elif self.method == "swap":
                aligned_face, params = self.aligner.align(img, landmarks, method="eyes_nose")
                dlatents_encoded = self.encoder.encode(aligned_face, optim_image_size=256, num_steps=30)
                dlatents_encoded = dlatents_encoded.cpu()

                latents = np.random.randn(1, 512).astype(np.float32)
                latents = torch.from_numpy(latents)
                dlatents_random = self.decoder.mapping(latents)

                dlatents = dlatents_encoded.clone()
                dlatents[:, 4:] = dlatents_random[:, 4:]

                imgs = self.decoder.synthesis(dlatents)
                imgs = self.decoder.postprocess(imgs)
                obfuscated_face = imgs[0]

                obfuscated_face_back = self.aligner.backproject(obfuscated_face, params, method="eyes_nose")
                obfuscated_face_canvas = self.aligner.append_face_to_black_canvas(obfuscated_face_back, img, params["center"], method="eyes_nose", face_size=aligned_face.shape[0])
                face_mask = self.aligner.get_mask(landmarks, img.shape[0], img.shape[1])
                face_center_xywh = cv2.boundingRect(face_mask[:, :, 0])
                face_center = (int(face_center_xywh[0] + face_center_xywh[2]/2), int(face_center_xywh[1] + face_center_xywh[3]/2))
                img = cv2.seamlessClone(obfuscated_face_canvas, img, face_mask*255, face_center, cv2.NORMAL_CLONE)
            else:
                raise NotImplementedError
        return img
