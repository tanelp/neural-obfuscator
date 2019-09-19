import cv2
import numpy as np

from .detector import FacialLandmarksModel

class FaceAligner:
    def __init__(self):
        self.output_right_eye_x = 0.35
        self.output_left_eye_x = 1.0 - self.output_right_eye_x
        self.output_eyes_y = 0.35
        self.output_eyes_center_x = (self.output_left_eye_x + self.output_right_eye_x) / 2.0
        self.output_width = 256
        self.output_height = 256
        self.landmarks_model = self._init_landmarks_model("dlib")

    def _init_landmarks_model(self, name):
        if name == "dlib":
            return FacialLandmarksModel()
        else:
            raise NotImplementedError

    def get_landmarks(self, img, rect):
        return self.landmarks_model.predict(img, rect)

    def get_rotation_angle(self, left_eye_center, right_eye_center):
        dy = left_eye_center[1] - right_eye_center[1]
        dx = left_eye_center[0] - right_eye_center[0]
        angle = np.degrees(np.arctan2(dy, dx))
        return angle

    def get_scaling_factor(self, left_eye_center, right_eye_center):
        output_dist = self.output_left_eye_x - self.output_right_eye_x
        output_dist *= self.output_width

        dy = left_eye_center[1] - right_eye_center[1]
        dx = left_eye_center[0] - right_eye_center[0]
        current_dist = np.sqrt(dx**2 + dy**2)

        factor = output_dist / current_dist
        return factor

    def align_by_eyes(self, img, landmarks):
        left_eye_center = landmarks.get_left_eye_center()
        right_eye_center = landmarks.get_right_eye_center()
        eyes_center = tuple((left_eye_center + right_eye_center) // 2)
        angle = self.get_rotation_angle(left_eye_center, right_eye_center)
        scale = self.get_scaling_factor(left_eye_center, right_eye_center)

        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

        # translate image such that eyes are in the desired location
        ty = eyes_center[1] - self.output_eyes_y * self.output_height
        output_eyes_dist = self.output_width * (self.output_left_eye_x - self.output_right_eye_x)
        tx = (eyes_center[0] - 0.5 * output_eyes_dist) - self.output_right_eye_x * self.output_width

        M[1, 2] -= ty
        M[0, 2] -= tx

        # apply the transformation
        output = cv2.warpAffine(img, M, (self.output_width, self.output_height), flags=cv2.INTER_CUBIC)

        params = {
            "center": eyes_center,
            "angle": angle,
            "scale": scale,
            "M": M,
        }

        return output, params

    def align_by_eyes_and_nose(self, img, landmarks, output_size = 1024):
        lm = np.array(landmarks.coords)
        lm_chin          = lm[0  : 17]  # left-right
        lm_eyebrow_left  = lm[17 : 22]  # left-right
        lm_eyebrow_right = lm[22 : 27]  # left-right
        lm_nose          = lm[27 : 31]  # top-down
        lm_nostrils      = lm[31 : 36]  # top-down
        lm_eye_left      = lm[36 : 42]  # left-clockwise
        lm_eye_right     = lm[42 : 48]  # left-clockwise
        lm_mouth_outer   = lm[48 : 60]  # left-clockwise
        lm_mouth_inner   = lm[60 : 68]  # left-clockwise

        # auxiliary vectors
        eye_left     = np.mean(lm_eye_left, axis=0)
        eye_right    = np.mean(lm_eye_right, axis=0)
        eye_avg      = (eye_left + eye_right) * 0.5
        eye_to_eye   = eye_right - eye_left
        mouth_left   = lm_mouth_outer[0]
        mouth_right  = lm_mouth_outer[6]
        mouth_avg    = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # oriented crop rectangle
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # get rotation angle
        dy = x[1]
        dx = x[0]
        angle = np.degrees(np.arctan2(dy, dx))

        # get scaling factor
        desired_dist = output_size
        current_dist = 2 * np.sqrt(x[0]**2 + x[1]**2)
        scale_factor = desired_dist / current_dist

        # center
        center = eye_avg + eye_to_mouth * 0.1
        center = tuple(center)

        M = cv2.getRotationMatrix2D(center, angle, scale_factor)

        # translate
        tx = output_size / 2.0 - center[0]
        ty = output_size / 2.0 - center[1]
        M[0, 2] += tx
        M[1, 2] += ty

        # apply the transformation
        output = cv2.warpAffine(img, M, (output_size, output_size), flags=cv2.INTER_CUBIC)

        params = {
            "center": center,
            "angle": angle,
            "scale": scale_factor,
            "M": M,
        }

        return output, params

    def align(self, img, landmarks, method="eyes"):
        if method == "eyes":
            output = self.align_by_eyes(img, landmarks)
        elif method == "eyes_nose":
            output = self.align_by_eyes_and_nose(img, landmarks)
        else:
            raise NotImplementedError
        return output

    def backproject_by_eyes(self, face, img, params):
        # 1. affine transform the face
        center = int(self.output_eyes_center_x * self.output_width), \
                 int(self.output_eyes_y * self.output_height)
        angle = -params["angle"]
        scale = 1.0 / params["scale"]
        M_back = cv2.getRotationMatrix2D(center, angle, scale)

        #back_height, back_width = int(scale * face.shape[0]), int(scale * face.shape[1])
        back_height, back_width = 500, 500
        face_back = cv2.warpAffine(face, M_back, (back_width, back_height), flags=cv2.INTER_CUBIC)
        return face_back

    def append_face_to_black_canvas_by_eyes(self, face_back, img, eyes_center_dst):
        # 2. move face to canvas of original image size and location
        out_img = np.zeros_like(img)

        eyes_center_src = int(self.output_eyes_center_x * self.output_width), \
                          int(self.output_eyes_y * self.output_height)
        #eyes_center_dst = center

        radius_x_left = min(eyes_center_src[0], eyes_center_dst[0])
        radius_x_right = min(face_back.shape[1] - eyes_center_src[0], img.shape[1] - eyes_center_dst[0])
        radius_y_top = min(eyes_center_src[1], eyes_center_dst[1])
        radius_y_bottom = min(face_back.shape[0] - eyes_center_src[1], img.shape[0] - eyes_center_dst[1])


        xmin_src = eyes_center_src[0] - radius_x_left
        ymin_src = eyes_center_src[1] - radius_y_top
        xmax_src = eyes_center_src[0] + radius_x_right
        ymax_src = eyes_center_src[1] + radius_y_bottom


        xmin_dst = eyes_center_dst[0] - radius_x_left
        ymin_dst = eyes_center_dst[1] - radius_y_top
        xmax_dst = eyes_center_dst[0] + radius_x_right
        ymax_dst = eyes_center_dst[1] + radius_y_bottom

        out_img[ymin_dst:ymax_dst, xmin_dst:xmax_dst] = face_back[ymin_src:ymax_src, xmin_src:xmax_src]

        return out_img

    def backproject(self, face, img, params, method="eyes"):
        if method == "eyes":
            output = self.backproject_by_eyes(face, img, params)
        else:
            raise NotImplementedError
        return output

    @staticmethod
    def get_mask(landmarks, height, width):
        convex_coords = cv2.convexHull(landmarks.coords)
        mask = np.zeros((height, width), dtype=np.uint8)
        mask = cv2.fillConvexPoly(mask, convex_coords, color=1)
        mask = np.array([mask, mask, mask]).transpose(1, 2, 0)
        return mask
