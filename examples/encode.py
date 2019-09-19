import cv2
from neural_obfuscator.detector import FacialLandmarksModel, DlibFaceDetector
from neural_obfuscator.aligner import FaceAligner
from neural_obfuscator.stylegan_encoder import StyleGANEncoder
from neural_obfuscator.stylegan import StyleGAN
from neural_obfuscator.utils import show_image

# initialize models
detector = DlibFaceDetector()
landmarks_model = FacialLandmarksModel()
aligner = FaceAligner()
encoder = StyleGANEncoder()
decoder = StyleGAN(weights="ffhq")

# detect faces
img = cv2.imread("../mila.jpg")
faces = detector.detect(img)

# align the first face
face = faces[0]
landmarks = landmarks_model.predict(img, face)
img_aligned, params = aligner.align(img, landmarks, method="eyes_nose")

# get the encoding
dlatents = encoder.encode(img_aligned, optim_image_size=256, num_steps=300)

# visualize the true and synthesized image
imgs_synth = decoder.synthesis(dlatents.cpu())
imgs_synth = decoder.postprocess(imgs_synth)
show_image(img_aligned)
show_image(imgs_synth[0])
