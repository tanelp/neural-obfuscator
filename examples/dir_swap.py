import cv2
import neural_obfuscator as no

model = no.Obfuscator(method="swap")
img = cv2.imread("path/to/image.jpg")
img_gdpr = model.obfuscate(img, direction="smile", distance=1.5)
no.show_image(img_gdpr)
