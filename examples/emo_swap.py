import cv2
import neural_obfuscator as no

model = no.Obfuscator(method="swap")
img = cv2.imread("path/to/image.jpg")
img_gdpr = model.obfuscate(img,
                           swap_to_random=False,
                           direction="smile",
                           distance=2.0)
no.show_image(img_gdpr)
