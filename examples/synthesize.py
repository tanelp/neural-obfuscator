import numpy as np
import torch
import neural_obfuscator as no

model = no.StyleGAN(weights="ffhq")
latents = np.random.RandomState(5).randn(1, 512).astype(np.float32)
latents = torch.from_numpy(latents)
imgs = model.forward(latents)

no.show_image(imgs[0])
