import time
import os

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from .stylegan import StyleGAN
from .utils import download_file_from_gdrive

class PerceptualLoss(nn.Module):
    def __init__(self, target):
        super(PerceptualLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        self.mean = mean.clone().view(-1, 1, 1)
        self.std = std.clone().view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

class TransformForVGG(nn.Module):
    def __init__(self, image_size):
        super(TransformForVGG, self).__init__()
        self.image_size = image_size

    def forward(self, imgs):
        # resize
        imgs = F.interpolate(imgs, size=self.image_size, mode="bilinear", align_corners=False)

        # scale from [-1, 1] -> [0, 1]
        imgs += 1
        imgs /= 2.0
        imgs = imgs.clamp(0, 1)
        return imgs

class StyleGANEncoder:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder_model = None

    @staticmethod
    def _get_input_optimizer(dlatents):
        optimizer = optim.LBFGS([dlatents.requires_grad_()])
        return optimizer

    def _get_model_and_losses(self, real_img, image_size, content_layers=["conv_6"]):
        # synth model
        synth_model = StyleGAN(weights="ffhq").synthesis.to(self.device).eval()

        # vgg model
        vgg_model = torchvision.models.vgg19(pretrained=True).features.to(self.device).eval()
        vgg_norm_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        vgg_norm_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)

        # normalization module
        normalization = Normalization(vgg_norm_mean, vgg_norm_std).to(self.device)

        content_losses = []

        # make new nn.Sequential
        model = nn.Sequential(normalization)

        # assumption: vgg_model is a nn.Sequential
        i = 0  # increment every time we see a conv
        for layer in vgg_model.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = "conv_{}".format(i)
            elif isinstance(layer, nn.ReLU):
                name = "relu_{}".format(i)
                # The in-place version doesn't play very nicely with the loss
                # So we replace with out-of-place ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = "pool_{}".format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = "bn_{}".format(i)
            else:
                raise RuntimeError("Unrecognized layer: {}".format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(real_img).detach()
                content_loss = PerceptualLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

        # now we trim off the layers after the last contentlosses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], PerceptualLoss):
                break

        model = model[:(i + 1)]
        model = nn.Sequential(synth_model, TransformForVGG(image_size), *model)

        return model, content_losses

    def _run_encoding(self, real_img, image_size, num_steps=300, default_dlatents=0):
        print("Building model and losses ...")
        dlatents = np.zeros((real_img.shape[0], 18, 512), dtype=np.float32)
        dlatents = dlatents + default_dlatents
        dlatents = torch.from_numpy(dlatents).to(self.device)
        model, losses = self._get_model_and_losses(real_img, image_size)
        #print(losses)
        #print(model)
        optimizer = self._get_input_optimizer(dlatents)

        print("Optimizing ...")
        run = [0]
        while run[0] <= num_steps:
            def closure():
                t0 = time.time()
                optimizer.zero_grad()
                pred = model(dlatents)
                loss = 0

                for cl in losses:
                    loss += cl.loss

                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0 or True:
                    print("run {}:".format(run))
                    print("Loss: {:4f}".format(loss.item()))
                    print("t:", time.time() - t0)
                    print()

                return loss

            optimizer.step(closure)

        return dlatents

    @staticmethod
    def _init_encoder_model():
        model_ft = torchvision.models.resnet18()
        num_features = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_features, 512)
        path = os.path.expanduser(os.path.join("~", "neural_obfuscator", "weights_full_epoch_4_loss_0.0314.pth"))
        if not os.path.exists(path):
            download_file_from_gdrive("https://drive.google.com/uc?id=1-cGn2Gmw-sL9N_CEEV0pw16AEhWmlqQU", path)
        model_ft.load_state_dict(torch.load(path))
        model_ft.eval()
        return model_ft

    def _init_dlatents(self, method="avg", img=None):
        if method == "avg":
            return DLATENT_MEAN
        elif method == "net":
            if self.encoder_model is None:
                self.encoder_model = self._init_encoder_model().to(self.device)
            print("Predicting initial dlatents...")
            # img: chw, rgb, [0, 1]
            # model inputs: nchw, rgb, [-1, 1]
            inputs = img.clone()
            image_size = 224
            inputs = F.interpolate(inputs, size=image_size, mode="bilinear", align_corners=False)
            inputs = inputs * 2 - 1
            return self.encoder_model(inputs).cpu().data.numpy()[0]

    def encode(self, img, optim_image_size=256, num_steps=300, method="avg"):
        # set up input image
        # vgg_model input: mini-batches of 3-channel RGB images of shape (3 x H x W)
        img_real = cv2.resize(img, (optim_image_size, optim_image_size), interpolation = cv2.INTER_AREA) # hwc, bgr
        img_real = img_real[:, :, ::-1].transpose(2, 0, 1) #  chw, rgb
        img_real = img_real.astype(np.float32) / 255.0 # [0, 1]
        img_real = np.expand_dims(img_real, axis=0)
        img_real = torch.from_numpy(img_real).to(self.device)

        # run encoding
        dlatents = self._init_dlatents(method=method, img=img_real)
        if num_steps > 0:
            dlatents = self._run_encoding(img_real, image_size=optim_image_size, num_steps=num_steps, default_dlatents=dlatents)
        else:
            dlatents = torch.from_numpy(dlatents).unsqueeze(0).unsqueeze(0).expand(-1, 18, -1).to(self.device)
        return dlatents


DLATENT_MEAN = np.array([
        6.35981783e-02,  8.55128467e-02, -3.02187242e-02,  2.31557004e-02,
       -1.19540654e-02, -1.59383134e-03, -5.75284176e-02, -4.93626706e-02,
        2.82883812e-02,  1.76863804e-01,  3.98273729e-02, -1.03522241e-01,
        5.54512478e-02, -8.44812579e-03,  2.79187094e-02,  4.68369797e-02,
       -6.80034608e-03,  1.26645580e-01, -4.48823534e-02,  4.42643277e-02,
        3.76411714e-02, -1.76016870e-03, -2.35711057e-02,  1.74724460e-02,
        6.53446987e-02,  6.76512253e-04, -5.29145487e-02, -9.66427773e-02,
        7.94632807e-02,  8.26998241e-03, -1.24771567e-02, -1.95246674e-02,
       -4.34599407e-02, -8.27912688e-02, -2.38842256e-02, -5.25071509e-02,
        5.85872419e-02, -8.82288888e-02,  2.57511660e-02, -8.93997848e-02,
        8.91450271e-02, -2.79247630e-02,  3.51615585e-02, -2.26825736e-02,
       -3.30308117e-02,  5.98681085e-02, -1.18438406e-02, -7.79363662e-02,
        4.89572361e-02, -1.09959478e-02, -2.77883410e-02, -6.20033592e-03,
       -1.09314136e-01,  2.76926216e-02, -4.79086377e-02,  4.33482230e-02,
       -2.64900178e-02,  1.20671384e-01,  7.30418116e-02,  9.50055756e-03,
        4.25065421e-02,  4.92642187e-02,  5.17565086e-02,  8.18104520e-02,
        5.70541285e-02, -4.31890041e-02,  5.86826950e-02,  1.65229309e-02,
        3.12814526e-02,  1.16302103e-01,  2.90719364e-02,  6.01321533e-02,
       -6.73829690e-02, -1.27229318e-01,  2.56261509e-02,  1.04466647e-01,
        6.80123866e-02,  1.07086375e-01,  3.49124074e-02,  5.05299866e-02,
        7.13847652e-02, -1.78203383e-03, -6.24515153e-02,  3.68248187e-02,
        1.43618211e-01,  4.51864824e-02,  2.26467699e-02, -1.70215443e-02,
       -1.68634336e-02, -1.42688807e-02,  1.59326978e-02,  1.16758667e-01,
        7.62554333e-02, -2.64006015e-03, -6.68190122e-02, -5.91090769e-02,
       -5.03153801e-02,  3.14770453e-02,  1.20916478e-01,  3.33817117e-02,
        5.15905097e-02,  9.28745717e-02,  5.01053967e-02,  1.39488131e-01,
       -9.02774259e-02,  1.01176441e-01, -3.77398767e-02,  1.04514852e-01,
        4.63266158e-03, -1.00377828e-01,  3.20265861e-03, -6.34529665e-02,
        3.29185203e-02,  1.97292008e-02, -6.69990480e-02,  6.76242039e-02,
       -1.05492389e-02, -7.88031071e-02,  1.10528834e-01,  1.84919849e-01,
        8.39212462e-02, -3.00159790e-02,  7.54897818e-02, -1.00029372e-01,
        5.76963089e-02, -5.16644865e-02,  7.26263747e-02,  8.16177875e-02,
       -7.78122023e-02, -3.51785570e-02,  6.31697848e-02,  1.21733379e-02,
       -5.89677505e-02,  1.20818265e-01, -2.06134340e-04,  5.76502532e-02,
        4.32651453e-02,  4.48477454e-02,  8.46477225e-02,  4.65766117e-02,
       -5.09775653e-02,  1.13212831e-01, -4.94804159e-02,  2.20478605e-03,
        9.37782899e-02,  5.75795732e-02,  2.24896520e-02,  1.47882868e-02,
       -5.10077141e-02, -2.34934520e-02,  1.70292500e-02,  2.27689780e-02,
        9.96289849e-02,  1.81440175e-01,  2.38523558e-02,  1.54378563e-01,
       -1.05763245e-02,  8.28704312e-02,  1.12912670e-01,  2.31495649e-02,
        1.37726754e-01,  8.88556540e-02,  5.34085892e-02,  8.61634240e-02,
        1.33887544e-01, -5.84739670e-02,  3.77358869e-02, -1.05192338e-03,
        7.02381646e-03,  1.62990727e-02, -7.21426308e-02, -1.77843254e-02,
        1.15339451e-01,  5.68640828e-02,  7.00203925e-02, -2.44299509e-02,
       -3.16844098e-02, -1.93873364e-02, -1.32395059e-03, -2.94126868e-02,
       -3.54698114e-02, -1.09007983e-02, -3.33869904e-02, -6.34807199e-02,
        6.50520176e-02,  2.74743531e-02,  3.03954687e-02,  1.71489686e-01,
       -6.90770894e-02,  3.78576145e-02, -1.16276965e-02,  1.59924790e-01,
       -2.87658833e-02, -1.66936349e-02,  2.22133100e-02,  1.90017864e-01,
       -1.91243012e-02,  6.82756770e-04, -2.04535145e-02,  1.10999167e-01,
        3.37514021e-02,  6.48889225e-03,  4.04045917e-02,  1.81715921e-01,
        4.25942093e-02,  2.32717805e-02,  6.41258899e-03,  3.49166878e-02,
        3.27025652e-02, -1.77936759e-02, -7.20648840e-02, -5.41994274e-02,
        8.61484408e-02,  4.07656953e-02,  5.44705726e-02,  6.81823120e-02,
       -6.68541864e-02, -2.32166406e-02, -1.77940279e-02, -9.30882171e-02,
        1.32710814e-01, -2.07984541e-03, -2.55896300e-02, -6.49319068e-02,
        6.19942173e-02, -4.20475528e-02,  2.13992875e-02,  1.51119843e-01,
        1.20156318e-01,  6.72486797e-02,  1.61147997e-01,  6.39618635e-02,
        1.35858878e-01,  3.52479443e-02, -1.17526539e-01, -8.79519507e-02,
        1.22957956e-02,  1.63845327e-02,  1.19006656e-01, -3.99893001e-02,
       -1.07579999e-01, -3.47397616e-03,  6.92256391e-02, -3.85871879e-03,
        2.18531433e-02,  1.31465822e-01,  8.39123689e-03,  5.35924435e-02,
       -8.10606871e-03,  1.06229462e-01, -2.32030731e-02, -3.75893414e-02,
       -5.57534695e-02,  6.49229139e-02, -1.34456679e-01, -8.22408963e-03,
        4.08374853e-02,  2.10058987e-01, -5.45223802e-03,  5.93109578e-02,
        9.33884680e-02, -2.85446085e-02,  3.45789753e-02,  2.32306644e-02,
        3.01123243e-02, -8.07293653e-02,  4.43668067e-02,  2.42832154e-02,
       -2.78804954e-02, -3.78682539e-02,  7.63577446e-02, -1.56479795e-02,
        5.78496233e-03,  1.30936014e-03, -3.24338824e-02,  7.22870454e-02,
        1.59136560e-02,  4.97199632e-02,  8.96074176e-02,  1.78015828e-02,
        3.52624543e-02, -7.37695992e-02,  2.20513046e-02,  1.56472754e-02,
        4.33518142e-02,  9.90440920e-02, -3.09860539e-02,  7.13728070e-02,
       -2.13298015e-02, -1.19566433e-02, -8.44760910e-02,  5.46137318e-02,
        1.70165539e-01, -1.92387868e-02, -6.59938976e-02,  3.14750597e-02,
        1.54648662e-01,  3.80281731e-02,  5.28211519e-03, -1.58181265e-02,
        1.59896597e-01,  4.10421118e-02,  8.46134275e-02,  2.71370653e-02,
        6.66486099e-02, -6.64886534e-02,  3.71034220e-02, -8.60561430e-02,
        2.72653345e-02,  2.61955522e-02, -3.65733053e-03, -5.27013540e-02,
       -6.06001653e-02, -1.01584494e-02, -4.61988635e-02, -2.52374783e-02,
        6.14997260e-02, -1.03787137e-02,  2.43821163e-02, -3.57034132e-02,
        4.26144563e-02, -8.09787540e-04, -9.34060588e-02,  7.01953471e-03,
        1.15653254e-01,  4.80650812e-02, -4.72549535e-03, -2.86619477e-02,
        5.88803813e-02, -5.50394133e-02,  1.50343515e-02, -3.19817923e-02,
        8.95110741e-02,  3.76794934e-02,  4.78837155e-02, -5.72572015e-02,
       -5.07388264e-02,  2.43112762e-02,  1.64637808e-02,  7.39918184e-03,
       -7.49268103e-03,  1.45131396e-02, -1.99801307e-02, -3.32650566e-03,
        8.78755525e-02, -4.37362343e-02,  1.01742279e-02, -6.29662909e-03,
        5.70467487e-02,  1.87493637e-01,  1.39520764e-02,  1.87005345e-02,
       -6.14981307e-03,  4.40039970e-02,  6.23373985e-02, -3.17663960e-02,
        5.59589174e-03,  7.78466538e-02, -2.77044326e-02,  6.08689710e-02,
       -1.38890846e-02,  6.80626407e-02, -1.31963445e-02, -5.31757157e-03,
        6.35656640e-02,  1.03892488e-02, -1.45396227e-02, -2.87259109e-02,
       -2.90757022e-03, -1.01341158e-02, -5.85375838e-02, -8.34064791e-04,
        4.54105577e-03, -1.74645893e-02,  1.13487773e-01,  1.91264506e-02,
       -1.19023167e-01,  3.59204300e-02,  1.21215411e-01,  1.48227558e-01,
       -8.95492546e-03,  8.48979279e-02,  6.97924644e-02, -1.02398042e-02,
        1.71327680e-01,  1.30616233e-01,  3.35412100e-02,  1.29781038e-01,
       -1.66921515e-03,  6.55391365e-02,  4.23320234e-02,  5.91462627e-02,
       -2.95158103e-02, -3.12160943e-02,  2.04226062e-01,  8.81015137e-02,
        4.92634885e-02, -4.93572801e-02, -2.01915726e-02,  9.33927521e-02,
        9.42913070e-02,  1.63939781e-02, -1.80795398e-02,  8.28238130e-02,
        7.97160063e-03,  3.73714343e-02, -5.36204688e-02, -3.36664356e-02,
       -7.89049268e-03,  1.54763851e-02,  1.70656312e-02, -2.40452569e-02,
        5.16169220e-02, -8.21152851e-02, -8.92949570e-03, -2.94994451e-02,
       -4.54499088e-02, -5.58266602e-02,  8.10718313e-02,  2.43032128e-02,
       -8.05576593e-02,  1.99730210e-02, -1.95205472e-02,  4.87695821e-02,
        6.75972924e-02, -1.02569163e-02,  1.34463638e-01, -9.85034741e-03,
        2.71359812e-02, -1.32981520e-02,  1.95574835e-02,  3.53508629e-02,
       -9.78120137e-03,  5.51260933e-02,  4.95274086e-03,  4.98738289e-02,
        2.04959288e-01, -5.51945679e-02,  4.75677997e-02, -1.22955173e-01,
        8.34895000e-02,  2.73876591e-04,  4.38170731e-02, -3.22818272e-02,
       -1.77396368e-02,  1.98409736e-01,  2.49378141e-02, -1.50481209e-01,
        1.59013700e-02, -8.18192363e-02,  1.77855477e-01,  1.27595672e-02,
       -1.72263123e-02,  8.24867934e-03,  9.52711180e-02,  2.94268946e-03,
        1.43057242e-01,  4.75650802e-02,  7.00129345e-02,  5.57481535e-02,
       -1.70518570e-02, -1.88817382e-02,  1.04339093e-01,  4.55024019e-02,
        3.70331593e-02, -2.41914429e-02,  6.08612262e-02,  6.63641766e-02,
       -1.61473751e-02, -9.83985960e-02, -4.87213470e-02,  2.12195292e-02,
        8.78384784e-02,  1.26854092e-01,  8.20072219e-02,  5.51773123e-02,
        5.94680943e-02, -5.54146618e-02,  7.91668072e-02,  8.92544631e-03,
        5.19806577e-04, -9.54308659e-02,  1.72452666e-02, -4.17278819e-02,
        4.76594567e-02, -5.33668362e-02,  1.26463220e-01, -2.71742195e-02,
       -3.63622196e-02, -3.27249914e-02, -3.85552049e-02,  5.12326509e-02,
        4.99561476e-03, -1.35556357e-02, -3.29122879e-02,  7.86489435e-03,
       -5.29906061e-03, -5.27730286e-02,  1.76893640e-02, -6.48731813e-02,
        1.44760972e-02, -2.36608880e-03,  4.81282622e-02, -2.79993340e-02,
        1.15723394e-01,  2.26459727e-02,  1.75362285e-02, -3.06216944e-02,
       -3.40406783e-02, -3.71998213e-02,  5.50636323e-03, -5.61082773e-02], dtype=np.float32)
