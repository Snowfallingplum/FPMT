import torch
import numpy as np
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import utils

# Generative adversarial loss
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self, vgg_normal_correct=True):
        super(VGGLoss, self).__init__()
        self.vgg_normal_correct = vgg_normal_correct
        if vgg_normal_correct:
            self.vgg = utils.VGG19_feature(vgg_normal_correct=True).cuda()
        else:
            self.vgg = utils.VGG19_feature().cuda()
        self.vgg.load_state_dict(torch.load('../../VGG_Model/vgg19_conv.pth'))
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        if self.vgg_normal_correct:
            x_vgg, y_vgg = self.vgg(x, ['r11', 'r21', 'r31', 'r41', 'r51'], preprocess=True), self.vgg(y, ['r11', 'r21',
                                                                                                           'r31', 'r41',
                                                                                                           'r51'],
                                                                                                       preprocess=True)
        else:
            x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

# identity loss
class SPLoss(nn.Module):
    def __init__(self):
        super(SPLoss, self).__init__()

    def __call__(self, input, reference):
        a = torch.sum(
            torch.sum(F.normalize(input, p=2, dim=2) * F.normalize(reference, p=2, dim=2), dim=2, keepdim=True))
        b = torch.sum(
            torch.sum(F.normalize(input, p=2, dim=3) * F.normalize(reference, p=2, dim=3), dim=3, keepdim=True))
        return -(a + b) / input.size(2)


class GPLoss(nn.Module):
    def __init__(self):
        super(GPLoss, self).__init__()
        self.trace = SPLoss()

    def get_image_gradients(self, input):
        f_v_1 = F.pad(input, (0, -1, 0, 0))
        f_v_2 = F.pad(input, (-1, 0, 0, 0))
        f_v = f_v_1 - f_v_2

        f_h_1 = F.pad(input, (0, 0, 0, -1))
        f_h_2 = F.pad(input, (0, 0, -1, 0))
        f_h = f_h_1 - f_h_2

        return f_v, f_h

    def __call__(self, input, reference):
        # comment these lines when you inputs and outputs are in [0,1] range already
        input = (input + 1) / 2
        reference = (reference + 1) / 2

        input_v, input_h = self.get_image_gradients(input)
        ref_v, ref_h = self.get_image_gradients(reference)

        trace_v = self.trace(input_v, ref_v)
        trace_h = self.trace(input_h, ref_h)
        return trace_v + trace_h


#  color loss that uses color histogram
class ColorLoss(nn.Module):
    def __init__(self,cluster_number=32,img_size=256):
        super(ColorLoss, self).__init__()
        self.cluster_number=cluster_number
        self.criterion = nn.L1Loss()
        self.spacing=2/cluster_number
        self.img_size=img_size

    def calc_hist(self,data_ab):
        H = data_ab.size(0)
        grid_a = torch.linspace(-1, 1, self.cluster_number + 1).cuda()
        grid_a = grid_a.view(self.cluster_number + 1, 1).expand(self.cluster_number + 1, H).cuda()
        hist_a = torch.max(self.spacing - torch.abs(grid_a - data_ab.view(-1)), torch.Tensor([0]).cuda()) * 10
        # return hist_a.mean(dim=1).view(-1) * H
        return hist_a.mean(dim=1).view(-1)  # removal H
    def forward(self,A_img,A_mask,B_img,B_mask):
        A_img=F.interpolate(A_img,size=(self.img_size,self.img_size),mode='bilinear')
        B_img=F.interpolate(B_img,size=(self.img_size,self.img_size),mode='bilinear')

        A_mask=F.interpolate(A_mask,size=(self.img_size,self.img_size),mode='nearest')
        B_mask=F.interpolate(B_mask,size=(self.img_size,self.img_size),mode='nearest')

        b,c,h,w=A_img.size()
        loss = 0
        for j in range(b):
            for i in range(3):
                temp_A = torch.masked_select(A_img[j,i:i+1,::],A_mask[j,i:i+1,::]>0.5).cuda()
                temp_B = torch.masked_select(B_img[j, i:i + 1, ::], B_mask[j, i:i + 1, ::] > 0.5).cuda()
                if temp_A.size(0)==0 or temp_B.size(0)==0:
                    continue
                loss+=self.criterion(self.calc_hist(temp_A),self.calc_hist(temp_B))
        return loss/(b*3)




