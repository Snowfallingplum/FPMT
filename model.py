import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import losses
import utils
from utils import init_net
from networks import Dis, MultiScaleDis
from LPLAnet import LPLANEt, Semantic_corr


class LPLANet(nn.Module):
    def __init__(self, opts):
        super(LPLANet, self).__init__()
        # parameters
        self.opts = opts
        self.lr = opts.lr
        self.batch_size = opts.batch_size
        self.gpu = torch.device('cuda:{}'.format(opts.gpu)) if opts.gpu >= 0 else torch.device('cpu')

        self.weight_corr = opts.weight_corr
        self.weight_bg = opts.weight_bg

        self.weight_self_recL1 = opts.weight_self_recL1
        self.weight_color = opts.weight_color
        self.weight_identity = opts.weight_identity
        self.weight_adv = opts.weight_adv

        if opts.phase == 'train':
            self.FloatTensor = torch.cuda.FloatTensor if opts.gpu >= 0 else torch.FloatTensor
            self.criterion_L1 = nn.L1Loss()
            self.criterion_L2 = nn.MSELoss()
            self.criterion_identity = losses.GPLoss()
            self.criterion_color = losses.ColorLoss(cluster_number=32)
            self.criterion_adv = losses.GANLoss(gan_mode=opts.gan_mode, tensor=self.FloatTensor).to(self.gpu)

        self._build_model()

    def _build_model(self):
        print('start build LPLANet ')
        if self.opts.dis_scale > 1:
            self.dis = init_net(
                MultiScaleDis((self.opts.num_high + 2) * 3, self.opts.dis_scale, norm=self.opts.dis_norm,
                              sn=self.opts.dis_sn),
                self.gpu, init_type=self.opts.init_type, gain=0.02)
        else:
            self.dis = init_net(Dis((self.opts.num_high + 2) * 3, norm=self.opts.dis_norm, sn=self.opts.dis_sn),
                                self.gpu,
                                init_type=self.opts.init_type, gain=0.02)

        self.gen = init_net(
            LPLANEt(img_dim=self.opts.img_dim, parse_dim=self.opts.parse_dim, num_high=self.opts.num_high), self.gpu,
            init_type=self.opts.init_type, gain=0.02)

        self.dis_opt = torch.optim.Adam(self.dis.parameters(), lr=self.lr, betas=(0.5, 0.99), weight_decay=0.0001)
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=self.lr, betas=(0.5, 0.99), weight_decay=0.0001)

        print('finish build LPLANet')

    def load_data(self, data):
        self.non_makeup_img_raw = data['non_makeup_img'].to(self.gpu).detach()
        self.non_makeup_split_parse_raw = data['non_makeup_split_parse'].to(self.gpu).detach()
        self.non_makeup_all_mask_raw = data['non_makeup_all_mask'].to(self.gpu).detach()
        self.non_makeup_face_mask_raw = data['non_makeup_face_mask'].to(self.gpu).detach()
        self.non_makeup_brow_mask_raw = data['non_makeup_brow_mask'].to(self.gpu).detach()
        self.non_makeup_eye_mask_raw = data['non_makeup_eye_mask'].to(self.gpu).detach()
        self.non_makeup_lip_mask_raw = data['non_makeup_lip_mask'].to(self.gpu).detach()

        self.makeup_img_raw = data['makeup_img'].to(self.gpu).detach()
        self.makeup_split_parse_raw = data['makeup_split_parse'].to(self.gpu).detach()
        self.makeup_all_mask_raw = data['makeup_all_mask'].to(self.gpu).detach()
        self.makeup_face_mask_raw = data['makeup_face_mask'].to(self.gpu).detach()
        self.makeup_brow_mask_raw = data['makeup_brow_mask'].to(self.gpu).detach()
        self.makeup_eye_mask_raw = data['makeup_eye_mask'].to(self.gpu).detach()
        self.makeup_lip_mask_raw = data['makeup_lip_mask'].to(self.gpu).detach()

        self.makeup_change_img_raw = data['makeup_change_img'].to(self.gpu).detach()
        self.makeup_change_img_gt_raw = data['makeup_change_img_gt'].to(self.gpu).detach()
        self.makeup_split_warp_parse_raw = data['makeup_split_parse_warp'].to(self.gpu).detach()
        self.makeup_all_warp_mask_raw = data['makeup_all_warp_mask'].to(self.gpu).detach()
        self.makeup_face_warp_mask_raw = data['makeup_face_warp_mask'].to(self.gpu).detach()
        self.makeup_brow_warp_mask_raw = data['makeup_brow_warp_mask'].to(self.gpu).detach()
        self.makeup_eye_warp_mask_raw = data['makeup_eye_warp_mask'].to(self.gpu).detach()
        self.makeup_lip_warp_mask_raw = data['makeup_lip_warp_mask'].to(self.gpu).detach()

    def check_data(self):
        for i in range(self.opts.parse_dim):
            index = torch.nonzero(self.non_makeup_split_parse_raw[:, i, ::])
            if index.shape[0] == 0:
                return False
            index = torch.nonzero(self.makeup_split_parse_raw[:, i, ::])
            if index.shape[0] == 0:
                return False
            index = torch.nonzero(self.makeup_split_warp_parse_raw[:, i, ::])
            if index.shape[0] == 0:
                return False
        self.non_makeup_img = self.non_makeup_img_raw
        self.non_makeup_split_parse = self.non_makeup_split_parse_raw
        self.non_makeup_all_mask = self.non_makeup_all_mask_raw
        self.non_makeup_face_mask = self.non_makeup_face_mask_raw
        self.non_makeup_brow_mask = self.non_makeup_brow_mask_raw
        self.non_makeup_eye_mask = self.non_makeup_eye_mask_raw
        self.non_makeup_lip_mask = self.non_makeup_lip_mask_raw

        self.makeup_img = self.makeup_img_raw
        self.makeup_split_parse = self.makeup_split_parse_raw
        self.makeup_all_mask = self.makeup_all_mask_raw
        self.makeup_face_mask = self.makeup_face_mask_raw
        self.makeup_brow_mask = self.makeup_brow_mask_raw
        self.makeup_eye_mask = self.makeup_eye_mask_raw
        self.makeup_lip_mask = self.makeup_lip_mask_raw

        self.makeup_change_img = self.makeup_change_img_raw
        self.makeup_change_img_gt = self.makeup_change_img_gt_raw
        self.makeup_split_warp_parse = self.makeup_split_warp_parse_raw
        self.makeup_all_warp_mask = self.makeup_all_warp_mask_raw
        self.makeup_face_warp_mask = self.makeup_face_warp_mask_raw
        self.makeup_brow_warp_mask = self.makeup_brow_warp_mask_raw
        self.makeup_eye_warp_mask = self.makeup_eye_warp_mask_raw
        self.makeup_lip_warp_mask = self.makeup_lip_warp_mask_raw
        return True

    def forward(self):
        #  Makeup transfer
        _, self.transfer_pyr, self.transfer_img, self.transfer_aligned_ref_img = self.gen(source_img=self.non_makeup_img,
                                                                                       source_parse=self.non_makeup_split_parse,
                                                                                       ref_img=self.makeup_img,
                                                                                       ref_parse=self.makeup_split_parse,
                                                                                       is_train=True)

        self.cycle_ref_img = self.gen.trans_low.aligned_only(source_img=self.makeup_img,
                                                             source_parse=self.makeup_split_parse,
                                                             ref_img=self.transfer_img,
                                                             ref_parse=self.non_makeup_split_parse)

        # Self supervision
        _, _, self.rec_img, self.rec_aligned_ref_img = self.gen(source_img=self.makeup_img,
                                                             source_parse=self.makeup_split_parse,
                                                             ref_img=self.makeup_change_img,
                                                             ref_parse=self.makeup_split_warp_parse,
                                                             is_train=True)

        self.cycle_ref_img2 = self.gen.trans_low.aligned_only(source_img=self.makeup_change_img,
                                                              source_parse=self.makeup_split_warp_parse,
                                                              ref_img=self.rec_img,
                                                              ref_parse=self.makeup_split_parse)

    def update_D(self):
        self.dis_opt.zero_grad()
        makeup_pyr = self.gen.lap_pyramid.pyramid_decom(self.makeup_img)
        real = [self.makeup_img]
        for i in makeup_pyr:
            temp = F.interpolate(i, size=self.makeup_img.shape[-2:])
            real.append(temp)
        fake = [self.transfer_img]
        for j in self.transfer_pyr:
            temp = F.interpolate(j, size=self.transfer_img.shape[-2:])
            fake.append(temp)

        loss_dis = self.backward_D(self.dis, torch.cat(real, dim=1), torch.cat(fake, dim=1))
        self.loss_dis = loss_dis.item()
        self.dis_opt.step()

    def backward_D(self, netD, real, fake):
        pred_fake = netD.forward(fake.detach())
        pred_real = netD.forward(real)
        ad_true_loss = self.criterion_adv(pred_real, target_is_real=True, for_discriminator=True)
        ad_fake_loss = self.criterion_adv(pred_fake, target_is_real=False, for_discriminator=True)
        loss_D = ad_true_loss + ad_fake_loss
        loss_D.backward()
        return loss_D

    def update_G(self):
        self.gen_opt.zero_grad()
        self.backward_G()
        self.gen_opt.step()

    def backward_G_GAN(self, fake, netD):
        outs_fake = netD.forward(fake)
        loss = self.criterion_adv(outs_fake, target_is_real=True, for_discriminator=False)
        return loss

    def backward_G(self):

        # rec
        loss_G_rec = self.criterion_L1(self.rec_img, self.makeup_change_img_gt)
        loss_G_rec = loss_G_rec * self.weight_self_recL1

        # identity
        non_makeup_img_HF=self.gen.lap_pyramid.pyramid_decom(self.non_makeup_img)[:-1]
        transfer_img_HF=self.gen.lap_pyramid.pyramid_decom(self.transfer_img)[:-1]
        
        loss_G_identity=0.0
        decay_coefficient=0.8

        for i in range(len(non_makeup_img_HF)):
            loss_G_identity+=decay_coefficient*self.criterion_identity(non_makeup_img_HF[i],transfer_img_HF[i])
            decay_coefficient=decay_coefficient*decay_coefficient
        loss_G_identity=loss_G_identity/(len(non_makeup_img_HF))*self.weight_identity

        # loss_G_identity = self.criterion_identity(self.transfer_img, self.non_makeup_img)

        # loss_G_identity = loss_G_identity * self.weight_identity

        # back
        loss_G_bg = self.criterion_L1(self.transfer_img * (1. - self.non_makeup_all_mask),
                                      self.non_makeup_img * (1. - self.non_makeup_all_mask))
        loss_G_bg = loss_G_bg * self.weight_bg

        # color
        loss_G_color_face = self.criterion_color(self.transfer_img, self.non_makeup_face_mask, self.makeup_img,
                                                 self.makeup_face_mask)
        loss_G_color_eye = self.criterion_color(self.transfer_img, self.non_makeup_eye_mask, self.makeup_img,
                                                self.makeup_eye_mask)
        loss_G_color_lip = self.criterion_color(self.transfer_img, self.non_makeup_lip_mask, self.makeup_img,
                                                self.makeup_lip_mask)

        loss_G_color = (loss_G_color_face * 0.2 + loss_G_color_eye + loss_G_color_lip) * 0.33
        loss_G_color = loss_G_color * self.weight_color

        # adv
        fake = [self.transfer_img]
        for j in self.transfer_pyr:
            temp = F.interpolate(j, size=self.transfer_img.shape[-2:])
            fake.append(temp)
        loss_G_GAN = self.backward_G_GAN(torch.cat(fake, dim=1), self.dis)
        loss_G_GAN = loss_G_GAN * self.weight_adv

        # corr
        # pyr_makeup_gt = self.gen.lap_pyramid.pyramid_decom(img=self.makeup_change_img_gt)
        total_makeup_mask = self.makeup_face_mask + self.makeup_eye_mask + self.makeup_lip_mask + self.makeup_brow_mask
        total_makeup_mask = F.interpolate(total_makeup_mask, size=self.cycle_ref_img.shape[-2:], mode='nearest')
        makeup_img_down = F.interpolate(self.makeup_img, size=self.cycle_ref_img.shape[-2:], mode='nearest')

        total_makeup_warp_mask = self.makeup_face_warp_mask + self.makeup_eye_warp_mask + self.makeup_lip_warp_mask + self.makeup_brow_warp_mask
        total_makeup_warp_mask = F.interpolate(total_makeup_warp_mask, size=self.cycle_ref_img2.shape[-2:],
                                               mode='nearest')
        makeup_change_img_down = F.interpolate(self.makeup_change_img, size=self.cycle_ref_img2.shape[-2:],
                                               mode='nearest')

        loss_G_corr = self.criterion_L1(makeup_img_down * total_makeup_mask, self.cycle_ref_img * total_makeup_mask) + \
                      self.criterion_L1(makeup_change_img_down * total_makeup_warp_mask,
                                        self.cycle_ref_img2 * total_makeup_warp_mask)

        loss_G_corr = loss_G_corr * self.weight_corr

        loss_G = loss_G_rec + loss_G_GAN + loss_G_color + loss_G_identity + loss_G_corr + loss_G_bg

        loss_G.backward()

        self.loss_G = loss_G.item()
        self.loss_G_rec = loss_G_rec.item()
        self.loss_G_color = loss_G_color.item()
        self.loss_G_identity = loss_G_identity.item()
        self.loss_G_GAN = loss_G_GAN.item()
        self.loss_G_corr = loss_G_corr.item()

    def set_scheduler(self, opts, last_ep=0):
        self.dis_sch = utils.get_scheduler(self.dis_opt, opts, last_ep)
        self.gen_sch = utils.get_scheduler(self.gen_opt, opts, last_ep)

    def update_lr(self):
        self.dis_sch.step()
        self.gen_sch.step()

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir)
        self.gen.load_state_dict(checkpoint['gen'], strict=False)
        return checkpoint['ep'], checkpoint['total_it']

    def save(self, filename, ep, total_it):
        state = {
            'gen': self.gen.state_dict(),
            'total_it': total_it,
            'ep': ep
        }
        torch.save(state, filename)
        return

    def normalize_image(self, x):
        return x[:, 0:3, :, :]

    def assemble_outputs(self):

        non_makeup_img = self.normalize_image(self.non_makeup_img).detach()
        makeup_img = self.normalize_image(self.makeup_img).detach()

        transfer_aligned_ref_img = F.interpolate(self.transfer_aligned_ref_img, size=non_makeup_img.shape[-2:],
                                                 mode='nearest')
        transfer_aligned_ref_img = self.normalize_image(transfer_aligned_ref_img).detach()

        cycle_ref_img = F.interpolate(self.cycle_ref_img, size=non_makeup_img.shape[-2:],
                                      mode='nearest')
        cycle_ref_img = self.normalize_image(cycle_ref_img).detach()
        transfer_img = self.normalize_image(self.transfer_img).detach()

        non_makeup_face_mask = self.normalize_image(self.non_makeup_face_mask).detach()
        makeup_face_mask = self.normalize_image(self.makeup_face_mask).detach()

        makeup_change_img = self.normalize_image(self.makeup_change_img).detach()
        makeup_change_img_gt = self.normalize_image(self.makeup_change_img_gt).detach()

        pyr_makeup_gt = self.gen.lap_pyramid.pyramid_decom(img=self.makeup_change_img_gt)
        total_makeup_mask = self.makeup_face_mask + self.makeup_eye_mask + self.makeup_lip_mask + self.makeup_brow_mask
        total_makeup_mask = F.interpolate(total_makeup_mask, size=pyr_makeup_gt[-1].shape[-2:], mode='nearest')
        rec_aligned_ref_img_gt = pyr_makeup_gt[-1] * total_makeup_mask

        rec_aligned_ref_img_gt = F.interpolate(rec_aligned_ref_img_gt, size=makeup_img.shape[-2:],
                                               mode='nearest')
        rec_aligned_ref_img_gt = self.normalize_image(rec_aligned_ref_img_gt).detach()

        rec_aligned_ref_img = F.interpolate(self.rec_aligned_ref_img, size=makeup_img.shape[-2:],
                                            mode='nearest')
        rec_aligned_ref_img = self.normalize_image(rec_aligned_ref_img).detach()

        cycle_ref_img2 = F.interpolate(self.cycle_ref_img2, size=makeup_img.shape[-2:],
                                       mode='nearest')
        cycle_ref_img2 = self.normalize_image(cycle_ref_img2).detach()
        rec_img = self.normalize_image(self.rec_img)
        # print(transfer_aligned_ref_img.shape)
        # print(cycle_ref_img.shape)

        row1 = torch.cat((non_makeup_img[0:1, ::], makeup_img[0:1, ::],
                          transfer_aligned_ref_img[0:1, ::], cycle_ref_img[0:1, ::], transfer_img[0:1, ::],
                          non_makeup_face_mask[0:1, ::],
                          makeup_face_mask[0:1, ::]), 3)

        row2 = torch.cat((makeup_img[0:1, ::], makeup_change_img[0:1, ::],
                          rec_aligned_ref_img[0:1, ::], rec_aligned_ref_img_gt[0:1, ::], cycle_ref_img2[0:1, ::],
                          rec_img[0:1, ::], makeup_change_img_gt[0:1, ::]), 3)
        return torch.cat((row1, row2), 2)

    def test_load_data(self, data):
        self.non_makeup_img_raw = data['non_makeup_img'].to(self.gpu).detach()
        self.non_makeup_split_parse_raw = data['non_makeup_split_parse'].to(self.gpu).detach()
        self.non_makeup_all_mask_raw = data['non_makeup_all_mask'].to(self.gpu).detach()
        self.non_makeup_face_mask_raw = data['non_makeup_face_mask'].to(self.gpu).detach()
        self.non_makeup_brow_mask_raw = data['non_makeup_brow_mask'].to(self.gpu).detach()
        self.non_makeup_eye_mask_raw = data['non_makeup_eye_mask'].to(self.gpu).detach()
        self.non_makeup_lip_mask_raw = data['non_makeup_lip_mask'].to(self.gpu).detach()

        self.makeup_img_raw = data['makeup_img'].to(self.gpu).detach()
        self.makeup_split_parse_raw = data['makeup_split_parse'].to(self.gpu).detach()
        self.makeup_all_mask_raw = data['makeup_all_mask'].to(self.gpu).detach()
        self.makeup_face_mask_raw = data['makeup_face_mask'].to(self.gpu).detach()
        self.makeup_brow_mask_raw = data['makeup_brow_mask'].to(self.gpu).detach()
        self.makeup_eye_mask_raw = data['makeup_eye_mask'].to(self.gpu).detach()
        self.makeup_lip_mask_raw = data['makeup_lip_mask'].to(self.gpu).detach()

    def test_check_data(self):
        # for i in range(self.opts.parse_dim):
        #     index = torch.nonzero(self.non_makeup_split_parse_raw[:, i, ::])
        #     if index.shape[0] == 0:
        #         return False
        #     index = torch.nonzero(self.makeup_split_parse_raw[:, i, ::])
        #     if index.shape[0] == 0:
        #         return False
        self.non_makeup_img = self.non_makeup_img_raw
        self.non_makeup_split_parse = self.non_makeup_split_parse_raw
        self.non_makeup_all_mask = self.non_makeup_all_mask_raw
        self.non_makeup_face_mask = self.non_makeup_face_mask_raw
        self.non_makeup_brow_mask = self.non_makeup_brow_mask_raw
        self.non_makeup_eye_mask = self.non_makeup_eye_mask_raw
        self.non_makeup_lip_mask = self.non_makeup_lip_mask_raw

        self.makeup_img = self.makeup_img_raw
        self.makeup_split_parse = self.makeup_split_parse_raw
        self.makeup_all_mask = self.makeup_all_mask_raw
        self.makeup_face_mask = self.makeup_face_mask_raw
        self.makeup_brow_mask = self.makeup_brow_mask_raw
        self.makeup_eye_mask = self.makeup_eye_mask_raw
        self.makeup_lip_mask = self.makeup_lip_mask_raw

        return True

    def test_pair(self):
        with torch.no_grad():

            self.mask_pyr,self.trans_pyr, self.transfer_img, self.transfer_aligned_ref_img = self.gen(
                source_img=self.non_makeup_img,
                source_parse=self.non_makeup_split_parse,
                ref_img=self.makeup_img,
                ref_parse=self.makeup_split_parse,
                is_train=True)


        non_makeup_img = self.normalize_image(self.non_makeup_img).detach()
        makeup_img = self.normalize_image(self.makeup_img).detach()

        transfer_aligned_ref_img = F.interpolate(self.transfer_aligned_ref_img, size=non_makeup_img.shape[-2:],
                                                 mode='nearest')
        transfer_aligned_ref_img = self.normalize_image(transfer_aligned_ref_img).detach()
        # cycle_ref_img = self.normalize_image(self.cycle_ref_img).detach()
        transfer_img = self.normalize_image(self.transfer_img).detach()
        

        row1 = torch.cat((non_makeup_img[0:1, ::], makeup_img[0:1, ::],
                          transfer_aligned_ref_img[0:1, ::], transfer_img[0:1, ::]), 3)

        return row1

   