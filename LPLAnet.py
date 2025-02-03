import torch
import torch.nn as nn
import torch.nn.functional as F

import networks


class Semantic_corr(nn.Module):
    def __init__(self, parse_dim):
        super(Semantic_corr, self).__init__()
        self.parse_dim = parse_dim
        self.semantic = networks.SemanticExtractor(parse_dim=parse_dim)
        self.atte = networks.Attention()

    def forward(self, source_img, source_parse, ref_img, ref_parse):
        source_semantic = self.semantic(source_parse)
        ref_semantic = self.semantic(ref_parse)
        aligned_ref_img = self.atte(unalign_fb=ref_img, fa=source_semantic, fa_parse=source_parse, fb=ref_semantic,
                                    fb_parse=ref_parse)
        return aligned_ref_img


class Trans_low(nn.Module):
    def __init__(self, img_dim, parse_dim):
        super(Trans_low, self).__init__()
        self.img_dim = img_dim
        self.parse_dim = parse_dim
        self.semantic = networks.SemanticExtractor(parse_dim=parse_dim)
        self.local_align = networks.LocalSemanticAlignment()
        self.content_extract = networks.FeatureExtractor(img_dim, parse_dim)
        self.makeup_extract = networks.FeatureExtractor(img_dim, parse_dim)
        self.makeup_inject = networks.MakeupInjection()

    def forward(self, source_img, source_parse, ref_img, ref_parse, is_train=True):

        source_semantic = self.semantic(source_parse)
        ref_semantic = self.semantic(ref_parse)

        n, c, h, w = source_img.shape
        source_parse = F.interpolate(source_parse, size=(h, w), mode='nearest')
        ref_parse = F.interpolate(ref_parse, size=(h, w), mode='nearest')

        content = self.content_extract(x=source_img, x_parse=source_parse)
        unaligned_makeup = self.makeup_extract(x=ref_img, x_parse=ref_parse)

        if is_train:
            aligned_ref_img = self.local_align(unalign_fb=ref_img, fa=source_semantic, fa_parse=source_parse,
                                               fb=ref_semantic, fb_parse=ref_parse)
            aligned_makeup = self.local_align(unalign_fb=unaligned_makeup, fa=source_semantic, fa_parse=source_parse,
                                              fb=ref_semantic, fb_parse=ref_parse)
            # print(content.shape,aligned_makeup.shape)
            transfer_img = self.makeup_inject(source_f=content, aligned_ref_f=aligned_makeup)
            return transfer_img, aligned_ref_img

        else:
            aligned_makeup = self.local_align(unalign_fb=unaligned_makeup, fa=source_semantic, fa_parse=source_parse,
                                              fb=ref_semantic, fb_parse=ref_parse)
            transfer_img = self.makeup_inject(source_f=content, aligned_ref_f=aligned_makeup)
            return transfer_img

    def aligned_only(self,source_img, source_parse, ref_img, ref_parse):
        source_semantic = self.semantic(source_parse)
        ref_semantic = self.semantic(ref_parse)

        n, c, h, w = source_img.shape
        source_parse = F.interpolate(source_parse, size=(h, w), mode='nearest')
        ref_parse = F.interpolate(ref_parse, size=(h, w), mode='nearest')

        aligned_ref_img = self.local_align(unalign_fb=ref_img, fa=source_semantic, fa_parse=source_parse,
                                           fb=ref_semantic, fb_parse=ref_parse)
        return aligned_ref_img

class Mask_block(nn.Module):
    def __init__(self,channels=32):
        super(Mask_block, self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(6, channels, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channels, 3, 3, padding=1),
            nn.Sigmoid())
    def forward(self,x):
        return self.model(x)

class Gate_block(nn.Module):
    def __init__(self,channels=32):
        super(Gate_block, self).__init__()
        self.conv_1=nn.Conv2d(3, channels, 3, padding=1)

        self.condition_scale = nn.Sequential(
            nn.Conv2d(3, channels, 1, stride=1, padding=0),
        )
        self.condition_shift = nn.Sequential(
            nn.Conv2d(3, channels, 1, stride=1, padding=0),
        )
        self.conv_2 = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self,x,y):
        out=self.conv_1(x)
        scale=self.condition_scale(y)
        shift = self.condition_shift(y)
        out=out*scale+shift
        out=self.conv_2(out)
        return out

class Trans_high(nn.Module):
    def __init__(self, num_residual_blocks,pyr_recon_f,num_high=3):
        super(Trans_high, self).__init__()

        self.pyr_recon=pyr_recon_f

        self.num_high = num_high

        model = [nn.Conv2d(6, 64, 3, padding=1),
                 nn.LeakyReLU()]

        for _ in range(num_residual_blocks):
            model += [networks.ResidualBlock(channels=64, norm=False)]

        model += [nn.Conv2d(64, 3, 3, padding=1),
                  nn.Sigmoid()]

        self.trans_mask_block_0 = nn.Sequential(*model)
        self.trans_gate_block_0 = Gate_block()

        for i in range(1, self.num_high):
            setattr(self, 'trans_mask_block_{}'.format(str(i)), Mask_block())
            setattr(self, 'trans_gate_block_{}'.format(str(i)), Gate_block())

            # trans_mask_block = nn.Sequential(
            #     nn.Conv2d(6, 16, 3, padding=1),
            #     nn.LeakyReLU(),
            #     nn.Conv2d(16, 3, 3, padding=1))
            # setattr(self, 'trans_res_block_{}'.format(str(i)), trans_mask_block)

    def forward(self, x, pyr_original, fake_low):

        pyr_result = [fake_low]
        mask_result=[]
        mask0 = self.trans_mask_block_0(x)
        fake_low_up = nn.functional.interpolate(fake_low, size=(pyr_original[-2].shape[2], pyr_original[-2].shape[3]))
        res0=self.trans_gate_block_0(pyr_original[-2]*mask0,fake_low_up)
        result_highfreq0 = pyr_original[-2] + res0
        pyr_result.insert(0,result_highfreq0)
        mask=mask0
        mask_result.insert(0,mask)
        for i in range(1, self.num_high):
            mask = nn.functional.interpolate(mask, size=(pyr_original[-2 - i].shape[2], pyr_original[-2 - i].shape[3]))
            self.trans_mask_block = getattr(self, 'trans_mask_block_{}'.format(str(i)))
            mask=self.trans_mask_block(torch.cat([pyr_original[-2- i],mask],dim=1))
            current_img=self.pyr_recon(pyr_result)
            current_img_up = nn.functional.interpolate(current_img,
                                                    size=(pyr_original[-2- i].shape[2], pyr_original[-2- i].shape[3]))
            self.trans_gate_block = getattr(self, 'trans_gate_block_{}'.format(str(i)))
            res=self.trans_gate_block(pyr_original[-2- i]*mask,current_img_up)
            result_highfreq = pyr_original[-2-i] + res
            pyr_result.insert(0, result_highfreq)
            mask_result.insert(0, mask)


        # setattr(self, 'result_highfreq_{}'.format(str(0)), result_highfreq)
        #
        # for i in range(1, self.num_high):
        #     res = nn.functional.interpolate(res, size=(pyr_original[-2 - i].shape[2], pyr_original[-2 - i].shape[3]))
        #     self.trans_res_block = getattr(self, 'trans_res_block_{}'.format(str(i)))
        #     res = self.trans_res_block(torch.cat([pyr_original[-2 - i], res], dim=1))
        #     result_highfreq = pyr_original[-2 - i] + res
        #     setattr(self, 'result_highfreq_{}'.format(str(i)), result_highfreq)
        #
        # for i in reversed(range(self.num_high)):
        #     result_highfreq = getattr(self, 'result_highfreq_{}'.format(str(i)))
        #     pyr_result.append(result_highfreq)
        #
        # pyr_result.append(fake_low)

        return pyr_result,mask_result


class LPLANEt(nn.Module):
    def __init__(self, img_dim, parse_dim, num_high=3):
        super(LPLANEt, self).__init__()

        self.num_high = num_high
        self.lap_pyramid = networks.Lap_Pyramid_Conv(num_high)

        self.trans_low = Trans_low(img_dim, parse_dim)
        self.trans_high = Trans_high(num_residual_blocks=2,pyr_recon_f=self.lap_pyramid.pyramid_recons, num_high=num_high)

    def forward(self, source_img, source_parse, ref_img, ref_parse, is_train=True):
        pyr_source = self.lap_pyramid.pyramid_decom(img=source_img)
        # pyr_ref = self.lap_pyramid.pyramid_decom(img=ref_img)
        ref_img = F.interpolate(ref_img, size=pyr_source[-1].shape[-2:], mode='bilinear', align_corners=True)
        if is_train:

            transfer_low, aligned_ref_img = self.trans_low(source_img=pyr_source[-1], source_parse=source_parse,
                                                           ref_img=ref_img, ref_parse=ref_parse, is_train=True)

            source_low_up = nn.functional.interpolate(pyr_source[-1], size=pyr_source[-2].shape[-2:], mode='nearest')
            transfer_low_up = nn.functional.interpolate(transfer_low, size=pyr_source[-2].shape[-2:], mode='nearest')
            high_with_low = torch.cat([pyr_source[-2], transfer_low_up], 1)
            pyr_trans,pyr_mask = self.trans_high(x=high_with_low, pyr_original=pyr_source, fake_low=transfer_low)
            transfer_img = self.lap_pyramid.pyramid_recons(pyr_trans)
            return pyr_mask,pyr_trans,transfer_img, aligned_ref_img
        else:
            transfer_low = self.trans_low(source_img=pyr_source[-1], source_parse=source_parse,
                                          ref_img=ref_img, ref_parse=ref_parse, is_train=False)

            source_low_up = nn.functional.interpolate(pyr_source[-1], size=pyr_source[-2].shape[-2:], mode='nearest')
            transfer_low_up = nn.functional.interpolate(transfer_low, size=pyr_source[-2].shape[-2:], mode='nearest')
            high_with_low = torch.cat([pyr_source[-2], source_low_up, transfer_low_up], 1)
            pyr_trans,pyr_mask = self.trans_high(x=high_with_low, pyr_original=pyr_source, fake_low=transfer_low)
            transfer_img = self.lap_pyramid.pyramid_recons(pyr_trans)
            return pyr_trans,transfer_img
