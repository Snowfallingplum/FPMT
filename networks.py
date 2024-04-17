import torch
import torch.nn as nn
import torch.nn.functional as F


####################################################################
# ------------------------- Discriminators --------------------------
####################################################################
class Dis(nn.Module):
    def __init__(self, input_dim, norm='None', sn=False):
        super(Dis, self).__init__()
        ch = 64
        n_layer = 5
        self.model = self._make_net(ch, input_dim, n_layer, norm, sn)

    def _make_net(self, ch, input_dim, n_layer, norm, sn):
        model = []
        model += [LeakyReLUConv2d(input_dim, ch, kernel_size=3, stride=2, padding=1, norm=norm, sn=sn)]
        tch = ch
        for i in range(1, n_layer):
            model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1, norm=norm, sn=sn)]
            tch *= 2
        model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1, norm='None', sn=sn)]
        tch *= 2
        if sn:
            model += [nn.utils.spectral_norm(nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0))]
        else:
            model += [nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0)]
        model += [nn.Sigmoid()]
        return nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)

        out = out.view(-1)
        outs = []
        outs.append(out)
        return outs


class MultiScaleDis(nn.Module):
    def __init__(self, input_dim, n_scale=3, n_layer=4, norm='None', sn=False):
        super(MultiScaleDis, self).__init__()
        ch = 64
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.Diss = nn.ModuleList()
        for _ in range(n_scale):
            self.Diss.append(self._make_net(ch, input_dim, n_layer, norm, sn))

    def _make_net(self, ch, input_dim, n_layer, norm, sn):
        model = []
        model += [LeakyReLUConv2d(input_dim, ch, 4, 2, 1, norm, sn)]
        tch = ch
        for _ in range(1, n_layer):
            model += [LeakyReLUConv2d(tch, tch * 2, 4, 2, 1, norm, sn)]
            tch *= 2
        if sn:
            model += [nn.utils.spectral_norm(nn.Conv2d(tch, 1, 1, 1, 0))]
        else:
            model += [nn.Conv2d(tch, 1, 1, 1, 0)]
        model += [nn.Sigmoid()]
        return nn.Sequential(*model)

    def forward(self, x):
        outs = []
        for Dis in self.Diss:
            outs.append(Dis(x))
            x = self.downsample(x)
        return outs


#############################################################
#                 Laplacian Pyramid
#############################################################
class Lap_Pyramid_Bicubic(nn.Module):
    """
        Laplacian Pyramid Based on Bicubic Interpolation
    """

    def __init__(self, num_high=3):
        super(Lap_Pyramid_Bicubic, self).__init__()
        self.interpolate_mode = 'bicubic'
        self.num_high = num_high

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for i in range(self.num_high):
            down = nn.functional.interpolate(current, size=(current.shape[2] // 2, current.shape[3] // 2),
                                             mode=self.interpolate_mode, align_corners=True)
            up = nn.functional.interpolate(down, size=(current.shape[2], current.shape[3]), mode=self.interpolate_mode,
                                           align_corners=True)
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[-1]
        for level in reversed(pyr[:-1]):
            image = F.interpolate(image, size=(level.shape[2], level.shape[3]), mode=self.interpolate_mode,
                                  align_corners=True) + level
        return image


class Lap_Pyramid_Conv(nn.Module):
    """
        Convolutional Laplacian Pyramid Based on Predefined Kernel Functions
    """

    def __init__(self, num_high=3):
        super(Lap_Pyramid_Conv, self).__init__()

        self.num_high = num_high
        self.kernel = self.gauss_kernel()

    def gauss_kernel(self, device=torch.device('cuda'), channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel

    def downsample(self, x):
        return x[:, :, ::2, ::2]  # down-sampling

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            up = self.upsample(down)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[-1]
        for level in reversed(pyr[:-1]):
            up = self.upsample(image)
            if up.shape[2] != level.shape[2] or up.shape[3] != level.shape[3]:
                up = nn.functional.interpolate(up, size=(level.shape[2], level.shape[3]))
            image = up + level
        return image


#############################################################
#                       FeatureExtractor
#############################################################

class FeatureExtractor(nn.Module):
    """
        Fused Features, (image, parse, coordinate)
    """

    def __init__(self, img_dim, parse_dim):
        super(FeatureExtractor, self).__init__()
        self.img_dim = img_dim
        self.parse_dim = parse_dim
        self.input_dim = img_dim + parse_dim
        self.block = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        )

    def forward(self, x, x_parse):
        out = torch.cat([x, x_parse], dim=1)
        out = self.block(out)
        return out


class SemanticExtractor(nn.Module):
    """
        Semantic Extractor
    """

    def __init__(self, parse_dim):
        super(SemanticExtractor, self).__init__()
        self.parse_dim = parse_dim
        ngf = 32
        self.conv_1 = nn.Sequential(
            nn.Conv2d(self.parse_dim + 2, ngf, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.conv_2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.conv_3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.conv_4 = nn.Sequential(
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)

        self.deconv_1 = nn.Sequential(
            nn.Conv2d(ngf * 4, ngf * 2, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.deconv_2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 2, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def connect_coordinates(self, input_x):
        x_range = torch.linspace(-1, 1, input_x.shape[-1], device=input_x.device)
        y_range = torch.linspace(-1, 1, input_x.shape[-2], device=input_x.device)
        y_coor, x_corr = torch.meshgrid(y_range, x_range)
        y_coor = y_coor.expand([input_x.shape[0], 1, -1, -1])
        x_corr = x_corr.expand([input_x.shape[0], 1, -1, -1])
        coord = torch.cat([x_corr, y_coor], 1)
        output = torch.cat([input_x, coord], 1)
        return output

    def forward(self, x_parse):
        x_parse = F.interpolate(x_parse, size=(256, 256), mode='nearest')
        out = self.connect_coordinates(x_parse)
        conv_1 = self.conv_1(out)
        conv_2 = self.conv_2(conv_1)
        conv_3 = self.conv_3(conv_2)
        conv_4 = self.conv_4(conv_3)
        deconv_1 = self.deconv_1(self.upsampling(conv_4) + conv_3)
        deconv_2 = self.deconv_2(self.upsampling(deconv_1) + conv_2)
        return deconv_2


#############################################################
#                    Local semantic alignment
#############################################################

class LocalSemanticAlignment(nn.Module):
    def __init__(self, ):
        super(LocalSemanticAlignment, self).__init__()
        self.softmax_alpha = 100

    def warp(self, unalign_fb, fa, fa_parse, fb, fb_parse, alpha):
        '''
            calculate correspondence matrix and warp the exemplar features
        '''
        assert fa.shape == fb.shape, \
            'Feature shape must match. Got %s in a and %s in b)' % (fa.shape, fb.shape)
        n, c, h, w = fa.shape
        _, c1, _, _ = fa_parse.shape
        _, c2, _, _ = unalign_fb.shape
        # subtract mean
        fa = fa - torch.mean(fa, dim=(2, 3), keepdim=True)
        fb = fb - torch.mean(fb, dim=(2, 3), keepdim=True)

        # vectorize (merge dim H, W) and normalize channelwise vectors
        fa = fa.view(n, c, -1)  # n c hw
        fb = fb.view(n, c, -1)  # n c hw
        fa = fa / torch.norm(fa, dim=1, keepdim=True)
        fb = fb / torch.norm(fb, dim=1, keepdim=True)

        unalign_fb = unalign_fb.view(n, c2, -1)
        aligned_fb = torch.zeros_like(unalign_fb)

        fa_parse = fa_parse.view(n, c1, -1)  # n c1 hw
        fb_parse = fb_parse.view(n, c1, -1)  # n c1 hw

        for i in range(1, c1):
            a_index = torch.nonzero(fa_parse[:, i, :])
            b_index = torch.nonzero(fb_parse[:, i, :])
            local_fa = fa[a_index[:, 0], :, a_index[:, 1]]  # np1 c
            local_fa = local_fa.contiguous().view(n, -1, c).transpose(-2, -1)  # n c p1
            local_fb = fb[b_index[:, 0], :, b_index[:, 1]]  # np2 c
            local_fb = local_fb.contiguous().view(n, -1, c).transpose(-2, -1)  # n c p2
            # print(local_fb.shape)
            energy_ab_T = torch.bmm(local_fb.transpose(-2, -1), local_fa) * alpha  # n p2 c * n c p1 -> n p2 p1
            corr_ab_T = F.softmax(energy_ab_T, dim=1)  # n p2 c * n c p1 -> n p2 p1
            local_unalign_fb = unalign_fb[b_index[:, 0], :, b_index[:, 1]]  # n c2 p2
            local_unalign_fb = local_unalign_fb.contiguous().view(n, -1, c2).transpose(-2, -1)  # n c2 p2
            local_aligned_fb = torch.bmm(local_unalign_fb.contiguous(), corr_ab_T)  # n c2 p2 * n p2 p1-> n c2 p1
            local_aligned_fb = local_aligned_fb.transpose(-2, -1).view(-1, c2)
            aligned_fb[a_index[:, 0], :, a_index[:, 1]] = local_aligned_fb

        aligned_fb = aligned_fb.view(n, c2, h, w)

        return aligned_fb

    def forward(self, unalign_fb, fa, fa_parse, fb, fb_parse):
        unalign_fb_raw = unalign_fb
        fa_parse = F.interpolate(fa_parse, size=fa.shape[-2:], mode='nearest')
        fb_parse = F.interpolate(fb_parse, size=fb.shape[-2:], mode='nearest')
        unalign_fb = F.interpolate(unalign_fb, size=fb.shape[-2:], mode='bilinear', align_corners=True)
        aligned_fb = self.warp(unalign_fb, fa, fa_parse, fb, fb_parse, self.softmax_alpha)
        aligned_fb = F.interpolate(aligned_fb, size=unalign_fb_raw.shape[-2:], mode='bilinear', align_corners=True)
        return aligned_fb


#############################################################
#                      Makeup injection
#############################################################
class MakeupInjection(nn.Module):
    def __init__(self):
        super(MakeupInjection, self).__init__()
        self.condition_scale = nn.Sequential(
            nn.Conv2d(64, 64, 1, stride=1, padding=0),
        )
        self.condition_shift = nn.Sequential(
            nn.Conv2d(64, 64, 1, stride=1, padding=0),
        )
        self.conv1_block = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
        )
        self.last_block = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, source_f, aligned_ref_f):
        source_f = self.conv1_block(source_f)
        scale = self.condition_scale(aligned_ref_f)
        shift = self.condition_shift(aligned_ref_f)
        # print(source_f.shape,scale.shape,shift.shape)
        source_f = source_f * scale + shift
        output = self.last_block(source_f)
        return output


#############################################################
#                       basic block
#############################################################
class ResidualBlock(nn.Module):
    def __init__(self, channels, norm=False):
        super(ResidualBlock, self).__init__()
        if norm:
            self.block = nn.Sequential(
                nn.Conv2d(channels, channels, 3, 1, 1),
                nn.InstanceNorm2d(channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(channels, channels, 3, 1, 1),
                nn.InstanceNorm2d(channels)
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(channels, channels, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(channels, channels, 3, 1, 1)
            )

    def forward(self, x):
        return x + self.block(x)


class LeakyReLUConv2d(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, stride, padding, norm='None', sn=False):
        super(LeakyReLUConv2d, self).__init__()
        model = []
        model += [nn.ReflectionPad2d(padding)]
        if sn:
            model += [nn.utils.spectral_norm(
                nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=0, bias=True))]
        else:
            model += [nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
        if norm == 'instance':
            model += [nn.InstanceNorm2d(outplanes, affine=False)]
        model += [nn.LeakyReLU(0.2, inplace=False)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
