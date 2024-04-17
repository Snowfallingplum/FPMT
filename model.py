import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import init_net
from FPMT import FPMTNet


class FPMT(nn.Module):
    def __init__(self, opts):
        super(FPMT, self).__init__()
        # parameters
        self.opts = opts
        self.batch_size = opts.batch_size
        self.gpu = torch.device('cuda:{}'.format(opts.gpu)) if opts.gpu >= 0 else torch.device('cpu')

        self._build_model()

    def _build_model(self):
        print('start build LPLANet ')

        self.gen = init_net(
            FPMTNet(img_dim=self.opts.img_dim, parse_dim=self.opts.parse_dim, num_high=self.opts.num_high), self.gpu,
            init_type=self.opts.init_type, gain=0.02)

        print('finish build LPLANet')

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir)
        self.gen.load_state_dict(checkpoint['gen'], strict=False)
        return checkpoint['ep'], checkpoint['total_it']

    def normalize_image(self, x):
        return x[:, 0:3, :, :]

    def test_load_data(self, data):
        self.non_makeup_img = data['non_makeup_img'].to(self.gpu).detach()
        self.non_makeup_split_parse = data['non_makeup_split_parse'].to(self.gpu).detach()

        self.makeup_img = data['makeup_img'].to(self.gpu).detach()
        self.makeup_split_parse = data['makeup_split_parse'].to(self.gpu).detach()

    def test_pair(self):
        with torch.no_grad():
            self.mask_pyr, self.trans_pyr, self.transfer_img, self.transfer_aligned_ref_img = self.gen(
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
        transfer_img = self.normalize_image(self.transfer_img).detach()

        row1 = torch.cat((non_makeup_img[0:1, ::], makeup_img[0:1, ::],
                          transfer_aligned_ref_img[0:1, ::], transfer_img[0:1, ::]), 3)

        return row1
