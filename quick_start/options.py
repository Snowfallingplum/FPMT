import argparse


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # data loader related
        self.parser.add_argument('--data_root', type=str, default='./MT_datasets/train/', help='path of data')
        self.parser.add_argument('--phase', type=str, default='train', help='phase for train or test')
        self.parser.add_argument('--img_dim', type=int, default=3, help='img_dim')
        self.parser.add_argument('--parse_dim', type=int, default=7, help='parse_dim')

        self.parser.add_argument('--batch_size', type=int, default=1, help='batch size')
        self.parser.add_argument('--resize_size', type=int, default=573, help='resized image size for training')
        self.parser.add_argument('--crop_size', type=int, default=512, help='cropped image size for training')
        self.parser.add_argument('--flip', type=bool, default=True, help='specified if  flipping')
        self.parser.add_argument('--nThreads', type=int, default=4, help='# of threads for data loader')

        # log related
        self.parser.add_argument('--name', type=str, default='FPMT', help='folder name to save outputs')
        self.parser.add_argument('--img_dir', type=str, default='./imgs', help='path for saving result images')
        self.parser.add_argument('--model_dir', type=str, default='./weights',
                                 help='path for saving result models')

        # generator
        self.parser.add_argument('--num_high', type=int, default=3, help='num_high, L')

        self.parser.add_argument('--gpu', type=int, default=0, help='gpu: e.g. 0 ,use -1 for CPU')
        self.parser.add_argument('--init_type', type=str, default='normal',
                                 help='init_type [normal, xavier, kaiming, orthogonal]')

        self.parser.add_argument('--resume', type=str, default=None,
                                 help='specified the dir of saved models for resume the training')

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('\n--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt
