import os
import torch
from saver import Saver
from options import Options
from model import FPMT
from dataset import MakeupDataset
import warnings

warnings.filterwarnings("ignore")


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
    print('Number of params: %.2fM' % (num_params / 1e6))


def eval_pair():
    # parse options
    parser = Options()
    opts = parser.parse()
    opts.phase = 'test_pair'

    opts.data_root = './examples/images'
    opts.img_dir = './examples_results'
    # data loader
    print('\n--- load dataset ---')
    dataset = MakeupDataset(opts)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # model
    print('\n--- load model ---')
    model = FPMT(opts)
    print_network(model.gen)
    ep0, total_it = model.resume(os.path.join(opts.model_dir, 'FPMT_L3_512.pth'))
    model.eval()
    print('start test pair')
    # saver for display and output
    saver = Saver(opts)
    for iter, data in enumerate(train_loader):
        with torch.no_grad():
            model.test_load_data(data)
            saver.write_test_pair_img(iter, model)


if __name__ == '__main__':
    eval_pair()
