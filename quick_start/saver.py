import os
import torchvision


class Saver():
    def __init__(self, opts):
        self.image_dir = opts.img_dir
        self.model_dir = opts.model_dir

    # save test pair images
    def write_test_pair_img(self, iter, model):
        root = os.path.join(self.image_dir, 'test_pair')
        if not os.path.exists(root):
            os.makedirs(root)
        test_pair_img = model.test_pair()
        img_filename = '%s/gen_%05d.jpg' % (root, iter)
        torchvision.utils.save_image(test_pair_img / 2 + 0.5, img_filename, nrow=1)
