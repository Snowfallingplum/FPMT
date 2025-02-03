import os
import cv2
import random
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms


class MakeupDataset(data.Dataset):
    def __init__(self, opts):
        self.opts = opts
        self.phase = opts.phase
        self.data_root = opts.data_root
        self.resize_size = opts.resize_size
        self.crop_size = opts.crop_size
        self.flip = opts.flip
        self.parse_dim = opts.parse_dim

        non_makeup_name = os.listdir(os.path.join(self.data_root, 'non_makeup'))
        self.non_makeup_path = [os.path.join(self.data_root, 'non_makeup', x) for x in non_makeup_name]
        sorted(self.non_makeup_path, reverse=True)

        makeup_name = os.listdir(os.path.join(self.data_root, 'makeup'))
        self.makeup_path = [os.path.join(self.data_root, 'makeup', x) for x in makeup_name]

        self.non_makeup_size = len(self.non_makeup_path)
        self.makeup_size = len(self.makeup_path)
        if self.phase == 'train':
            self.data_size = self.non_makeup_size
        else:
            self.data_size = self.non_makeup_size * self.makeup_size

        self.color_transform = transforms.Compose(
            [transforms.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
             transforms.ToTensor()])
        self.type_transform = transforms.Compose(
            [transforms.ToTensor()])

        print('size', self.data_size)

    def load_img(self, path, angle=0):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.rotate(img, angle)
        return img

    def load_parse(self, path, angle=0):
        parse = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        parse = self.rotate(parse, angle)
        return parse

    def rotate(self, img, angle):
        img = Image.fromarray(img)
        img = img.rotate(angle)
        img = np.array(img)
        return img

    def preprocessing(self, img, parse):
        img = cv2.resize(img, (self.resize_size, self.resize_size))
        parse = cv2.resize(parse, (self.resize_size, self.resize_size), interpolation=cv2.INTER_NEAREST)
        if random.random() > 0.5:
            h1 = int(np.ceil(np.random.uniform(1e-2, self.resize_size - self.crop_size)))
            w1 = int(np.ceil(np.random.uniform(1e-2, self.resize_size - self.crop_size)))
            img = img[h1:h1 + self.crop_size, w1:w1 + self.crop_size]
            parse = parse[h1:h1 + self.crop_size, w1:w1 + self.crop_size]
        if self.flip:
            if random.random() > 0.5:
                img = np.fliplr(img)
                parse = np.fliplr(parse)
        img = cv2.resize(img, (self.crop_size, self.crop_size))
        parse = cv2.resize(parse, (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST)
        return img, parse
    def test_preprocessing(self, img, parse):
        img = cv2.resize(img, (self.resize_size, self.resize_size))
        parse = cv2.resize(parse, (self.resize_size, self.resize_size), interpolation=cv2.INTER_NEAREST)
        h1 = int(self.resize_size - self.crop_size)//2
        w1 = int(self.resize_size - self.crop_size)//2
        img = img[h1:h1 + self.crop_size, w1:w1 + self.crop_size]
        parse = parse[h1:h1 + self.crop_size, w1:w1 + self.crop_size]
        return img, parse
    def split_parse(self, parse):
        # 0背景；1面部；2左眉毛；3右眉毛；4左眼；5右眼；6眼镜
        # 7左耳朵；8右耳朵；9耳环；10鼻子；11口内部；12上嘴唇
        # 13下嘴唇；14脖子；15项链；16衣服；17头发；18帽子
        # ---------->
        # 0背景；1面部；2眉毛；3眼睛；4鼻子;5嘴唇；6脖子
        h, w = parse.shape
        result = np.zeros([h, w, self.parse_dim])
        # 与面部无关的都是背景
        result[:, :, 0][np.where(parse == 0)] = 1
        result[:, :, 0][np.where(parse == 4)] = 1
        result[:, :, 0][np.where(parse == 5)] = 1
        result[:, :, 0][np.where(parse == 9)] = 1
        result[:, :, 0][np.where(parse == 11)] = 1
        result[:, :, 0][np.where(parse == 15)] = 1
        result[:, :, 0][np.where(parse == 16)] = 1
        result[:, :, 0][np.where(parse == 17)] = 1
        result[:, :, 0][np.where(parse == 18)] = 1

        result[:, :, 1][np.where(parse == 1)] = 1
        result[:, :, 1][np.where(parse == 6)] = 1
        result[:, :, 1][np.where(parse == 7)] = 1
        result[:, :, 1][np.where(parse == 8)] = 1

        result[:, :, 2][np.where(parse == 2)] = 1
        result[:, :, 2][np.where(parse == 3)] = 1

        result[:, :, 3][np.where(parse == 4)] = 1
        result[:, :, 3][np.where(parse == 5)] = 1

        result[:, :, 4][np.where(parse == 10)] = 1

        result[:, :, 5][np.where(parse == 12)] = 1
        result[:, :, 5][np.where(parse == 13)] = 1

        result[:, :, 6][np.where(parse == 14)] = 1
        result = np.array(result)
        return result

    def local_masks(self, split_parse):
        h, w, c = split_parse.shape
        all_mask = np.zeros([h, w])
        all_mask[np.where(split_parse[:, :, 0] == 0)] = 1

        brow_mask = np.zeros([h, w])
        brow_mask[np.where(split_parse[:, :, 2] == 1)] = 1

        eye_mask = np.zeros([h, w])
        eye_mask[np.where(split_parse[:, :, 3] == 1)] = 1
        kernel = np.ones((30, 30), np.uint8)
        if self.crop_size>=512:
            kernel = np.ones((60, 60), np.uint8)
        if self.crop_size>=1024:
            kernel = np.ones((120, 120), np.uint8)
        eye_maskall = cv2.dilate(eye_mask, kernel, iterations=1)
        eye_mask = eye_maskall - eye_mask
        eye_mask[np.where(all_mask == 0)] = 0
        eye_mask[np.where(brow_mask == 1)] = 0

        lip_mask = np.zeros([h, w])
        lip_mask[np.where(split_parse[:, :, 5] == 1)] = 1

        # neck_mask = np.zeros([h, w])
        # neck_mask[np.where(split_parse[:, :, 6] == 1)] = 1

        face_mask = np.zeros([h, w])
        face_mask[np.where(all_mask == 1)] = 1
        face_mask[np.where(brow_mask == 1)] = 0
        face_mask[np.where(eye_mask == 1)] = 0
        face_mask[np.where(lip_mask == 1)] = 0
        # face_mask[np.where(neck_mask == 1)] = 0

        all_mask = np.expand_dims(all_mask, axis=2)  # Expansion of the dimension
        face_mask = np.expand_dims(face_mask, axis=2)
        brow_mask = np.expand_dims(brow_mask, axis=2)
        eye_mask = np.expand_dims(eye_mask, axis=2)
        lip_mask = np.expand_dims(lip_mask, axis=2)
        # neck_mask = np.expand_dims(neck_mask, axis=2)

        all_mask = np.concatenate((all_mask, all_mask, all_mask), axis=2)
        face_mask = np.concatenate((face_mask, face_mask, face_mask), axis=2)
        brow_mask = np.concatenate((brow_mask, brow_mask, brow_mask), axis=2)
        eye_mask = np.concatenate((eye_mask, eye_mask, eye_mask), axis=2)
        lip_mask = np.concatenate((lip_mask, lip_mask, lip_mask), axis=2)
        # neck_mask = np.concatenate((neck_mask, neck_mask, neck_mask), axis=2)

        return all_mask, face_mask, brow_mask, eye_mask, lip_mask

    def affine_transform(self, x, theta):
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, mode='nearest', align_corners=True)
        return x

    def self_supervised_processing(self, color_img, parse):
        color_img = Image.fromarray(color_img.astype('uint8'))
        color_img_change = self.color_transform(color_img).unsqueeze(0)

        parse = Image.fromarray((parse).astype('uint8'))
        parse = self.type_transform(parse).unsqueeze(0)

        # generate reference image
        theta1 = np.zeros(9)
        theta1[0:6] = np.random.randn(6) * 0.15
        theta1 = theta1 + np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
        affine1 = np.reshape(theta1, (3, 3))
        affine1 = np.reshape(affine1, -1)[0:6]
        affine1 = torch.from_numpy(affine1).type(torch.FloatTensor)
        color_img_change_warp = self.affine_transform(color_img_change, affine1)

        color_img_change = np.array(color_img_change.squeeze(0).data) * 255.
        color_img_change = np.transpose(color_img_change, (1, 2, 0))
        color_img_change_warp = np.array(color_img_change_warp.squeeze(0).data) * 255.
        color_img_change_warp = np.transpose(color_img_change_warp, (1, 2, 0))

        parse_warp = self.affine_transform(parse, affine1)
        parse_warp = np.array(parse_warp.squeeze(0).data) * 255.
        parse_warp = np.transpose(parse_warp, (1, 2, 0))
        parse_warp = parse_warp.astype('uint8')  # 变换到[0,1]
        return color_img_change, color_img_change_warp, np.round(parse_warp)

    def __getitem__(self, index):
        if self.phase == 'train':
            non_makeup_index = index
            makeup_index = random.randint(0, self.makeup_size - 1)

            if random.random() > 1:
                non_makeup_angle = random.randint(0, 60) - 30
                makeup_angle = random.randint(0, 60) - 30
            else:
                non_makeup_angle = 0
                makeup_angle = 0

            non_makeup_img = self.load_img(self.non_makeup_path[non_makeup_index], non_makeup_angle)
            makeup_img = self.load_img(self.makeup_path[makeup_index], makeup_angle)

            non_makeup_parse = self.load_parse(
                self.non_makeup_path[non_makeup_index].replace('images', 'seg1')[:-4] + '.png',non_makeup_angle)
            makeup_parse = self.load_parse(self.makeup_path[makeup_index].replace('images', 'seg1')[:-4] + '.png',makeup_angle)

            non_makeup_img, non_makeup_parse = self.preprocessing(non_makeup_img, non_makeup_parse)
            makeup_img, makeup_parse = self.preprocessing(makeup_img, makeup_parse)

            makeup_color_change_img, makeup_color_change_warp_img, makeup_warp_parse = self.self_supervised_processing(
                makeup_img, makeup_parse)
            makeup_warp_parse = np.squeeze(makeup_warp_parse, axis=-1)

            if self.flip:
                if random.random() > 0.5:
                    makeup_color_change_warp_img = np.fliplr(makeup_color_change_warp_img)
                    makeup_warp_parse = np.fliplr(makeup_warp_parse)

            non_makeup_split_parse = self.split_parse(non_makeup_parse)
            makeup_split_parse = self.split_parse(makeup_parse)
            makeup_split_parse_warp = self.split_parse(makeup_warp_parse)

            non_makeup_all_mask, non_makeup_face_mask, non_makeup_brow_mask, non_makeup_eye_mask, non_makeup_lip_mask = self.local_masks(
                non_makeup_split_parse)
            makeup_all_mask, makeup_face_mask, makeup_brow_mask, makeup_eye_mask, makeup_lip_mask = self.local_masks(
                makeup_split_parse)

            makeup_all_warp_mask, makeup_face_warp_mask, makeup_brow_warp_mask, makeup_eye_warp_mask, makeup_lip_warp_mask = self.local_masks(
                makeup_split_parse_warp)

            non_makeup_split_parse[:, :, 3]=0
            non_makeup_split_parse[:, :, 3]=non_makeup_eye_mask[:,:,0]
            non_makeup_split_parse[:, :, 1]=non_makeup_split_parse[:, :, 1]-non_makeup_eye_mask[:,:,0]

            makeup_split_parse[:, :, 3] = 0
            makeup_split_parse[:, :, 3] = makeup_eye_mask[:, :, 0]
            makeup_split_parse[:, :, 1] = makeup_split_parse[:, :, 1] - makeup_eye_mask[:, :, 0]

            makeup_split_parse_warp[:, :, 3] = 0
            makeup_split_parse_warp[:, :, 3] = makeup_eye_warp_mask[:, :, 0]
            makeup_split_parse_warp[:, :, 1] = makeup_split_parse_warp[:, :, 1] - makeup_eye_warp_mask[:, :, 0]

            
            # 各种归一化
            makeup_change_img_gt = makeup_img * (1 - makeup_all_mask) + makeup_color_change_img * makeup_all_mask

            makeup_change_img_gt = makeup_change_img_gt / 127.5 - 1
            makeup_change_img_gt = np.transpose(makeup_change_img_gt, (2, 0, 1))

            # makeup_color_change_img = makeup_color_change_img / 127.5 - 1
            # makeup_color_change_img = np.transpose(makeup_color_change_img, (2, 0, 1))

            makeup_color_change_warp_img = makeup_color_change_warp_img / 127.5 - 1
            makeup_color_change_warp_img = np.transpose(makeup_color_change_warp_img, (2, 0, 1))

            makeup_split_parse_warp = np.transpose(makeup_split_parse_warp, (2, 0, 1))
            makeup_all_warp_mask = np.transpose(makeup_all_warp_mask, (2, 0, 1))
            makeup_face_warp_mask = np.transpose(makeup_face_warp_mask, (2, 0, 1))
            makeup_brow_warp_mask = np.transpose(makeup_brow_warp_mask, (2, 0, 1))
            makeup_eye_warp_mask = np.transpose(makeup_eye_warp_mask, (2, 0, 1))
            makeup_lip_warp_mask = np.transpose(makeup_lip_warp_mask, (2, 0, 1))

            non_makeup_img = non_makeup_img / 127.5 - 1
            non_makeup_img = np.transpose(non_makeup_img, (2, 0, 1))

            non_makeup_split_parse = np.transpose(non_makeup_split_parse, (2, 0, 1))
            non_makeup_all_mask = np.transpose(non_makeup_all_mask, (2, 0, 1))
            non_makeup_face_mask = np.transpose(non_makeup_face_mask, (2, 0, 1))
            non_makeup_brow_mask = np.transpose(non_makeup_brow_mask, (2, 0, 1))
            non_makeup_eye_mask = np.transpose(non_makeup_eye_mask, (2, 0, 1))
            non_makeup_lip_mask = np.transpose(non_makeup_lip_mask, (2, 0, 1))

            makeup_img = makeup_img / 127.5 - 1
            makeup_img = np.transpose(makeup_img, (2, 0, 1))

            makeup_split_parse = np.transpose(makeup_split_parse, (2, 0, 1))
            makeup_all_mask = np.transpose(makeup_all_mask, (2, 0, 1))
            makeup_face_mask = np.transpose(makeup_face_mask, (2, 0, 1))
            makeup_brow_mask = np.transpose(makeup_brow_mask, (2, 0, 1))
            makeup_eye_mask = np.transpose(makeup_eye_mask, (2, 0, 1))
            makeup_lip_mask = np.transpose(makeup_lip_mask, (2, 0, 1))

            data = {'non_makeup_img': torch.from_numpy(non_makeup_img).type(torch.FloatTensor),
                    'non_makeup_split_parse': torch.from_numpy(non_makeup_split_parse).type(torch.FloatTensor),
                    'non_makeup_all_mask': torch.from_numpy(non_makeup_all_mask).type(torch.FloatTensor),
                    'non_makeup_face_mask': torch.from_numpy(non_makeup_face_mask).type(torch.FloatTensor),
                    'non_makeup_brow_mask': torch.from_numpy(non_makeup_brow_mask).type(torch.FloatTensor),
                    'non_makeup_eye_mask': torch.from_numpy(non_makeup_eye_mask).type(torch.FloatTensor),
                    'non_makeup_lip_mask': torch.from_numpy(non_makeup_lip_mask).type(torch.FloatTensor),

                    'makeup_img': torch.from_numpy(makeup_img).type(torch.FloatTensor),
                    'makeup_split_parse': torch.from_numpy(makeup_split_parse).type(torch.FloatTensor),
                    'makeup_all_mask': torch.from_numpy(makeup_all_mask).type(torch.FloatTensor),
                    'makeup_face_mask': torch.from_numpy(makeup_face_mask).type(torch.FloatTensor),
                    'makeup_brow_mask': torch.from_numpy(makeup_brow_mask).type(torch.FloatTensor),
                    'makeup_eye_mask': torch.from_numpy(makeup_eye_mask).type(torch.FloatTensor),
                    'makeup_lip_mask': torch.from_numpy(makeup_lip_mask).type(torch.FloatTensor),

                    'makeup_change_img': torch.from_numpy(makeup_color_change_warp_img).type(torch.FloatTensor),
                    'makeup_change_img_gt': torch.from_numpy(makeup_change_img_gt).type(torch.FloatTensor),
                    'makeup_split_parse_warp': torch.from_numpy(makeup_split_parse_warp).type(torch.FloatTensor),
                    'makeup_all_warp_mask': torch.from_numpy(makeup_all_warp_mask).type(torch.FloatTensor),
                    'makeup_face_warp_mask': torch.from_numpy(makeup_face_warp_mask).type(torch.FloatTensor),
                    'makeup_brow_warp_mask': torch.from_numpy(makeup_brow_warp_mask).type(torch.FloatTensor),
                    'makeup_eye_warp_mask': torch.from_numpy(makeup_eye_warp_mask).type(torch.FloatTensor),
                    'makeup_lip_warp_mask': torch.from_numpy(makeup_lip_warp_mask).type(torch.FloatTensor)
                    }
            return data
        if self.phase == 'test_pair':
            # non_makeup_index = index % self.non_makeup_size
            # makeup_index = index // self.non_makeup_size

            non_makeup_index = index // self.makeup_size
            makeup_index = index % self.makeup_size
            print(self.non_makeup_size,self.makeup_size,non_makeup_index,makeup_index)
            non_makeup_angle = 0
            makeup_angle = 0
            # if random.random() > 1:
            #     non_makeup_angle = random.randint(0, 60) - 30
            #     makeup_angle = random.randint(0, 60) - 30
            # else:
            #     non_makeup_angle = 0
            #     makeup_angle = 0

            non_makeup_img = self.load_img(self.non_makeup_path[non_makeup_index], non_makeup_angle)
            makeup_img = self.load_img(self.makeup_path[makeup_index], makeup_angle)

            non_makeup_parse = self.load_parse(
                self.non_makeup_path[non_makeup_index].replace('images', 'seg1')[:-4] + '.png')
            makeup_parse = self.load_parse(self.makeup_path[makeup_index].replace('images', 'seg1')[:-4] + '.png')

            # non_makeup_parse = self.load_parse(
            #     self.non_makeup_path[non_makeup_index].replace('non_makeup', 'non_makeup_parse'),
            #     non_makeup_angle)
            # makeup_parse = self.load_parse(self.makeup_path[makeup_index].replace('makeup', 'makeup_parse'),
            #                                makeup_angle)

            non_makeup_img, non_makeup_parse = self.test_preprocessing(non_makeup_img, non_makeup_parse)
            makeup_img, makeup_parse = self.test_preprocessing(makeup_img, makeup_parse)

            # makeup_color_change_img, makeup_color_change_warp_img, makeup_warp_parse = self.self_supervised_processing(
            #     makeup_img, makeup_parse)
            # makeup_warp_parse = np.squeeze(makeup_warp_parse, axis=-1)
            #
            # if self.flip:
            #     if random.random() > 0.5:
            #         makeup_color_change_warp_img = np.fliplr(makeup_color_change_warp_img)
            #         makeup_warp_parse = np.fliplr(makeup_warp_parse)

            non_makeup_split_parse = self.split_parse(non_makeup_parse)
            makeup_split_parse = self.split_parse(makeup_parse)
            # makeup_split_parse_warp = self.split_parse(makeup_warp_parse)

            non_makeup_all_mask, non_makeup_face_mask, non_makeup_brow_mask, non_makeup_eye_mask, non_makeup_lip_mask = self.local_masks(
                non_makeup_split_parse)
            makeup_all_mask, makeup_face_mask, makeup_brow_mask, makeup_eye_mask, makeup_lip_mask = self.local_masks(
                makeup_split_parse)

            # makeup_all_warp_mask, makeup_face_warp_mask, makeup_brow_warp_mask, makeup_eye_warp_mask, makeup_lip_warp_mask = self.local_masks(
            #     makeup_split_parse_warp)

            non_makeup_split_parse[:, :, 3] = 0
            non_makeup_split_parse[:, :, 3] = non_makeup_eye_mask[:, :, 0]
            non_makeup_split_parse[:, :, 1] = non_makeup_split_parse[:, :, 1] - non_makeup_eye_mask[:, :, 0]

            makeup_split_parse[:, :, 3] = 0
            makeup_split_parse[:, :, 3] = makeup_eye_mask[:, :, 0]
            makeup_split_parse[:, :, 1] = makeup_split_parse[:, :, 1] - makeup_eye_mask[:, :, 0]

            # print(np.unique(makeup_split_parse))
            # print(np.unique(non_makeup_split_parse))


            # makeup_split_parse_warp[:, :, 3] = 0
            # makeup_split_parse_warp[:, :, 3] = makeup_eye_warp_mask[:, :, 0]
            # makeup_split_parse_warp[:, :, 1] = makeup_split_parse_warp[:, :, 1] - makeup_eye_warp_mask[:, :, 0]

            # # 各种归一化
            # makeup_change_img_gt = makeup_img * (1 - makeup_all_mask) + makeup_color_change_img * makeup_all_mask
            #
            # makeup_change_img_gt = makeup_change_img_gt / 127.5 - 1
            # makeup_change_img_gt = np.transpose(makeup_change_img_gt, (2, 0, 1))

            # makeup_color_change_img = makeup_color_change_img / 127.5 - 1
            # makeup_color_change_img = np.transpose(makeup_color_change_img, (2, 0, 1))

            # makeup_color_change_warp_img = makeup_color_change_warp_img / 127.5 - 1
            # makeup_color_change_warp_img = np.transpose(makeup_color_change_warp_img, (2, 0, 1))
            #
            # makeup_split_parse_warp = np.transpose(makeup_split_parse_warp, (2, 0, 1))
            # makeup_all_warp_mask = np.transpose(makeup_all_warp_mask, (2, 0, 1))
            # makeup_face_warp_mask = np.transpose(makeup_face_warp_mask, (2, 0, 1))
            # makeup_brow_warp_mask = np.transpose(makeup_brow_warp_mask, (2, 0, 1))
            # makeup_eye_warp_mask = np.transpose(makeup_eye_warp_mask, (2, 0, 1))
            # makeup_lip_warp_mask = np.transpose(makeup_lip_warp_mask, (2, 0, 1))

            non_makeup_img = non_makeup_img / 127.5 - 1
            non_makeup_img = np.transpose(non_makeup_img, (2, 0, 1))

            non_makeup_split_parse = np.transpose(non_makeup_split_parse, (2, 0, 1))
            non_makeup_all_mask = np.transpose(non_makeup_all_mask, (2, 0, 1))
            non_makeup_face_mask = np.transpose(non_makeup_face_mask, (2, 0, 1))
            non_makeup_brow_mask = np.transpose(non_makeup_brow_mask, (2, 0, 1))
            non_makeup_eye_mask = np.transpose(non_makeup_eye_mask, (2, 0, 1))
            non_makeup_lip_mask = np.transpose(non_makeup_lip_mask, (2, 0, 1))

            makeup_img = makeup_img / 127.5 - 1
            makeup_img = np.transpose(makeup_img, (2, 0, 1))

            makeup_split_parse = np.transpose(makeup_split_parse, (2, 0, 1))
            makeup_all_mask = np.transpose(makeup_all_mask, (2, 0, 1))
            makeup_face_mask = np.transpose(makeup_face_mask, (2, 0, 1))
            makeup_brow_mask = np.transpose(makeup_brow_mask, (2, 0, 1))
            makeup_eye_mask = np.transpose(makeup_eye_mask, (2, 0, 1))
            makeup_lip_mask = np.transpose(makeup_lip_mask, (2, 0, 1))

            data = {'non_makeup_img': torch.from_numpy(non_makeup_img).type(torch.FloatTensor),
                    'non_makeup_split_parse': torch.from_numpy(non_makeup_split_parse).type(torch.FloatTensor),
                    'non_makeup_all_mask': torch.from_numpy(non_makeup_all_mask).type(torch.FloatTensor),
                    'non_makeup_face_mask': torch.from_numpy(non_makeup_face_mask).type(torch.FloatTensor),
                    'non_makeup_brow_mask': torch.from_numpy(non_makeup_brow_mask).type(torch.FloatTensor),
                    'non_makeup_eye_mask': torch.from_numpy(non_makeup_eye_mask).type(torch.FloatTensor),
                    'non_makeup_lip_mask': torch.from_numpy(non_makeup_lip_mask).type(torch.FloatTensor),

                    'makeup_img': torch.from_numpy(makeup_img).type(torch.FloatTensor),
                    'makeup_split_parse': torch.from_numpy(makeup_split_parse).type(torch.FloatTensor),
                    'makeup_all_mask': torch.from_numpy(makeup_all_mask).type(torch.FloatTensor),
                    'makeup_face_mask': torch.from_numpy(makeup_face_mask).type(torch.FloatTensor),
                    'makeup_brow_mask': torch.from_numpy(makeup_brow_mask).type(torch.FloatTensor),
                    'makeup_eye_mask': torch.from_numpy(makeup_eye_mask).type(torch.FloatTensor),
                    'makeup_lip_mask': torch.from_numpy(makeup_lip_mask).type(torch.FloatTensor)
                    }
            return data
        


    def __len__(self):
        return self.data_size


if __name__ == '__main__':
    def save_imgs(imgs, names, path):
        if not os.path.exists(path):
            os.mkdir(path)
        for img, name in zip(imgs, names):
            img = tensor2img(img)
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(path, name + '.jpg'), img)


    def tensor2img(img):
        img = img[0].cpu().float().numpy()
        if img.shape[0] == 1:
            img = np.tile(img, (3, 1, 1))
        img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0
        return img.astype(np.uint8)


    from options import Options

    parser = Options()
    opts = parser.parse()

    # data loader
    print('\n--- load dataset ---')
    dataset = MakeupDataset(opts)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=True,
                                               num_workers=opts.nThreads)

    data_names = ['non_makeup_img', 'non_makeup_all_mask', 'non_makeup_face_mask', 'non_makeup_brow_mask',
                  'non_makeup_eye_mask', 'non_makeup_lip_mask',
                  'makeup_img', 'makeup_all_mask', 'makeup_face_mask', 'makeup_brow_mask', 'makeup_eye_mask',
                  'makeup_lip_mask',
                  'makeup_change_img', 'makeup_change_img_gt', 'makeup_all_warp_mask']

    print('dataset size:', len(train_loader))
    for i, data in enumerate(train_loader):
        if i > 2:
            break
        print(i)
        imgs = [data[x] for x in data_names]
        print(torch.sum(data['non_makeup_split_parse']))
        print(torch.sum(data['makeup_split_parse']))
        print(torch.sum(data['makeup_split_parse_warp']))
        names = [str(i) + x + '.jpg' for x in data_names]
        save_imgs(imgs, names, path='./dataset_load_test')
