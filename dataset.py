import os
import cv2
import torch
import numpy as np
from PIL import Image
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

    def test_preprocessing(self, img, parse):
        img = cv2.resize(img, (self.resize_size, self.resize_size))
        parse = cv2.resize(parse, (self.resize_size, self.resize_size), interpolation=cv2.INTER_NEAREST)
        h1 = int(self.resize_size - self.crop_size) // 2
        w1 = int(self.resize_size - self.crop_size) // 2
        img = img[h1:h1 + self.crop_size, w1:w1 + self.crop_size]
        parse = parse[h1:h1 + self.crop_size, w1:w1 + self.crop_size]
        return img, parse

    def split_parse(self, parse):
        h, w = parse.shape
        result = np.zeros([h, w, self.parse_dim])
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
        if self.crop_size >= 512:
            kernel = np.ones((60, 60), np.uint8)
        if self.crop_size >= 1024:
            kernel = np.ones((120, 120), np.uint8)
        eye_maskall = cv2.dilate(eye_mask, kernel, iterations=1)
        eye_mask = eye_maskall - eye_mask
        eye_mask[np.where(all_mask == 0)] = 0
        eye_mask[np.where(brow_mask == 1)] = 0

        lip_mask = np.zeros([h, w])
        lip_mask[np.where(split_parse[:, :, 5] == 1)] = 1

        face_mask = np.zeros([h, w])
        face_mask[np.where(all_mask == 1)] = 1
        face_mask[np.where(brow_mask == 1)] = 0
        face_mask[np.where(eye_mask == 1)] = 0
        face_mask[np.where(lip_mask == 1)] = 0

        all_mask = np.expand_dims(all_mask, axis=2)  # Expansion of the dimension
        face_mask = np.expand_dims(face_mask, axis=2)
        brow_mask = np.expand_dims(brow_mask, axis=2)
        eye_mask = np.expand_dims(eye_mask, axis=2)
        lip_mask = np.expand_dims(lip_mask, axis=2)

        all_mask = np.concatenate((all_mask, all_mask, all_mask), axis=2)
        face_mask = np.concatenate((face_mask, face_mask, face_mask), axis=2)
        brow_mask = np.concatenate((brow_mask, brow_mask, brow_mask), axis=2)
        eye_mask = np.concatenate((eye_mask, eye_mask, eye_mask), axis=2)
        lip_mask = np.concatenate((lip_mask, lip_mask, lip_mask), axis=2)

        return all_mask, face_mask, brow_mask, eye_mask, lip_mask

    def __getitem__(self, index):
        if self.phase == 'test_pair':
            non_makeup_index = index // self.makeup_size
            makeup_index = index % self.makeup_size
            print(
                f'The size of non_makeup, makeup images: {self.non_makeup_size},{self.makeup_size}, Processing:{non_makeup_index + 1},{makeup_index + 1}')
            non_makeup_angle = 0
            makeup_angle = 0

            non_makeup_img = self.load_img(self.non_makeup_path[non_makeup_index], non_makeup_angle)
            makeup_img = self.load_img(self.makeup_path[makeup_index], makeup_angle)

            non_makeup_parse = self.load_parse(
                self.non_makeup_path[non_makeup_index].replace('images', 'seg1')[:-4] + '.png')
            makeup_parse = self.load_parse(self.makeup_path[makeup_index].replace('images', 'seg1')[:-4] + '.png')

            non_makeup_img, non_makeup_parse = self.test_preprocessing(non_makeup_img, non_makeup_parse)
            makeup_img, makeup_parse = self.test_preprocessing(makeup_img, makeup_parse)

            non_makeup_split_parse = self.split_parse(non_makeup_parse)
            makeup_split_parse = self.split_parse(makeup_parse)

            non_makeup_all_mask, non_makeup_face_mask, non_makeup_brow_mask, non_makeup_eye_mask, non_makeup_lip_mask = self.local_masks(
                non_makeup_split_parse)
            makeup_all_mask, makeup_face_mask, makeup_brow_mask, makeup_eye_mask, makeup_lip_mask = self.local_masks(
                makeup_split_parse)

            non_makeup_split_parse[:, :, 3] = 0
            non_makeup_split_parse[:, :, 3] = non_makeup_eye_mask[:, :, 0]
            non_makeup_split_parse[:, :, 1] = non_makeup_split_parse[:, :, 1] - non_makeup_eye_mask[:, :, 0]

            makeup_split_parse[:, :, 3] = 0
            makeup_split_parse[:, :, 3] = makeup_eye_mask[:, :, 0]
            makeup_split_parse[:, :, 1] = makeup_split_parse[:, :, 1] - makeup_eye_mask[:, :, 0]

            non_makeup_img = non_makeup_img / 127.5 - 1
            non_makeup_img = np.transpose(non_makeup_img, (2, 0, 1))

            non_makeup_split_parse = np.transpose(non_makeup_split_parse, (2, 0, 1))

            makeup_img = makeup_img / 127.5 - 1
            makeup_img = np.transpose(makeup_img, (2, 0, 1))

            makeup_split_parse = np.transpose(makeup_split_parse, (2, 0, 1))

            data = {'non_makeup_img': torch.from_numpy(non_makeup_img).type(torch.FloatTensor),
                    'non_makeup_split_parse': torch.from_numpy(non_makeup_split_parse).type(torch.FloatTensor),

                    'makeup_img': torch.from_numpy(makeup_img).type(torch.FloatTensor),
                    'makeup_split_parse': torch.from_numpy(makeup_split_parse).type(torch.FloatTensor),

                    }
            return data

    def __len__(self):
        return self.data_size
