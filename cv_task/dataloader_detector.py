import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import os
import time
import collections
import random
from scipy.ndimage import zoom
import warnings
from scipy.ndimage.interpolation import rotate
from .utils.utils import *
from .utils.datasets import *
from .utils.parse_config import *

import pandas as pd
import json
import matplotlib.pyplot as plt


def list2int(A):
    return [int(x) for x in A]


class Crop(object):
    def __init__(self):
        # self.crop_size = config['crop_size']
        # self.bound_size = config['bound_size']
        # self.stride = config['stride']
        # self.pad_value = config['pad_value']

        self.crop_size = [128, 128, 128]
        self.bound_size = 12
        self.stride = 4
        self.pad_value = 170

    def __call__(self, imgs, target, bboxes, isScale=False, isRand=False):
        if isScale:
            radiusLim = [8., 120.]
            scaleLim = [0.75, 1.25]
            scaleRange = [np.min([np.max([(radiusLim[0] / target[3]), scaleLim[0]]), 1])
                , np.max([np.min([(radiusLim[1] / target[3]), scaleLim[1]]), 1])]
            scale = np.random.rand() * (scaleRange[1] - scaleRange[0]) + scaleRange[0]
            crop_size = (np.array(self.crop_size).astype('float') / scale).astype('int')
        else:
            crop_size = self.crop_size
        bound_size = self.bound_size
        target = np.copy(target)
        bboxes = np.copy(bboxes)

        start = []
        for i in range(3):
            if not isRand:
                r = target[3] / 2
                s = np.floor(target[i] - r) + 1 - bound_size
                e = np.ceil(target[i] + r) + 1 + bound_size - crop_size[i]
            else:
                s = np.max([imgs.shape[i + 1] - crop_size[i] / 2, imgs.shape[i + 1] / 2 + bound_size])
                e = np.min([crop_size[i] / 2, imgs.shape[i + 1] / 2 - bound_size])
                target = np.array([np.nan, np.nan, np.nan, np.nan])
            if s > e:
                start.append(np.random.randint(e, s))  # !
            else:
                start.append(int(target[i]) - crop_size[i] / 2 + np.random.randint(-bound_size / 2, bound_size / 2))

        normstart = np.array(start).astype('float32') / np.array(imgs.shape[1:]) - 0.5
        normsize = np.array(crop_size).astype('float32') / np.array(imgs.shape[1:])
        xx, yy, zz = np.meshgrid(np.linspace(normstart[0], normstart[0] + normsize[0], self.crop_size[0] / self.stride),
                                 np.linspace(normstart[1], normstart[1] + normsize[1], self.crop_size[1] / self.stride),
                                 np.linspace(normstart[2], normstart[2] + normsize[2], self.crop_size[2] / self.stride),
                                 indexing='ij')
        coord = np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, :]], 0).astype('float32')

        pad = []
        pad.append([0, 0])
        for i in range(3):
            leftpad = max(0, -start[i])
            rightpad = max(0, start[i] + crop_size[i] - imgs.shape[i + 1])
            pad.append([leftpad, rightpad])
        crop = imgs[:,
               max(start[0], 0):min(start[0] + crop_size[0], imgs.shape[1]),
               max(start[1], 0):min(start[1] + crop_size[1], imgs.shape[2]),
               max(start[2], 0):min(start[2] + crop_size[2], imgs.shape[3])]
        crop = np.pad(crop, pad, 'constant', constant_values=self.pad_value)
        for i in range(3):
            target[i] = target[i] - start[i]
        for i in range(len(bboxes)):
            for j in range(3):
                bboxes[i][j] = bboxes[i][j] - start[j]

        if isScale:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                crop = zoom(crop, [1, scale, scale, scale], order=1)
            newpad = self.crop_size[0] - crop.shape[1:][0]
            if newpad < 0:
                crop = crop[:, :-newpad, :-newpad, :-newpad]
            elif newpad > 0:
                pad2 = [[0, 0], [0, newpad], [0, newpad], [0, newpad]]
                crop = np.pad(crop, pad2, 'constant', constant_values=self.pad_value)
            for i in range(4):
                target[i] = target[i] * scale
            for i in range(len(bboxes)):
                for j in range(4):
                    bboxes[i][j] = bboxes[i][j] * scale

        return crop, target, bboxes, coord


def augment(sample, target, bboxes, coord, ifflip=True, ifrotate=True, ifswap=True):
    #                     angle1 = np.random.rand()*180
    if ifrotate:
        validrot = False
        counter = 0
        while not validrot:
            newtarget = np.copy(target)
            angle1 = np.random.rand() * 180
            size = np.array(sample.shape[2:4]).astype('float')
            rotmat = np.array([[np.cos(angle1 / 180 * np.pi), -np.sin(angle1 / 180 * np.pi)],
                               [np.sin(angle1 / 180 * np.pi), np.cos(angle1 / 180 * np.pi)]])
            newtarget[1:3] = np.dot(rotmat, target[1:3] - size / 2) + size / 2
            if np.all(newtarget[:3] > target[3]) and np.all(newtarget[:3] < np.array(sample.shape[1:4]) - newtarget[3]):
                validrot = True
                target = newtarget
                sample = rotate(sample, angle1, axes=(2, 3), reshape=False)
                coord = rotate(coord, angle1, axes=(2, 3), reshape=False)
                for box in bboxes:
                    box[1:3] = np.dot(rotmat, box[1:3] - size / 2) + size / 2
            else:
                counter += 1
                if counter == 3:
                    break
    if ifswap:
        if sample.shape[1] == sample.shape[2] and sample.shape[1] == sample.shape[3]:
            axisorder = np.random.permutation(3)
            sample = np.transpose(sample, np.concatenate([[0], axisorder + 1]))
            coord = np.transpose(coord, np.concatenate([[0], axisorder + 1]))
            target[:3] = target[:3][axisorder]
            bboxes[:, :3] = bboxes[:, :3][:, axisorder]

    if ifflip:
        #         flipid = np.array([np.random.randint(2),np.random.randint(2),np.random.randint(2)])*2-1
        flipid = np.array([1, np.random.randint(2), np.random.randint(2)]) * 2 - 1
        sample = np.ascontiguousarray(sample[:, ::flipid[0], ::flipid[1], ::flipid[2]])
        coord = np.ascontiguousarray(coord[:, ::flipid[0], ::flipid[1], ::flipid[2]])
        for ax in range(3):
            if flipid[ax] == -1:
                target[ax] = np.array(sample.shape[ax + 1]) - target[ax]
                bboxes[:, ax] = np.array(sample.shape[ax + 1]) - bboxes[:, ax]
    return sample, target, bboxes, coord


class LabelMapping(object):
    def __init__(self, config, phase):
        self.stride = np.array(config['stride'])
        self.num_neg = int(config['num_neg'])
        self.th_neg = config['th_neg']
        self.anchors = np.asarray(config['anchors'])
        self.phase = phase
        if phase == 'train':
            self.th_pos = config['th_pos_train']
        elif phase == 'validation':
            self.th_pos = config['th_pos_val']

    def __call__(self, input_size, target, bboxes):
        stride = self.stride
        num_neg = self.num_neg
        th_neg = self.th_neg
        anchors = self.anchors
        th_pos = self.th_pos

        output_size = []
        for i in range(3):
            assert (input_size[i] % stride == 0)
            output_size.append(input_size[i] / stride)

        # 最后feature 13x13x13，然后还得有9个anchors，每个anchor还有[classfication 0/1, 和四个坐标]
        # 最后是 [13,13,13,9,1+4=5]
        label = -1 * np.ones(list2int(output_size + [len(anchors), 5]), np.float32)

        # 这里跟meshgrid一个意思，zhw三个方向的坐标，不知为何这么定义，xyz多好
        offset = ((stride.astype('float')) - 1) / 2
        oz = np.arange(offset, offset + stride * (output_size[0] - 1) + 1, stride)
        oh = np.arange(offset, offset + stride * (output_size[1] - 1) + 1, stride)
        ow = np.arange(offset, offset + stride * (output_size[2] - 1) + 1, stride)

        # 这里很奇怪，每个bbox，从全局选择合适的anchor，打标签成0，坐标为何不赋值？
        for bbox in bboxes:
            for i, anchor in enumerate(anchors):
                # print('xxx',bbox)
                iz, ih, iw = select_samples(bbox, anchor, th_neg, oz, oh, ow)
                label[iz, ih, iw, i, 0] = 0

        # 这里更奇怪，training的话，所有都赋值成0（把前面覆盖了？），随机选一些label赋值成-1
        if self.phase == 'train' and self.num_neg > 0:
            neg_z, neg_h, neg_w, neg_a = np.where(label[:, :, :, :, 0] == -1)
            neg_idcs = random.sample(range(len(neg_z)), min(num_neg, len(neg_z)))
            neg_z, neg_h, neg_w, neg_a = neg_z[neg_idcs], neg_h[neg_idcs], neg_w[neg_idcs], neg_a[neg_idcs]
            label[:, :, :, :, 0] = 0
            label[neg_z, neg_h, neg_w, neg_a, 0] = -1

        # target 跟bbox有什么区别？忘记了
        if np.isnan(target[0]):
            return label
        iz, ih, iw, ia = [], [], [], []
        for i, anchor in enumerate(anchors):
            # print(target)
            iiz, iih, iiw = select_samples(target, anchor, th_pos, oz, oh, ow)
            iz.append(iiz)
            ih.append(iih)
            iw.append(iiw)
            ia.append(i * np.ones((len(iiz),), np.int64))
        iz = np.concatenate(iz, 0)
        ih = np.concatenate(ih, 0)
        iw = np.concatenate(iw, 0)
        ia = np.concatenate(ia, 0)
        flag = True
        if len(iz) == 0:
            pos = []
            for i in range(3):
                pos.append(max(0, int(np.round((target[i] - offset) / stride))))
            idx = np.argmin(np.abs(np.log(target[3] / anchors)))
            pos.append(idx)
            flag = False
        else:
            # 每个target只random选取一个anchor
            idx = random.sample(range(len(iz)), 1)[0]
            pos = [iz[idx], ih[idx], iw[idx], ia[idx]]
        dz = (target[0] - oz[pos[0]]) / anchors[pos[3]]
        dh = (target[1] - oh[pos[1]]) / anchors[pos[3]]
        dw = (target[2] - ow[pos[2]]) / anchors[pos[3]]
        dd = np.log(target[3] / anchors[pos[3]])
        label[pos[0], pos[1], pos[2], pos[3], :] = [1, dz, dh, dw, dd]
        return label


def select_samples(bbox, anchor, th, oz, oh, ow):
    z, h, w, d = bbox
    if d == 0:
        # d==0 实际是当前图像中没有任何的bounding box，后续的判断新版本会报错 杜绝了/0的情况
        return np.zeros((0,), np.int64), np.zeros((0,), np.int64), np.zeros((0,), np.int64)
    max_overlap = min(d, anchor)
    min_overlap = np.power(max(d, anchor), 3) * th / max_overlap / max_overlap
    if min_overlap > max_overlap:
        return np.zeros((0,), np.int64), np.zeros((0,), np.int64), np.zeros((0,), np.int64)
    else:
        s = z - 0.5 * np.abs(d - anchor) - (max_overlap - min_overlap)
        e = z + 0.5 * np.abs(d - anchor) + (max_overlap - min_overlap)
        mz = np.logical_and(oz >= s, oz <= e)
        iz = np.where(mz)[0]

        s = h - 0.5 * np.abs(d - anchor) - (max_overlap - min_overlap)
        e = h + 0.5 * np.abs(d - anchor) + (max_overlap - min_overlap)
        mh = np.logical_and(oh >= s, oh <= e)
        ih = np.where(mh)[0]

        s = w - 0.5 * np.abs(d - anchor) - (max_overlap - min_overlap)
        e = w + 0.5 * np.abs(d - anchor) + (max_overlap - min_overlap)
        mw = np.logical_and(ow >= s, ow <= e)
        iw = np.where(mw)[0]

        if len(iz) == 0 or len(ih) == 0 or len(iw) == 0:
            return np.zeros((0,), np.int64), np.zeros((0,), np.int64), np.zeros((0,), np.int64)

        lz, lh, lw = len(iz), len(ih), len(iw)
        iz = iz.reshape((-1, 1, 1))
        ih = ih.reshape((1, -1, 1))
        iw = iw.reshape((1, 1, -1))
        iz = np.tile(iz, (1, lh, lw)).reshape((-1))
        ih = np.tile(ih, (lz, 1, lw)).reshape((-1))
        iw = np.tile(iw, (lz, lh, 1)).reshape((-1))
        centers = np.concatenate([
            oz[iz].reshape((-1, 1)),
            oh[ih].reshape((-1, 1)),
            ow[iw].reshape((-1, 1))], axis=1)

        r0 = anchor / 2
        s0 = centers - r0
        e0 = centers + r0

        r1 = d / 2
        s1 = bbox[:3] - r1
        s1 = s1.reshape((1, -1))
        e1 = bbox[:3] + r1
        e1 = e1.reshape((1, -1))

        overlap = np.maximum(0, np.minimum(e0, e1) - np.maximum(s0, s1))

        intersection = overlap[:, 0] * overlap[:, 1] * overlap[:, 2]
        union = anchor * anchor * anchor + d * d * d - intersection

        iou = intersection / union

        mask = iou >= th
        # if th > 0.4:
        #   if np.sum(mask) == 0:
        #      print(['iou not large', iou.max()])
        # else:
        #    print(['iou large', iou[mask]])
        iz = iz[mask]
        ih = ih[mask]
        iw = iw[mask]
        return iz, ih, iw


def collate(batch):
    if torch.is_tensor(batch[0]):
        return [b.unsqueeze(0) for b in batch]
    elif isinstance(batch[0], np.ndarray):
        return batch
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], collections.Iterable):
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]


class xgdataloader(Dataset):
    def __init__(self, dfBboxes, dfDatapath, config={}, phase='train', split_comber=None):
        assert (phase == 'train' or phase == 'validation' or phase == 'test')
        self.phase = phase
        self.isScale = True
        self.ratio_rand = 0.3  # 改成过0.01，效果还行，
        self.dfBboxes = dfBboxes
        self.dfDatapath = dfDatapath.set_index('filename')
        if self.phase == 'train':
            self.isScale = True
        elif self.phase == 'validation':
            self.isScale = False
        self.augtype = {'flip': True, 'swap': False, 'scale': True, 'rotate': False}
        self.crop = Crop()
        self.label_mapping = LabelMapping(config, phase=self.phase)
        self.config = config
        self.stride = config['stride']
        self.pad_value = config['pad_value']
        self.split_comber = split_comber

    def __getitem__(self, idx):
        if (self.phase is 'train') or (self.phase is 'validation'):
            isRandom = False
            if idx < len(self.dfBboxes):
                curBboxDetail = self.dfBboxes.iloc[idx]
                curFilename = curBboxDetail['filename']
                bbox = json.loads(curBboxDetail.bbox)
                # curFileIndex = curBboxDetail.fileIndex
                # curFullFilePath = self.dfDatapath.loc[curFilename].filePath
                curFullFilePath = curBboxDetail['npz_path']
                isScale = self.isScale and (self.phase == 'train')
            else:
                isRandom = True
                curBboxDetail = []
                bbox = []
                tmpIndex = np.random.randint(0, len(self.dfDatapath))
                curFilename = self.dfDatapath.index.values[tmpIndex]
                curFullFilePath = self.dfDatapath.loc[curFilename].filePath
                isScale = False and (self.phase == 'train')

            with np.load(curFullFilePath, allow_pickle=True) as tmpNpz:
                im = tmpNpz['im']
                bboxes = tmpNpz['labels']
                bboxes = np.asarray(bboxes, dtype=np.float32)

            # now we have [im, bbox, bboxes, and isRandom]

            sample, target, bboxes, coord = self.crop(im, bbox, bboxes, isScale=isScale, isRand=isRandom)

            if self.phase == 'train' and not isRandom:
                sample, target, bboxes, coord = augment(sample, target, bboxes, coord,
                                                        ifflip=self.augtype['flip'], ifrotate=self.augtype['rotate'],
                                                        ifswap=self.augtype['swap'])
            # Convert Bbox to anchor
            # 这里有的sample里面是完全没有bbox的，这种按说不用经理下面的函数，直接给出一个固定的label就行了
            label = self.label_mapping(sample.shape[1:], target, bboxes)
            sample = (sample.astype(np.float32) - 128) / 128

            # 返回的有 图像，标签（sx,sy,sz,anchor,1+4=5）就好了，coord干什么用的？
            return torch.from_numpy(sample), torch.from_numpy(label), coord
        else:  # phase=='test'
            curFilename = self.dfDatapath.index.values[idx]
            curFullFilePath = self.dfDatapath.loc[curFilename].filePath

            with np.load(curFullFilePath, allow_pickle=True) as tmpNpz:
                im = tmpNpz['im']
                bboxes = tmpNpz['labels']
                bboxes = np.asarray(bboxes, dtype=np.float32)

            name = self.dfDatapath.iloc[idx]['filePath']
            imgs = im
            nz, nh, nw = imgs.shape[1:]
            pz = int(np.ceil(float(nz) / self.stride)) * self.stride
            ph = int(np.ceil(float(nh) / self.stride)) * self.stride
            pw = int(np.ceil(float(nw) / self.stride)) * self.stride
            imgs = np.pad(imgs, [[0, 0], [0, pz - nz], [0, ph - nh], [0, pw - nw]], 'constant',
                          constant_values=self.pad_value)
            xx, yy, zz = np.meshgrid(np.linspace(-0.5, 0.5, imgs.shape[1] / self.stride),
                                     np.linspace(-0.5, 0.5, imgs.shape[2] / self.stride),
                                     np.linspace(-0.5, 0.5, imgs.shape[3] / self.stride), indexing='ij')
            coord = np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, :]], 0).astype('float32')
            imgs, nzhw = self.split_comber.split(imgs)

            # side_len = self.split_comber.side_len / self.stride
            # max_stride = self.split_comber.max_stride / self.stride
            # margin = self.split_comber.margin / self.stride

            coord2, nzhw2 = self.split_comber.split(coord,
                                                    side_len=int(self.split_comber.side_len / self.stride),
                                                    max_stride=int(self.split_comber.max_stride / self.stride),
                                                    margin=int(self.split_comber.margin / self.stride))
            assert np.all(nzhw == nzhw2)
            imgs = (imgs.astype(np.float32) - 128) / 128
            return torch.from_numpy(imgs), bboxes, coord2, nzhw, name, im

    def __len__(self):
        if self.phase == 'train':
            return int(len(self.dfBboxes) / (1 - self.ratio_rand))
        elif self.phase == 'validation':
            return len(self.dfBboxes)
        elif self.phase == 'test':
            return len(self.dfDatapath)
        else:
            return 0


def get_trainloader(phase, config, config_default, split_comber=None):

    work_path = os.getcwd()
    dfBboxes = pd.read_csv(work_path+"/../cv_task/csv_files/new_luna_bboxlist.csv")
    dfDatapath = pd.read_csv(work_path+"/../cv_task/csv_files/new_luna_datapath.csv")

    if phase=='train':
        if config_default.training_subset == 100:
            dfBboxes = dfBboxes[dfBboxes.from_subset != config_default.validation_subset]
        else:
            dfBboxes = dfBboxes[dfBboxes.from_subset == config_default.training_subset]
    elif phase=='validation':
        dfBboxes = dfBboxes[dfBboxes.from_subset == config_default.validation_subset]

    xgloader = xgdataloader(dfBboxes, dfDatapath, config, phase, split_comber)
    print('======', phase, "dataset constructed, len:", xgloader.__len__())

    return xgloader


def get_dataset(phase):
    wd = os.getcwd()
    data_config = parse_data_config(os.path.join(wd, "../cv_task/config/custom.data"))
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    if phase == 'train':
        dataset = ListDataset(train_path, augment=True, multiscale=True)
    elif phase == 'valid':
        dataset = ListDataset(valid_path, augment=False, multiscale=False)

    return dataset, class_names



class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data


if __name__ == 'main':
    # if True:

    dfBboxes = pd.read_csv('./csv_files/luna_bboxlist.csv')
    dfDatapath = pd.read_csv('./csv_files/luna_datapath.csv')

    from split_combine import SplitComb
    import net
    from torch.utils.data import DataLoader

    phase = 'test'
    # 模型在这里
    config, model, loss, get_pbb = net.get_model()
    margin = 32
    sidelen = 144  # 96
    split_comber = SplitComb(sidelen, config['max_stride'], config['stride'], margin, config['pad_value'])

    dataset_test = xgdataloader(dfBboxes, dfDatapath, config, phase, split_comber)

    # data, target, coord, nzhw, name = next(iter(dataset_test))

    testloader = DataLoader(dataset_test,
                            batch_size=1,  # 必须是1，每张单独做
                            shuffle=False,
                            pin_memory=True,
                            num_workers=0)
    for i, (data, target, coord, nzhw, name, im) in enumerate(testloader):
        print(i, '=============================================')
        print('data', data.shape)
        print('target', target.shape)
        print('coord', coord.shape)
        print('nzhw', nzhw)
        print('name', name)
        if i > 10:
            break