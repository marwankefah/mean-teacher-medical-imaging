import os
import torch
import random
import numpy as np
import h5py
from scipy.ndimage.interpolation import zoom
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import pandas as pd
from PIL import Image, ImageFile
import torchio as tio

from torch.utils.data import Dataset



class BaseFetaDataSets(Dataset):
    
    def __init__(self, base_dir=None, split='train',configs=None,mode='meanTeacher', num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.mode=mode
        self.labeled_idxs = []
        self.unlabeled_idxs = []
        self.configs=configs

        try:
            df = pd.read_csv(base_dir+'/data.csv', delimiter=",")
        except Exception as e:
            print("Error: please try and close the csv file, or check whether path is correct", e)
            raise

            #optimize by sending
        if self.split == 'train':
            dfTrainSubset=df[(df['datamode']=='train_labelled') | (df['datamode']=='train_unlabelled')]
            dfTrainSubset=dfTrainSubset.reset_index()

            #Todo clean this when cleaning the project
            # dfTrainSubset=self.update_unlabeled_path(dfTrainSubset)

            self.sample_list =list(zip(dfTrainSubset['image'].values.tolist(),dfTrainSubset['manual'].values.tolist()))
                #TODO check the indicies is actually true
            self.labeled_idxs=dfTrainSubset.index[dfTrainSubset['datamode']=='train_labelled'].tolist()
            self.unlabeled_idxs=dfTrainSubset.index[dfTrainSubset['datamode']=='train_unlabelled'].tolist()

        elif self.split == 'val':
            dfTrainSubset = df[df['datamode'] == 'val_labelled']
            dfTrainSubset=dfTrainSubset.reset_index()

            self.sample_list = list(zip(dfTrainSubset['image'].values.tolist(), dfTrainSubset['manual'].values.tolist()))
            self.labeled_idxs=dfTrainSubset.index[dfTrainSubset['datamode']=='val_labelled'].tolist()

        elif self.split == 'test':
            dfTrainSubset = df[df['datamode'] == 'test_labelled']
            dfTrainSubset=dfTrainSubset.reset_index()
            self.sample_list = list(zip(dfTrainSubset['image'].values.tolist(), dfTrainSubset['manual'].values.tolist()))
            self.labeled_idxs = dfTrainSubset.index[dfTrainSubset['datamode'] == 'test_labelled'].tolist()
        elif self.split=='train_labelled':
            dfTrainSubset = df[(df['datamode'] == 'train_labelled')]
            dfTrainSubset = dfTrainSubset.reset_index()

            self.sample_list = list(
                zip(dfTrainSubset['image'].values.tolist(), dfTrainSubset['manual'].values.tolist()))
            # TODO check the indicies is actually true
            self.labeled_idxs = dfTrainSubset.index[dfTrainSubset['datamode'] == 'train_labelled'].tolist()
        elif self.split=='train_unlabelled':
            dfTrainSubset = df[(df['datamode'] == 'train_unlabelled')]
            dfTrainSubset = dfTrainSubset.reset_index()

            self.sample_list = list(
                zip(dfTrainSubset['image'].values.tolist(), dfTrainSubset['manual'].values.tolist()))
            # TODO check the indicies is actually true
            self.unlabeled_idxs = dfTrainSubset.index[dfTrainSubset['datamode'] == 'train_unlabelled'].tolist()



        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        imgPath=self.sample_list[idx][0]
        labelsPath=self.sample_list[idx][1]
        image = Image.open(self._base_dir+imgPath).convert('L')
    
        if labelsPath != 'none':
            # Masks are 0/255
            mask = Image.open(self._base_dir+labelsPath).convert('L')
            mask = mask.point(lambda p: p == 1 and 1)
        else:
            mask = Image.new('L', image.size)

        # sample = {'image': image, 'label': mask}
        sample = {'image': np.asarray(image), 'label': np.asarray(mask)}

        # sample = {'image':, 'label':}

        sampleSubject=tio.Subject(image=tio.ScalarImage(tensor=sample['image'].reshape((1,)+image.size+(1,))), label=tio.LabelMap(tensor=sample['label'].reshape((1,)+image.size+(1,))))
        if self.transform:
            # sample = self.transform(sample)
            transformedSubject=self.transform(sampleSubject)
            sample['image']=transformedSubject['image']['data'].squeeze(-1)
            sample['label']=transformedSubject['label']['data'].squeeze()

        sample["idx"] = idx

        return sample

    #TODO update to take folder to update sample list instead and reconstruct
    def update_unlabeled_path(self,df):
        if self.mode !='studyBuddy':
            return df

        psuedoLabelsPath='/psuedo/'

        if not os.path.exists(self._base_dir+psuedoLabelsPath):
            os.makedirs(self._base_dir+psuedoLabelsPath)

        newPsuedoList=(psuedoLabelsPath+ df[df['datamode'] == 'train_unlabelled']['image'].str.rsplit("\\", n=1, expand=True)[
                                                                      1]).values
        newPsuedoImageList=df[df['datamode'] == 'train_unlabelled']['image'].values

        for i,j in zip(newPsuedoList,newPsuedoImageList):

            image = Image.open(self._base_dir + j).convert('L')

            mask = Image.new('L', image.size)
            mask.save(self._base_dir+i)

        df.loc[df['datamode'] == 'train_unlabelled', 'manual'] = newPsuedoList

        return df


class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        if self.split == 'train':
            with open(self._base_dir + '/train_slices.list', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]

        elif self.split == 'val':
            with open(self._base_dir + '/val.list', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir +
                            "/data/slices/{}.h5".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.split == "train":
            sample = self.transform(sample)
        sample["idx"] = idx
        return sample


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)


        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)

        label = torch.from_numpy(label.astype(np.uint8))

        sample = {'image': image, 'label': label}
        return sample


class ResizeTransform(object):
    def __init__(self, output_size,mode='train_unlabeled'):
        self.output_size = output_size
        self.mode=mode

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        x, y = image.shape
        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)


        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {'image': image, 'label': label}
        return sample

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


