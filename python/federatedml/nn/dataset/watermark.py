import os
import numpy as np
import pandas as pd
from federatedml.nn.dataset.base import Dataset
from federatedml.util import LOGGER
from federatedml.nn.dataset.image import ImageDataset


class WaterMarkImageDataset(Dataset):

    """
    A basic WaterMark Dataset built on pytorch ImageFolder
    This Dataset is used for Fed-IPR algorithm, see: https://arxiv.org/abs/2109.13236 for details
    It will contain two part: A normal dataset and a watermark dataset
    When training, the FedIPR Trainer will retrieve the normal dataset and watermark dataset from it
    Given a path to image folder, WaterMarkImageDataset will load images from this folder, by default,
    folder named 'normal' will be treated as normal dataset, folder named 'watermark' will be treated as watermark dataset
    You can adjust this behavior by setting normal_folder_name and watermark_folder_name in the parameters

    Parameters:
    ----------
    normal_folder_name: str, default is 'normal', the folder name of normal dataset
    watermark_folder_name: str, default is 'watermark', the folder name of watermark dataset
    """

    def __init__(self, normal_folder_name='normal', watermark_folder_name='watermark',
                 center_crop=False, center_crop_shape=None,
                 generate_id_from_file_name=True, file_suffix='.jpg',
                 float64=False, label_dtype='long'
                 ):

        super(WaterMarkImageDataset, self).__init__()
        self.normal_folder_name = normal_folder_name
        self.watermark_folder_name = watermark_folder_name

        self.normal_dataset = None
        self.watermark_dataset = None

        self.center_crop = center_crop
        self.size = center_crop_shape
        self.generate_id_from_file_name = generate_id_from_file_name
        self.file_suffix = file_suffix
        self.float64 = float64
        self.label_type = label_dtype

    def __getitem__(self, item):
        
        if item < 0:
            item = len(self) + item
        if item < 0:
            raise IndexError('index out of range')
        
        if item < len(self.normal_dataset):
            return ('normal', self.normal_dataset[item])
        else:
            return ('watermark', self.watermark_dataset[item - len(self.normal_dataset)])
        
    def __len__(self):
        return len(self.normal_dataset) + len(self.watermark_dataset)

    def load(self, file_path):
        
        # normal dataset path
        normal_path = os.path.join(file_path, self.normal_folder_name)
        # watermark dataset path
        watermark_path = os.path.join(file_path, self.watermark_folder_name)

        # load normal dataset
        self.normal_dataset = ImageDataset(
            center_crop=self.center_crop,
            center_crop_shape=self.size,
            generate_id_from_file_name=self.generate_id_from_file_name,
            file_suffix=self.file_suffix,
            float64=self.float64,
            label_dtype=self.label_type
        )
        self.normal_dataset.load(normal_path)
        # load watermark dataset
        self.watermark_dataset = ImageDataset(
            center_crop=self.center_crop,
            center_crop_shape=self.size,
            generate_id_from_file_name=self.generate_id_from_file_name,
            file_suffix=self.file_suffix,
            float64=self.float64,
            label_dtype=self.label_type
        )
        self.watermark_dataset.load(watermark_path)

    def get_normal_dataset(self):
        return self.normal_dataset
    
    def get_watermark_dataset(self):
        return self.watermark_dataset

    def get_classes(self):
        return self.noraml_dataset.get_classes()

    # def get_sample_ids(self):
    #     return self.sample_ids

    # def get_match_ids(self):
    #     return self.match_ids
