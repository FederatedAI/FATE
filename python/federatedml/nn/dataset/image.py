import torch
from federatedml.nn.dataset.base import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np


class ImageDataset(Dataset):

    """

    A basic Image Dataset built on pytorch ImageFolder, supports simple image transform
    Given a folder path, ImageDataset will load images from this folder, images in this
    folder need to be organized in a Torch-ImageFolder format, see
    https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html for details.

    Image name will be automatically taken as the sample id.

    Parameters
    ----------
    center_crop : bool, use center crop transformer
    center_crop_shape: tuple or list
    generate_id_from_file_name: bool, whether to take image name as sample id
    file_suffix: str, default is '.jpg', if generate_id_from_file_name is True, will remove this suffix from file name,
                 result will be the sample id
    return_label: bool, return label or not, this option is for host dataset, when running hetero-NN
    float64: bool, returned image tensors will be transformed to double precision
    label_dtype: str, long, float, or double, the dtype of return label
    """

    def __init__(self, center_crop=False, center_crop_shape=None,
                 generate_id_from_file_name=True, file_suffix='.jpg',
                 return_label=True, float64=False, label_dtype='long'):

        super(ImageDataset, self).__init__()
        self.image_folder: ImageFolder = None
        self.center_crop = center_crop
        self.size = center_crop_shape
        self.return_label = return_label
        self.generate_id_from_file_name = generate_id_from_file_name
        self.file_suffix = file_suffix
        self.float64 = float64
        self.dtype = torch.float32 if not self.float64 else torch.float64
        avail_label_type = ['float', 'long', 'double']
        self.sample_ids = None
        assert label_dtype in avail_label_type, 'available label dtype : {}'.format(
            avail_label_type)
        if label_dtype == 'double':
            self.label_dtype = torch.float64
        elif label_dtype == 'long':
            self.label_dtype = torch.int64
        else:
            self.label_dtype = torch.float32

    def load(self, folder_path):

        # read image from folders
        if self.center_crop:
            transformer = transforms.Compose(
                [transforms.CenterCrop(size=self.size), transforms.ToTensor()])
        else:
            transformer = transforms.Compose([transforms.ToTensor()])

        if folder_path.endswith('/'):
            folder_path = folder_path[: -1]
        image_folder_path = folder_path
        folder = ImageFolder(root=image_folder_path, transform=transformer)
        self.image_folder = folder

        if self.generate_id_from_file_name:
            # use image name as its sample id
            file_name = self.image_folder.imgs
            ids = []
            for name in file_name:
                sample_id = name[0].split(
                    '/')[-1].replace(self.file_suffix, '')
                ids.append(sample_id)
            self.sample_ids = ids

    def __getitem__(self, item):
        if self.return_label:
            item = self.image_folder[item]
            return item[0].type(
                self.dtype), torch.tensor(
                item[1]).type(
                self.label_dtype)
        else:
            return self.image_folder[item][0].type(self.dtype)

    def __len__(self):
        return len(self.image_folder)

    def __repr__(self):
        return self.image_folder.__repr__()

    def get_classes(self):
        return np.unique(self.image_folder.targets).tolist()

    def get_sample_ids(self):
        return self.sample_ids


if __name__ == '__main__':
    pass
