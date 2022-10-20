from federatedml.nn_.dataset.base import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms


class ImageDataset(Dataset):

    def __init__(self, center_crop=False, center_crop_shape=None):
        super(ImageDataset, self).__init__()
        self.image_folder = None
        self.center_crop = center_crop
        self.size = center_crop_shape
        
    def load(self, file_path):
        # read image from folders
        if self.center_crop:
            transformer = transforms.Compose([transforms.CenterCrop(size=self.size), transforms.ToTensor()])
        else:
            transformer = transforms.Compose([transforms.ToTensor()])
            
        folder = ImageFolder(root=file_path, transform=transformer)
        self.image_folder = folder

    def __getitem__(self, item):
        return self.image_folder[item]

    def __len__(self):
        return len(self.image_folder)

    def __repr__(self):
        return self.image_folder.__repr__()


if __name__ == '__main__':
    pass
