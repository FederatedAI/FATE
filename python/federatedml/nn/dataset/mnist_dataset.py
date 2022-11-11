import numpy as np
from federatedml.nn.dataset.base import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms


class MNISTDataset(Dataset):
    
    def __init__(self, return_label=True):  # guest方有标签，return label = True, host方无标签，return label = False
        super(MNISTDataset, self).__init__() # 记得这个
        self.return_label = return_label
        self.image_folder = None
        
    def load(self, path):  # 实现label 接口，从path读取图像， 设置sample ids
        
        # 读取
        self.image_folder = ImageFolder(root=path, transform=transforms.Compose([transforms.ToTensor()]))
        # 用image的名字作为id
        ids = []
        for image_name in self.image_folder.imgs:
            ids.append(image_name[0].split('/')[-1].replace('.jpg', ''))
        self.set_sample_ids(ids)

        return self
        
    def get_classes(self, ): # get classes接口，返回class种类， guest方需要用到
        return np.unique(self.image_folder.targets).tolist()
    
    def __len__(self,):  # len接口
        return len(self.image_folder)
    
    def __getitem__(self, idx): # get item 接口, 注意return label
        ret = self.image_folder[idx]
        img = ret[0][0].flatten() # 转换为一个flatten tensor 784维度
        if self.return_label:
            return img, ret[1] # img & label
        else:
            return img # no label, for host
