import torch as t
import pandas as pd
from federatedml.nn.dataset.base import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from transformers import BertTokenizerFast

class MixFeatureDataset(Dataset):
    
    def __init__(self, output_image_size):
        super(MixFeatureDataset, self).__init__() # 记得这个
        self.output_image_size = output_image_size
        self.image_folder = None
        self.text = None
        self.word_idx = None
        self.vocab_size = 0
        self.sample_ids = None
        
    # 需要实现的接口 load, load接受一个参数path, 算法运行时将会把path传给这个接口
    def load(self, path):
        # 处理图像数据集
        transformer = transforms.Compose([transforms.CenterCrop(size=self.output_image_size), transforms.ToTensor()])
        self.image_folder = ImageFolder(root=path+'/flicker/images', transform=transformer)
        
        # 处理文本数据集，将其符号化（tokenize)
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false" # avoid tokenizer problem
        
        self.text = pd.read_csv(path+'/text.csv')
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased') # 用bert tokenizer
        text_list = list(self.text.text)
        self.word_idx = tokenizer(text_list, padding=True, return_tensors='pt',
                                  truncation=True, max_length=20)['input_ids']
        self.vocab_size = tokenizer.vocab_size
        
        # 保证image数据集图片的id能与文本的id对应上
        img_ids = [i[0].split('/')[-1].replace('.jpg', '') for i in self.image_folder.imgs]
        text_ids = list(self.text.id)
        assert img_ids == text_ids
        print('id match!')
        self.sample_ids = text_ids
    
    
    # 需要实现的接口1 len
    def __len__(self):
        return len(self.image_folder)
    
    # 需要实现的接口2 getitem, 返回（数据，label)
    def __getitem__(self, idx):
        img, label = self.image_folder[idx]
        text = self.word_idx[idx]
        return (img, text), t.tensor(label).type(t.float32)
    
    # 此接口可选，如果不实现，FATE将拿不到sample id，便自动生成
    def get_sample_ids(self,):
        return self.sample_ids
