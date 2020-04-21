import torch
from torch import nn
from cv_task.layers import *

config = {}
# config['anchors'] = [ 10.0, 30.0, 60.]
config['anchors'] = [30.0]
config['chanel'] = 1
config['crop_size'] = [128, 128, 128]
config['stride'] = 4
config['max_stride'] = 16
config['num_neg'] = 800
config['th_neg'] = 0.02
config['th_pos_train'] = 0.5
config['th_pos_val'] = 1
config['num_hard'] = 1  ## 原来是2，每个图里选两个negative
config['bound_size'] = 12
config['reso'] = 1
config['sizelim'] = 6. #mm
config['sizelim2'] = 30
config['sizelim3'] = 40
config['aug_scale'] = True
config['r_rand_crop'] = 0.3
config['pad_value'] = 170
config['augtype'] = {'flip':True,'swap':False,'scale':True,'rotate':False}
config['blacklist'] = ['868b024d9fa388b7ddab12ec1c06af38','990fbe3f0a1b53878669967b9afd1441','adc3bbc63d40f8761c59be10f1e504c3']

#config['blacklist'] = ['868b024d9fa388b7ddab12ec1c06af38','d92998a73d4654a442e6d6ba15bbb827','990fbe3f0a1b53878669967b9afd1441','820245d8b211808bd18e78ff5be16fdb','adc3bbc63d40f8761c59be10f1e504c3',
#                       '417','077','188','876','057','087','130','468']

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # The first few layers consumes the most memory, so use simple convolution to save memory.
        # Call these layers preBlock, i.e., before the residual blocks of later layers.
        self.preBlock = nn.Sequential(
            nn.Conv3d(1, 24, kernel_size = 3, padding = 1),
            #nn.BatchNorm3d(24),
            nn.ReLU(inplace = True),
            nn.Conv3d(24, 24, kernel_size = 3, padding = 1),
            #nn.BatchNorm3d(24),
            nn.ReLU(inplace = True))
        
        # 3 poolings, each pooling downsamples the feature map by a factor 2.
        # 3 groups of blocks. The first block of each group has one pooling.
        num_blocks_forw = [2,2,3,3]
        num_blocks_back = [3,3]
        self.featureNum_forw = [24,32,64,64,64]
        self.featureNum_back =    [128,64,64]
        for i in range(len(num_blocks_forw)):
            blocks = []
            for j in range(num_blocks_forw[i]):
                if j == 0:
                    blocks.append(PostRes(self.featureNum_forw[i], self.featureNum_forw[i+1]))
                else:
                    blocks.append(PostRes(self.featureNum_forw[i+1], self.featureNum_forw[i+1]))
            setattr(self, 'forw' + str(i + 1), nn.Sequential(*blocks))

            
        for i in range(len(num_blocks_back)):
            blocks = []
            for j in range(num_blocks_back[i]):
                if j == 0:
                    if i==0:
                        addition = 3
                    else:
                        addition = 0
                    blocks.append(PostRes(self.featureNum_back[i+1]+self.featureNum_forw[i+2]+addition, self.featureNum_back[i]))
                else:
                    blocks.append(PostRes(self.featureNum_back[i], self.featureNum_back[i]))
            setattr(self, 'back' + str(i + 2), nn.Sequential(*blocks))

        self.maxpool1 = nn.MaxPool3d(kernel_size=2,stride=2,return_indices =True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2,stride=2,return_indices =True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2,stride=2,return_indices =True)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2,stride=2,return_indices =True)
        self.unmaxpool1 = nn.MaxUnpool3d(kernel_size=2,stride=2)
        self.unmaxpool2 = nn.MaxUnpool3d(kernel_size=2,stride=2)

        self.path1 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size = 2, stride = 2),
            #nn.BatchNorm3d(64),
            nn.ReLU(inplace = True))
        self.path2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size = 2, stride = 2),
            #nn.BatchNorm3d(64),
            nn.ReLU(inplace = True))
        self.drop = nn.Dropout3d(p = 0.5, inplace = False)
        self.output = nn.Sequential(nn.Conv3d(self.featureNum_back[0], 64, kernel_size = 1),
                                    nn.ReLU(inplace = True),
                                    #nn.Dropout3d(p = 0.3),
                                   nn.Conv3d(64, 5 * len(config['anchors']), kernel_size = 1))
    '''
    def forward(self, x, coord):
        #x, coord 
        out = self.preBlock(x)#16
        out_pool,indices0 = self.maxpool1(out)
        del out ## Siyang:节约内存
        out1 = self.forw1(out_pool)#32
        del out_pool ## Siyang:节约内存
        out1_pool,indices1 = self.maxpool2(out1)
        out2 = self.forw2(out1_pool)#64
        del out1_pool  ## Siyang:节约内存
        del indices1  ## Siyang:节约内存
        #out2 = self.drop(out2)
        out2_pool,indices2 = self.maxpool3(out2)
        out3 = self.forw3(out2_pool)#96
        del out2_pool  ## Siyang:节约内存
        del indices2  ## Siyang:节约内存
        out3_pool,indices3 = self.maxpool4(out3)
        out4 = self.forw4(out3_pool)#96
        del out3_pool  ## Siyang:节约内存
        del indices3  ## Siyang:节约内存
        #out4 = self.drop(out4)
        
        rev3 = self.path1(out4)
        comb3 = self.back3(torch.cat((rev3, out3), 1))#96+96
        del rev3  ## Siyang:节约内存
        #comb3 = self.drop(comb3)
        rev2 = self.path2(comb3)
        del comb3  ## Siyang:节约内存
        
        comb2 = self.back2(torch.cat((rev2, out2, coord), 1))#64+64
        del rev2  ## Siyang:节约内存
        comb2 = self.drop(comb2)
        out = self.output(comb2)
        del comb2  ## Siyang:节约内存
        size = out.size()
        out = out.view(out.size(0), out.size(1), -1)
        #out = out.transpose(1, 4).transpose(1, 2).transpose(2, 3).contiguous()
        out = out.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], len(config['anchors']), 5)
        #out = out.view(-1, 5)
        return out
    '''

    def forward(self, x, coord):
        out = self.preBlock(x)#16
        out_pool,indices0 = self.maxpool1(out)
        del out ## Siyang:节约内存
        out1 = self.forw1(out_pool)#32
        del out_pool ## Siyang:节约内存
        out1_pool,indices1 = self.maxpool2(out1)
        out2 = self.forw2(out1_pool)#64
        del out1_pool  ## Siyang:节约内存
        del indices1  ## Siyang:节约内存
        #out2 = self.drop(out2)
        out2_pool,indices2 = self.maxpool3(out2)
        out3 = self.forw3(out2_pool)#96
        del out2_pool  ## Siyang:节约内存
        del indices2  ## Siyang:节约内存
        out3_pool,indices3 = self.maxpool4(out3)
        out4 = self.forw4(out3_pool)#96
        del out3_pool  ## Siyang:节约内存
        del indices3  ## Siyang:节约内存
        #out4 = self.drop(out4)
        
        rev3 = self.path1(out4)
        comb3 = self.back3(torch.cat((rev3, out3), 1))#96+96
        del rev3  ## Siyang:节约内存
        #comb3 = self.drop(comb3)
        rev2 = self.path2(comb3)
        del comb3  ## Siyang:节约内存
        
        comb2 = self.back2(torch.cat((rev2, out2, coord), 1))#64+64
        del rev2  ## Siyang:节约内存
        comb2 = self.drop(comb2)
        out = self.output(comb2)
        del comb2  ## Siyang:节约内存
        size = out.size()
        out = out.view(out.size(0), out.size(1), -1)
        #out = out.transpose(1, 4).transpose(1, 2).transpose(2, 3).contiguous()
        out = out.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], len(config['anchors']), 5)
        #out = out.view(-1, 5)
        return out

    def forward_zhiyuan(self, model_input):
        #x, coord 
        out = self.preBlock(model_input[0,0:1*1*128*128*128].viewq([1, 1, 128, 128, 128]))#16
        out_pool,indices0 = self.maxpool1(out)
        del out ## Siyang:节约内存
        out1 = self.forw1(out_pool)#32
        del out_pool ## Siyang:节约内存
        out1_pool,indices1 = self.maxpool2(out1)
        out2 = self.forw2(out1_pool)#64
        del out1_pool  ## Siyang:节约内存
        del indices1  ## Siyang:节约内存
        #out2 = self.drop(out2)
        out2_pool,indices2 = self.maxpool3(out2)
        out3 = self.forw3(out2_pool)#96
        del out2_pool  ## Siyang:节约内存
        del indices2  ## Siyang:节约内存
        out3_pool,indices3 = self.maxpool4(out3)
        out4 = self.forw4(out3_pool)#96
        del out3_pool  ## Siyang:节约内存
        del indices3  ## Siyang:节约内存
        #out4 = self.drop(out4)
        
        rev3 = self.path1(out4)
        comb3 = self.back3(torch.cat((rev3, out3), 1))#96+96
        del rev3  ## Siyang:节约内存
        #comb3 = self.drop(comb3)
        rev2 = self.path2(comb3)
        del comb3  ## Siyang:节约内存
        
        comb2 = self.back2(torch.cat((rev2, out2, model_input[0,1*1*128*128*128:].view([1,3,32,32,32])), 1))#64+64
        del rev2  ## Siyang:节约内存
        comb2 = self.drop(comb2)
        out = self.output(comb2)
        del comb2  ## Siyang:节约内存
        size = out.size()
        out = out.view(out.size(0), out.size(1), -1)
        #out = out.transpose(1, 4).transpose(1, 2).transpose(2, 3).contiguous()
        out = out.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], len(config['anchors']), 5)
        #out = out.view(-1, 5)
        return out

    
def get_model():
    net = Net()
    loss = layers_loss(config['num_hard'])
    get_pbb = GetPBB(config)
    return config, net, loss, get_pbb
