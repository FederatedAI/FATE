import os
import shutil
import numpy as np

from scipy.io import loadmat
import numpy as np
import h5py
import pandas
import scipy
from scipy.ndimage.interpolation import zoom
from skimage import measure
import SimpleITK as sitk
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
from skimage.morphology import convex_hull_image
import pandas
from multiprocessing import Pool
from functools import partial
import sys
import glob
from tqdm import tqdm
import pandas as pd
import cv2


def resample(imgs, spacing, new_spacing, order=2):
    if len(imgs.shape) == 3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        imgs = zoom(imgs, resize_factor, mode = 'nearest',order=order)
        return imgs, true_spacing
    elif len(imgs.shape)==4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:,:,:,i]
            newslice,true_spacing = resample(slice,spacing,new_spacing)
            newimg.append(newslice)
        newimg=np.transpose(np.array(newimg),[1,2,3,0])
        return newimg,true_spacing
    else:
        raise ValueError('wrong shape')

def worldToVoxelCoord(worldCoord, origin, spacing):
     
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord

def lumTrans(img):
    lungwin = np.array([-1200.,600.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype('uint8')
    return newimg

def load_itk_image(filename):
    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transformM = np.round(transformM)
        if np.any(transformM != np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
     
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
     
    return numpyImage, numpyOrigin, numpySpacing,isflip

def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1  = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1)>0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2)>1.5*np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3,1)  
    dilatedMask = binary_dilation(convex_mask,structure=struct,iterations=10) 
    return dilatedMask

def savenpy_luna(id,annos,filelist,luna_segment,luna_data,savepath):
    islabel = True
    isClean = True
    resolution = np.array([1,1,1])
    # resolution = np.array([0.5, 0.5, 0.5])
    name = str(filelist[id])
    
    Mask, origin, spacing, isflip = load_itk_image(os.path.join(luna_segment,name+'.mhd'))
    if isflip:
        Mask = Mask[:,::-1,::-1]
    newshape = np.round(np.array(Mask.shape)*spacing/resolution).astype('int')
    m1 = Mask==3
    m2 = Mask==4
    Mask = m1+m2
    
    xx,yy,zz= np.where(Mask)
    box = np.array([[np.min(xx),np.max(xx)],[np.min(yy),np.max(yy)],[np.min(zz),np.max(zz)]])
    box = box*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
    box = np.floor(box).astype('int')
    margin = 5
    extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T

    #this_annos = annos[annos['seriesuid'] == name].values      

    if isClean:
        convex_mask = m1
        dm1 = process_mask(m1)
        dm2 = process_mask(m2)
        dilatedMask = dm1+dm2
        Mask = m1+m2
        extramask = dilatedMask ^ Mask
        bone_thresh = 210
        pad_value = 170

        sliceim,origin,spacing,isflip = load_itk_image(os.path.join(luna_data[id],name+'.mhd'))
        if isflip:
            sliceim = sliceim[:,::-1,::-1]
            print('flip!')
        sliceim = lumTrans(sliceim)
        sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
        bones = (sliceim*extramask)>bone_thresh
        sliceim[bones] = pad_value
        
        #print(sliceim.shape)

        sliceim1,_ = resample(sliceim,spacing,resolution,order=1)
        sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],
                    extendbox[1,0]:extendbox[1,1],
                    extendbox[2,0]:extendbox[2,1]]
        sliceim = sliceim2[np.newaxis,...]
        #np.save(os.path.join(savepath,name+'_clean.npy'),sliceim)

    #print(sliceim.shape)

    if islabel:
        this_annos = annos[annos['seriesuid'] == name].values
        label = []
        if len(this_annos)>0:
            
            for c in this_annos:
                pos = worldToVoxelCoord(c[1:4][::-1],origin=origin,spacing=spacing)
                if isflip:
                    pos[1:] = Mask.shape[1:3]-pos[1:]
                label.append(np.concatenate([pos,[c[4]/spacing[1]]]))
            
        label = np.array(label)
        if len(label)==0:
            label2 = np.array([[0,0,0,0]])
        else:
            label2 = np.copy(label).T
            label2[:3] = label2[:3]*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
            label2[3] = label2[3]*spacing[1]/resolution[1]
            label2[:3] = label2[:3]-np.expand_dims(extendbox[:,0],1)
            label2 = label2[:4].T

        if label2[0][0]>0:
            preview_idx = int(label2[0][0])
            r = sliceim[0,preview_idx,:,:]
            g = r.copy()
            b = r
            xmin = int(label2[0][1]-label2[0][3]/2)
            xmax = int(label2[0][1]+label2[0][3]/2)
            ymin = int(label2[0][2]-label2[0][3]/2)
            ymax = int(label2[0][2]+label2[0][3]/2)
            g[xmin:xmax,ymin:ymax]=200
        else:
            preview_idx = 100
            r = sliceim[0,preview_idx,:,:]
            g = r
            b = r
        preview_im = np.stack((r,g,b),axis=2)
        cv2.imwrite(os.path.join(savepath,name+'_preview.jpg'),preview_im)
        np.save(os.path.join(savepath,name+'_label.npy'),label2)
        
    np.savez(os.path.join(savepath,name+'.npz'),im = sliceim,labels = label2)

    if len(label)>0:
        return name, label2
    else:
        return name, []

subset_no = 10
# Input Dirs
luna_base = '/workspace/LUNA/data/'
luna_raw_dirs = [luna_base+'subset'+str(i)+'/' for i in range(subset_no)]
luan_lung_seg_dir = luna_base+'seg-lungs-LUNA16/'

annos = pd.read_csv(luna_base+'CSVFILES/annotations.csv')

# Ouput Dir
luna_npy_dir = './luna_npy/'
csv_files_dir = './csv_files/'

if not os.path.exists(luna_npy_dir):
    os.makedirs(luna_npy_dir)
if not os.path.exists(csv_files_dir):
    os.makedirs(csv_files_dir)

filelistFull = []
for i in range(len(luna_raw_dirs)):
    tmpRawDir = luna_raw_dirs[i]
    tmpFileList = glob.glob(tmpRawDir+'*.mhd')
    filelistFull += tmpFileList


luna_segment = luan_lung_seg_dir
filelist = [os.path.basename(x)[:-4] for x in filelistFull]
luna_data = [os.path.split(x)[0] for x in filelistFull]
filelist_datapath = [os.path.join(luna_npy_dir, x+'.npz') for x in filelist]


name_rec = []
label_rec = []
for id in tqdm(range(len(filelist))):
    name, label = savenpy_luna(id,annos,filelist,luna_segment,luna_data,luna_npy_dir)
    tmpN = len(label)
    if tmpN>0:
        tmp_name = [name for i in range(tmpN)]
        name_rec += tmp_name
        label_rec += label.tolist()

dfDatapath = pd.DataFrame(data=[filelist,filelist_datapath],index=['filename','filePath']).transpose()
dfBboxes = pd.DataFrame(data=[name_rec,label_rec], index=['filename','bbox']).transpose()

#%%
#import pandas as pd
#dfBboxes = pd.read_csv('luna_bboxlist.csv',index_col=0)
#dfDatapath = pd.read_csv('luna_datapath.csv', index_col=0)
#dfBboxes = dfBboxes.rename(columns={'name':'filename'})
#dfDatapath = dfDatapath.rename(columns={'name':'filename'})

#%%
dfDatapath = dfDatapath.set_index('filename')
dfFromSubset = pd.DataFrame(data = [filelist,luna_data], index = ['filename','from_subset']).transpose()
#dfFromSubset = pd.read_csv('tmp.csv')
dfFromSubset = dfFromSubset.set_index('filename')
def getNpzPath(t):
    return dfDatapath.loc[t.filename].filePath
def getSubset(t):
    
    return int(dfFromSubset.loc[t.filename].from_subset[-1])

dfBboxes['npz_path'] = dfBboxes.apply(getNpzPath,axis=1)
dfBboxes['from_subset'] = dfBboxes.apply(getSubset,axis=1)

#%%
dfDatapath.to_csv(csv_files_dir+'luna_datapath.csv')
dfBboxes.to_csv(csv_files_dir+'luna_bboxlist.csv')
