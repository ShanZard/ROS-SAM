import torch.utils.data.dataset as Dataset
import numpy as np
import torch
import os 
import skimage.io as skio
from mmdet.datasets.transforms.transforms import *
from mmcv.transforms import RandomResize,CenterCrop
import random
class ROSSAM_Dataset(Dataset.Dataset):
    def __init__(self, datasetroot,split):
        self.imgpath = os.path.join(datasetroot,split,'img')
        self.boxespath = os.path.join(datasetroot,split,'boxes')
        self.onehotpath = os.path.join(datasetroot,split,'onehotmask')
        self.img = os.listdir(self.imgpath)
        self.img = random.sample(self.img,len(self.img))
        self.transRandomResize = RandomResize(scale=(1024,1024),ratio_range=(0.1, 4.0),resize_type='Resize',keep_ratio=True)
        self.transRandomCrop =RandomCrop(crop_size=(1024,1024),crop_type='absolute',allow_negative_crop =True,recompute_bbox=True)
        self.transFixShapeResize = FixShapeResize(width=1024,height=1024,pad_val=dict(img=144, seg=0),keep_ratio=True)
        self.transRandomFlip = RandomFlip(0.1,'diagonal')
        self.transPhotoMetricDistortion= PhotoMetricDistortion()
        self.transPad = Pad(size=(1024,1024),pad_val=dict(img=144, seg=0))
        self.train_mode = split
    def __len__(self):
        return len(self.img)
    def __getitem__(self, index):
        sample ={}
        imagename = self.img[index]
        img  = skio.imread(os.path.join(self.imgpath,self.img[index]))
        boxes = np.load(os.path.join(self.boxespath,self.img[index].replace('.png','.npy')))
        ori_boxes = boxes.copy()
        if len(boxes) == 0:
            onehotmask = np.zeros_like(img)
            orginalonehotmask = onehotmask
            img = torch.tensor(img.copy(), dtype=torch.float32)
            img = torch.transpose(torch.transpose(img,1,2),0,1)
            onehotmask = torch.tensor(onehotmask.copy(), dtype=torch.float32)
            onehotmask = onehotmask.permute(2,0,1)
            boxes = torch.tensor(boxes.copy(), dtype=torch.float32)
            sample ={'img':img,'boxes':boxes,'onehotmask':onehotmask,'ori_mask':orginalonehotmask,'ori_boxes':ori_boxes,'filename':imagename}
            return  sample
        
        onehotmask = np.load(os.path.join(self.onehotpath,self.img[index].replace('.png','.npy'))).transpose(1, 2, 0)
        orginalonehotmask = onehotmask.copy()
        sample = {'img':img,'gt_bboxes':boxes,'gt_seg_map':onehotmask}    
        if self.train_mode == 'train':    
            sample = self.transRandomResize(sample)
            sample = self.transRandomCrop(sample)
            sample = self.transRandomFlip(sample)           
            sample = self.transPhotoMetricDistortion(sample)
            sample = self.transPad(sample)            
        else :
            sample =sample
        img = sample['img']
        onehotmask = sample['gt_seg_map']
        boxes = sample['gt_bboxes']
        if len(boxes) == 0:
            onehotmask = np.zeros_like(img)
            orginalonehotmask = onehotmask
            img = torch.tensor(img.copy(), dtype=torch.float32)
            img = torch.transpose(torch.transpose(img,1,2),0,1)
            onehotmask = torch.tensor(onehotmask.copy(), dtype=torch.float32)
            onehotmask = onehotmask.permute(2,0,1)
            boxes = torch.tensor(boxes.copy(), dtype=torch.float32)
            sample ={'img':img,'boxes':boxes,'onehotmask':onehotmask,'ori_mask':orginalonehotmask,'ori_boxes':ori_boxes,'filename':imagename}
            return  sample
        if  len(onehotmask.shape) == 2:
            onehotmask = np.expand_dims(onehotmask, axis=2) 
        if onehotmask.shape[2] !=boxes.shape[0]:   
            zero_channels = np.all(onehotmask == 0, axis=(0, 1))
            onehotmask = onehotmask[ :, :,~zero_channels]       
        img = torch.tensor(img.copy(), dtype=torch.float32)
        img = torch.transpose(torch.transpose(img,1,2),0,1)
        onehotmask = torch.tensor(onehotmask.copy(), dtype=torch.float32)
        onehotmask = onehotmask.permute(2,0,1)
        boxes = torch.tensor(boxes.copy(), dtype=torch.float32)
        sample ={'img':img,'boxes':boxes,'onehotmask':onehotmask,'ori_mask':orginalonehotmask,'ori_boxes':ori_boxes,'filename':imagename}
        return sample
