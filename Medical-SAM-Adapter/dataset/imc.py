import os
import joblib 
import numpy as np
import pandas as pd
import torch
from PIL import Image
import tifffile
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import random_box, random_click
import random 

def train_transform(crop_size=256,img_size=256):
    transforms = []
    transforms.append(A.RandomCrop(crop_size,crop_size,p=1.0))
    transforms.append(A.Resize(img_size,img_size,p=1.0))
    # transforms.append(A.HorizontalFlip(p=0.5))
    # transforms.append(A.RandomRotate90(p=0.3))
    # transforms.append(A.GaussNoise(p=0.3))
    transforms.append(ToTensorV2(p=1.0))
    return A.Compose(transforms,p=1.)

def test_transform(crop_size=256,img_size=256):
    transforms = []
    # transforms.append(A.RandomCrop(crop_size,crop_size,p=1.0))
    # transforms.append(A.Resize(img_size,img_size,p=1.0))
    # transforms.append(A.HorizontalFlip(p=0.5))
    # transforms.append(A.RandomRotate90(p=0.3))
    # transforms.append(A.GaussNoise(p=0.3))
    transforms.append(ToTensorV2(p=1.0))
    return A.Compose(transforms,p=1.)


class IMCdataset(Dataset):
    def __init__(self, args, data_path, mean_std_path, label_path, mode):
        self.mean_std = joblib.load(mean_std_path)
        self.pixel_mean = np.array(self.mean_std['ch_mean'])
        self.pixel_std = np.array(self.mean_std['ch_std'])
        self.data_ext = '_fullstacks.tiff'
        self.label_ext = '_pred_Probabilities_cell_mask.tiff'
        img_paths_all = ['_'.join(f.split('_')[:3]) for f in os.listdir(label_path)]
        if mode == 'Training':
            self.img_path = img_paths_all
        if mode == 'Validing':
            self.img_path = random.sample(img_paths_all,len(img_paths_all)//5)
        if mode == 'Testing':
            self.img_path = img_paths_all
        self.prompt = 'click'
        self.label_path = label_path
        self.hv_path = args.hv_path
        self.data_path = data_path
        self.point_num = args.point_num
        self.data_path = data_path
        self.image_size = args.image_size
        self.mode = mode 
    def __len__(self):
        if self.mode == 'Training':
            return len(self.img_path)
        else:
            return len(self.img_path)
    def __getitem__(self,idx):
        inouts = 1
        index = random.randint(0,len(self.img_path)-1)
        image = tifffile.imread(os.path.join(self.data_path,self.img_path[index]+self.data_ext)).transpose(1,2,0).astype(np.float32)
        H_img,W_img,C = image.shape
        label = tifffile.imread(os.path.join(self.label_path,self.img_path[index]+self.label_ext)).astype(np.int16)
        hv = tifffile.imread(os.path.join(self.hv_path,self.img_path[index]+self.label_ext))
        H_L,W_L = label.shape

        if image.shape[0]>label.shape[0] and image.shape[1]>label.shape[1]:
            image = image[:H_L,:W_L]
        else:
            label = label[:H_img,:W_img]
            hv = hv[:H_img,:W_img]

        # image = (image-self.pixel_mean)/self.pixel_std
        image = image[:,:,3:40]
        # print(self.image_size)
        transform = train_transform(img_size=self.image_size)
        
        try:
            after_aug = transform(image=image, masks=[label,hv])
        except: 
            print("error")
            # print(self.img_path[index],label.shape,image.shape)
        image_tensor = after_aug['image']
        cell_tensor = after_aug['masks'][0]
        hv_tensor = after_aug['masks'][1]
        # print('img,label',image_tensor.shape,label_tensor.shape)
        label_tensor=cell_tensor.clone()
        label_tensor[cell_tensor>0]=1
        prompt_tensor = (label_tensor>0).long()
        _, pt = random_click(np.array(prompt_tensor),point_num=self.point_num)
        
        return {
            'image': image_tensor,
            'label': label_tensor,
            'hv_label': hv_tensor,
            'cell_label': cell_tensor,
            'p_label': torch.LongTensor([inouts]*self.point_num),
            'pt':pt,
            'image_meta_dict': {'filename_or_obj':self.img_path[index]}
        }

class IMCdataset_Test(Dataset):
    def __init__(self, args, data_path, mean_std_path, label_path, mode):
        self.mean_std = joblib.load(mean_std_path)
        self.pixel_mean = np.array(self.mean_std['ch_mean'])
        self.pixel_std = np.array(self.mean_std['ch_std'])
        self.data_ext = '_fullstacks.tiff'
        self.label_ext = '_pred_Probabilities_cell_mask.tiff'
        img_paths_all = ['_'.join(f.split('_')[:3]) for f in os.listdir(label_path)]
        if mode == 'Training':
            self.img_path = img_paths_all
        if mode == 'Validing':
            self.img_path = random.sample(img_paths_all,len(img_paths_all)//5)
        if mode == 'Testing':
            self.img_path = img_paths_all
        self.prompt = 'click'
        self.label_path = label_path
        self.hv_path = args.hv_path
        self.data_path = data_path
        self.point_num = args.point_num
        self.data_path = data_path
        self.image_size = args.image_size
        self.mode = mode 
        self.test_infer_dict = joblib.load("/mnt/mydisk/zzf/Medical-SAM-Adapter/dataset/IMC_melanoma.pkl")
    def __len__(self):
        return len(self.test_infer_dict)
    def __getitem__(self,index):
        inouts = 1
        # index = random.randint(0,len(self.img_path)-1)
        img_p = self.test_infer_dict[index][0]
        image = tifffile.imread(os.path.join(self.data_path,img_p+self.data_ext)).transpose(1,2,0).astype(np.float32)
        H_img,W_img,C = image.shape
        label = tifffile.imread(os.path.join(self.label_path,img_p+self.label_ext)).astype(np.int16)
        hv = tifffile.imread(os.path.join(self.hv_path,img_p+self.label_ext))
        H_L,W_L = label.shape

        if image.shape[0]>label.shape[0] and image.shape[1]>label.shape[1]:
            image = image[:H_L,:W_L]
        else:
            label = label[:H_img,:W_img]
            hv = hv[:H_img,:W_img]

        # image = (image-self.pixel_mean)/self.pixel_std
        Hi,Wi = self.test_infer_dict[index][1], self.test_infer_dict[index][2]

        image = image[:,:,3:40]
        image_infer = np.zeros((self.image_size,self.image_size,37)).astype(np.float32)
        label_infer = np.zeros((self.image_size,self.image_size)).astype(np.int16)
        hv_infer = np.zeros((self.image_size,self.image_size,2))

        image_ = image[Hi:Hi+self.image_size,Wi:Wi+self.image_size]
        label_ = label[Hi:Hi+self.image_size,Wi:Wi+self.image_size]
        hv_ = hv[Hi:Hi+self.image_size,Wi:Wi+self.image_size]

        image_infer[:image_.shape[0],:image_.shape[1]] = image_
        label_infer[:label_.shape[0],:label_.shape[1]] = label_ 
        hv_infer[:hv_.shape[0],:hv_.shape[1]] = hv_

        transform = test_transform(img_size=self.image_size)
        
        try:
            after_aug = transform(image=image_infer, masks=[label_infer,hv_infer])
        except: 
            print("error")
            # print(self.img_path[index],label.shape,image.shape)
        image_tensor = after_aug['image']
        cell_tensor = after_aug['masks'][0]
        hv_tensor = after_aug['masks'][1]
        # print('img,label',image_tensor.shape,label_tensor.shape)
        label_tensor=cell_tensor.clone()
        label_tensor[cell_tensor>0]=1
        prompt_tensor = (label_tensor>0).long()
        _, pt = random_click(np.array(prompt_tensor),point_num=self.point_num)
        
        return {
            'image': image_tensor,
            'label': label_tensor,
            'hv_label': hv_tensor,
            'cell_label': cell_tensor,
            'p_label': torch.LongTensor([inouts]*self.point_num),
            'pt':pt,
            'image_meta_dict': {'filename_or_obj':img_p+f'_{Hi}_{Wi}'}
        }