import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import random
import torch
import joblib
import json
from skimage import io
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms.functional as F
from torch.utils.data import Dataset


class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, gt_root, trainsize, augmentations):
        self.trainsize = trainsize
        self.augmentations = augmentations
        print(self.augmentations)
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        if self.augmentations == True:
            print('Using RandomRotation, RandomFlip')
            self.img_transform = transforms.Compose([
                transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            self.gt_transform = transforms.Compose([
                transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
            
        else:
            print('no augmentation')
            self.img_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            
            self.gt_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
            

    def __getitem__(self, index):
        
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.img_transform is not None:
            image = self.img_transform(image)
            
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)
        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(image_size,mean_std_path, folds_path, fold_i, batchsize, num_workers=8, mode='Training', shuffle=True, pin_memory=True, augmentation=False):

    # dataset = PolypDataset(image_root, gt_root, trainsize, augmentation)
    # '/mnt/mydisk/zzf/code/Medical-SAM-Adapter-main/data'
    # '/home/zzf/dataset/IMC_Cell/Dice-XMBD/datas/melanoma/ch_dict.pkl'
    # '/mnt/mydisk/zzf/code/Medical-SAM-Adapter-main/data/Melanoma_instance/train_val_test.pkl'
    
    dataset = Melanomadataset(image_size=image_size,mean_std_path=mean_std_path,folds_path=folds_path,fold_i=fold_i,mode=mode,is_test=True)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    
    return data_loader


def img_transforms(img_size, ori_h, ori_w):
    transforms = []
    transforms.append(A.PadIfNeeded(min_height=round(img_size), min_width=round(img_size), border_mode=cv2.BORDER_CONSTANT, value=0))
    transforms.append(A.CenterCrop(height=img_size, width=img_size))
    return A.Compose(transforms, p=1.)
    
def train_transforms(img_size, ori_h, ori_w):
    transforms = []
    transforms.append(A.RandomCrop(img_size,img_size,p=1.0))
    transforms.append(A.HorizontalFlip(p=0.5))
    transforms.append(A.RandomRotate90(p=0.3))
    transforms.append(A.OneOf([
            A.GaussNoise(),
        ], p=0.2))
    transforms.append(ToTensorV2(p=1.0))
    return A.Compose(transforms, p=1.)

def test_transforms(img_size, ori_h, ori_w):

    transforms = []
    # transforms.append(A.PadIfNeeded(min_height=round(img_size), min_width=round(img_size), border_mode=cv2.BORDER_CONSTANT, value=0))
    transforms.append(A.CenterCrop(img_size,img_size,p=1.0))
    transforms.append(ToTensorV2(p=1.0))
    return A.Compose(transforms, p=1.)


class Melanomadataset(Dataset):
    def __init__(self, image_size, mean_std_path, folds_path, fold_i, patch_number='/mnt/mydisk/zzf/code/Medical-SAM-Adapter-main/data/Melanoma_instance/number_patch.pkl', mode = 'Training',prompt = 'click', plane = False, is_test=False):
        self.mean_std = joblib.load(mean_std_path)
        self_folds = joblib.load(folds_path)
        self.pixel_mean = np.array(self.mean_std['ch_mean'])
        self.pixel_std = np.array(self.mean_std['ch_std'])
        self.patch_number = joblib.load(patch_number)
        if mode == 'Training':
            self.image_paths = self_folds[f"train_{fold_i}"]
            self.label_paths = [im_name.replace("IMC_fullstacks/","IMC_segmentation/").replace("_fullstacks.tiff","_pred_Probabilities_cell_mask.tiff") for im_name in self.image_paths]
        if mode == 'Validing':
            self.image_paths = self_folds[f"val_{fold_i}"]
            self.label_paths = [im_name.replace("IMC_fullstacks/","IMC_segmentation/").replace("_fullstacks.tiff","_pred_Probabilities_cell_mask.tiff") for im_name in self.image_paths]
        if mode == 'Testing':
            self.image_paths = self_folds[f"test_{fold_i}"]
            self.label_paths = [im_name.replace("IMC_fullstacks/","IMC_segmentation/").replace("_fullstacks.tiff","_pred_Probabilities_cell_mask.tiff") for im_name in self.image_paths]

        self.img_tuple = {}
        now_i = 0
        for img_path in self.image_paths:
            for i in range(self.patch_number[img_path]):
                self.img_tuple[now_i+i] = (img_path,i)
            now_i += self.patch_number[img_path]
        
        self.mode = mode
        self.prompt = prompt
        self.image_size = image_size
        self.is_test=is_test
    
    def __len__(self,):
        return len(self.img_tuple)
    
    def get_patch_coords(self,h,w,patch_id):
        H = (h//224)+1 
        W = (w//224)+1 
        hi = patch_id//W 
        wi = patch_id%W 
        return (hi*224,(hi+1)*224,wi*224,(wi+1)*224)
    
    def __getitem__(self,index):
        idx = random.choice(list(np.arange(len(self.img_tuple))))
        image = io.imread(self.img_tuple[idx][0]).transpose(1,2,0)
        patch_id = self.img_tuple[idx][1]
        while image.shape[2]!=41:
            idx = random.choice(list(np.arange(len(self.img_tuple))))
            image = io.imread(self.img_tuple[idx][0]).transpose(1,2,0)
            patch_id = self.img_tuple[idx][1]
        image = (image - self.pixel_mean) / self.pixel_std
        image = image[:,:,3:40]

        h, w, _ = image.shape
        if not self.is_test:
            transforms = train_transforms(384, h, w)
        else:
            transforms = test_transforms(384, h, w)
        img_transforms_ = img_transforms(384, h, w)

        mask_path = self.img_tuple[idx][0].replace("IMC_fullstacks/","IMC_segmentation/").replace("_fullstacks.tiff","_pred_Probabilities_cell_mask.tiff")
        cell_type_mask = io.imread(mask_path.replace("IMC_segmentation/","IMC_seg_res_cell/")).astype(np.int16)
        pre_mask = io.imread(mask_path.replace("IMC_segmentation/","IMC_seg_cell_20/")).astype(np.int16)
        
        pre_mask = pre_mask[:h,:w]
        cell_type_mask = cell_type_mask[:h,:w]
        hs,he,ws,we = self.get_patch_coords(h,w,patch_id)
        # print(hs,he,ws,we)
        pre_aug = img_transforms_(image=image[hs:he,ws:we], masks=[pre_mask[hs:he,ws:we], cell_type_mask[hs:he,ws:we]])

        image_, pre_mask_, cell_type_mask_ = pre_aug['image'], pre_aug['masks'][0], pre_aug["masks"][1]
        augments = transforms(image=image_, masks=[pre_mask_, cell_type_mask_])
        image_tensor, mask_tensor, cell_type_tensor = augments['image'], augments['masks'][0].to(torch.int64), augments['masks'][1].to(torch.int64)
        
        # return image_tensor, cell_type_tensor,
        return {'img_t':image_tensor, 'target_t':mask_tensor, 'img_name':self.img_tuple[idx][0].split('/')[-1][:-5], 'hs':hs, 'ws':ws}

class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
