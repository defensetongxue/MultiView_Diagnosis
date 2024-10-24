
import os
from torchvision import  transforms
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image
import json
import torch

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
class CustomDataset(Dataset):
    def __init__(self, split,data_path,split_name,resize=224,norm_method='custom',enhanced=False,bin=False):
        '''
        as the retfound model is pretrained in the image_net norm(mean,std),
        we keep the mean and std for this method, but for the other model, 
        we use the std,mean cal by 20 images of custom dataset
        '''
        with open(os.path.join(data_path,'split',f'{split_name}.json'), 'r') as f:
            self.split_list=json.load(f)[split]
        with open(os.path.join(data_path,'annotations.json'),'r') as f:
            self.data_dict=json.load(f)
        
        self.preprocess=transforms.Compose([
            # CropPadding(),
            transforms.Resize((resize,resize))
        ])
        self.enhance_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # as the rop  lesion is slender which will be distortion when rotate
            # we only rotate the 90 180 270 degree for augument
            Fix_RandomRotation(),
        ])
        self.img_enhanced=enhanced
        self.split = split
        if norm_method=='imagenet':
            self.img_transforms=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD)])
        elif norm_method=='custom':
            self.img_transforms=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4623, 0.3856, 0.2822],
                std=[0.2527, 0.1889, 0.1334])])
        else:
            raise
        self.bin=bin 
    def __getitem__(self, idx):
        image_name = self.split_list[idx]
        data=self.data_dict[image_name]
        if self.img_enhanced:
            img=Image.open(data['enhanced_path']).convert("RGB")
        else:
            img=Image.open(data['image_path']).convert("RGB")
        img=self.preprocess(img)
        if self.split=='train':
            img=self.enhance_transforms(img)
            
        img=self.img_transforms(img)
        if self.bin:
            ridge_label = 1 if data['stage']>0 else 0
        else:
            ridge_label = data["stage"]
        return img,ridge_label,image_name


    def __len__(self):
        return len(self.split_list)
    
class Fix_RandomRotation(object):

    def __init__(self, degrees=360, resample=False, expand=False, center=None):
        self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params():
        p = torch.rand(1)

        if p >= 0 and p < 0.25:
            angle = -180
        elif p >= 0.25 and p < 0.5:
            angle = -90
        elif p >= 0.5 and p < 0.75:
            angle = 90
        else:
            angle = 0
        return angle

    def __call__(self, img):
        angle = self.get_params()
        return F.rotate(img, angle, F.InterpolationMode.NEAREST , self.expand, self.center)

    def __repr__(self):
        format_string = self.__class__.__name__ + \
            '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string
    
class CropPadding:
    def __init__(self, box=(80, 0, 1570, 1200)):
        self.box = box

    def __call__(self, img):
        return img.crop(self.box)