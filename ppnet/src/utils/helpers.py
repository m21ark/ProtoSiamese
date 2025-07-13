import copy
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import ppnet.src.ppnet as ppnet
from ppnet.src.utils.settings import *

MY_GPU_DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'mps'
    
my_mean = (0.485, 0.456, 0.406)
my_std = (0.229, 0.224, 0.225)
my_normalize = transforms.Normalize(mean=my_mean, std=my_std)

def preprocess(x, mean, std):
    assert x.size(1) == 3
    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = (x[:, i, :, :] - mean[i]) / std[i]
    return y

def extract_patch(img, corners, offset = 0):
    return img[corners[0 + offset]:corners[1 + offset], corners[2 + offset]:corners[3 + offset], :]

def preprocess_input_function(x):
    '''
    allocate new tensor like x and apply the normalization used in the
    pretrained model
    '''
    return preprocess(x, mean=my_mean, std=my_std)

def overlay_heatmap_on_img(img, heatmap):
    rescaled_heatmap = heatmap - np.amin(heatmap)
    rescaled_heatmap = rescaled_heatmap / np.amax(rescaled_heatmap)
    heatmap = cv2.applyColorMap(np.uint8(255*rescaled_heatmap), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[...,::-1]
    overlayed_img = 0.5 * img + 0.3 * heatmap
    return overlayed_img

def ppnet_model_loader(model_path):
    my_model = ppnet.construct_PPNet()
    my_model.load_state_dict(torch.load(model_path, map_location=torch.device(MY_GPU_DEVICE_NAME), weights_only=False))
    my_model = my_model.to(MY_GPU_DEVICE_NAME)
    while type(my_model) == torch.nn.DataParallel:
        my_model = my_model.module
    return my_model

def save_code_state_ppnet(save_path):
    makedir(save_path)
    cwd = os.getcwd()
    
    for py_file in glob.glob("*.py"):
        shutil.copy(py_file, save_path)
        
    for root, dirs, files in os.walk(cwd + '/src'):
        for file in files:
            if file.endswith(".py"):
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, cwd) 
                dest_path = os.path.join(save_path, rel_path)

                os.makedirs(os.path.dirname(dest_path), exist_ok=True) 
                shutil.copy2(src_path, dest_path)
                
    # zip the code folder
    shutil.make_archive(save_path + "/../code", 'zip', save_path)
    shutil.rmtree(save_path)
    
def save_code_state_psiamese(save_path):
    makedir(save_path)
    cwd = os.getcwd()
    
    for py_file in glob.glob("*.py"):
        shutil.copy(py_file, save_path)
        
    for root, dirs, files in os.walk(cwd + '/ppnet'):
        for file in files:
            if file.endswith(".py"):
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, cwd) 
                dest_path = os.path.join(save_path, rel_path)

                os.makedirs(os.path.dirname(dest_path), exist_ok=True) 
                shutil.copy2(src_path, dest_path)
                
    # zip the code folder
    shutil.make_archive(save_path + "/../code", 'zip', save_path)
    shutil.rmtree(save_path)


def make_one_hot(target, target_one_hot):
    target = target.view(-1,1)
    target_one_hot.zero_()
    target_one_hot.scatter_(dim=1, index=target, value=1.)


def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)


def print_and_write(str, file):
    print(str)
    file.write(str + '\n')


def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:,j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:,j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y+1, lower_x, upper_x+1


def create_logger(log_filename, display=True):
    f = open(log_filename, 'a')
    counter = [0]
    # this function will still have access to f after create_logger terminates
    def logger(text):
        if display:
            print(text)
        f.write(text + '\n')
        counter[0] += 1
        if counter[0] % 10 == 0:
            f.flush()
            os.fsync(f.fileno())
    return logger, f.close


# ==================================== PLOTTING HELPER FUNCS ====================================

def undo_preprocess_img(processed_img):
        
    def undo_preprocess(x, mean, std):
        assert x.size(1) == 3
        y = torch.zeros_like(x)
        for i in range(3):
            y[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
        return y

    img_copy = copy.deepcopy(processed_img)
    img = undo_preprocess(img_copy, my_mean, my_std)
    img = img[0]
    img = img.detach().cpu().numpy()
    img = np.transpose(img, [1,2,0])
    return img


def imsave_with_bbox(fname, img_rgb, bb_dims):
    
    bbox_height_start, bbox_height_end, bbox_width_start, bbox_width_end = bb_dims 
    
    img_bgr_uint8 = cv2.cvtColor(np.uint8(255*img_rgb), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1), (0, 255, 255), thickness=2)
    img_rgb_uint8 = img_bgr_uint8[...,::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    plt.imsave(fname, img_rgb_float)


def save_og_prototype_withBB(load_img_dir, fname, epoch, index, bb_dims):
    
    bbox_height_start, bbox_height_end, bbox_width_start, bbox_width_end = bb_dims
                                        
    p_img_bgr = cv2.imread(os.path.join(load_img_dir, f'epoch-{epoch}', f'prototype-img-original{index}.png'))
    
    pos_start = (bbox_width_start, bbox_height_start)
    pos_end = (bbox_width_end-1, bbox_height_end-1)
    cv2.rectangle(p_img_bgr, pos_start, pos_end, (0, 255, 255), thickness=2)
    
    p_img_rgb = p_img_bgr[...,::-1]
    p_img_rgb = np.float32(p_img_rgb) / 255

    plt.imsave(fname, p_img_rgb)
    
# ==================================== DATASET LOADERS ====================================

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # List of all subfolders (representing classes) in the image directory
        self.class_names = sorted(os.listdir(image_dir))  # Folders are the classes
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}

        # List of image and mask files, with associated class labels
        self.images = []
        self.masks = []
        self.labels = []

        # Traverse each class folder
        for class_name in self.class_names:
            
            if class_name == '.DS_Store':
                continue
            
            class_folder_image = os.path.join(image_dir, class_name)
            class_folder_mask = os.path.join(mask_dir, class_name)

            # Collect images and masks inside each class folder
            image_files = sorted(os.listdir(class_folder_image))
            mask_files = sorted(os.listdir(class_folder_mask))

            for img_file, mask_file in zip(image_files, mask_files):
                image_path = os.path.join(class_folder_image, img_file)
                mask_path = os.path.join(class_folder_mask, mask_file)

                # Store paths and associated class index
                self.images.append(image_path)
                self.masks.append(mask_path)
                self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load the image, mask, and class label
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')
        
        # during mask augmentation, there are some interpolated pixels 
        # that are not 0, 128 or 255. We need to set them to 0
        if self.mask_dir == my_mask_train_aug_dir:
            mask = np.array(mask)
            mask[(mask != 0) & (mask != 128) & (mask != 255)] = 0
            mask = torch.tensor(mask)
        else:
            mask = torch.tensor(np.array(mask))
            
        mask = mask.permute(2, 0, 1) # [3, 224, 224]
        
        # Apply transformations (only normalize + to_tensor) only to the image and not the mask
        if self.transform:
            image = self.transform(image)

        return image, mask, label

def train_push_loader_helper(use_shuffle):

    transforms_train = transforms.Compose([
                transforms.Resize(size=(my_img_size, my_img_size)),
                transforms.ToTensor(),
                # No normalization !
    ])
        
    dataset = SegmentationDataset(my_img_train_push_dir, my_mask_train_push_dir, transform=transforms_train)
    return DataLoader(dataset, batch_size=my_batch_size, shuffle=use_shuffle, num_workers=4)
    
def train_aug_loader_helper(use_shuffle):
    
    transforms_train = transforms.Compose([
                transforms.Resize(size=(my_img_size, my_img_size)),
                transforms.ToTensor(),
                my_normalize,
    ])
        
    dataset = SegmentationDataset(my_img_train_aug_dir, my_mask_train_aug_dir, transform=transforms_train)
    return DataLoader(dataset, batch_size=my_batch_size, shuffle=use_shuffle, num_workers=4)
    
def test_loader_helper(use_shuffle, normalized):
    
    test_transforms = transforms.Compose([
        transforms.Resize(size=(my_img_size, my_img_size)),
        transforms.ToTensor(),
        my_normalize if normalized else transforms.Lambda(lambda x: x),
    ])
    
    dataset = SegmentationDataset(my_img_test_dir, my_mask_test_dir, transform=test_transforms)
    return DataLoader(dataset, batch_size=my_batch_size, shuffle=use_shuffle, num_workers=4)
    