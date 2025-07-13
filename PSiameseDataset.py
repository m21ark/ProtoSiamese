from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import random
import os

from ppnet.src.utils.settings import *


my_mean = (0.485, 0.456, 0.406)
my_std = (0.229, 0.224, 0.225)
my_normalize = transforms.Normalize(mean=my_mean, std=my_std)


class MyDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        # List of all subfolders (representing classes) in the image directory
        self.class_names = sorted(os.listdir(image_dir))  # Folders are the classes
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}

        # List of image and mask files, with associated class labels
        self.units = {} # [class_index: (image_path, mask_path), ...]

        # Traverse each class folder
        for class_name in self.class_names:
            
            if class_name == '.DS_Store':
                continue
            
            class_folder_image = os.path.join(image_dir, class_name)

            # Collect images and masks inside each class folder
            image_files = sorted(os.listdir(class_folder_image))

            for img_file in image_files:
                image_path = os.path.join(class_folder_image, img_file)

                # Store paths and associated class index
                key = self.class_to_idx[class_name]
                if key not in self.units:
                    self.units[key] = []
                self.units[key].append(image_path)
                
        self._process_units()

    def __len__(self):
        return len(self.pairs)

    def _process_units(self):
        self.pairs = []
        
        for key in self.units.keys():
            for anchor_img_path in self.units[key]:
                
                # choose if the pair will be positive or negative and get the corresponding image
                label = random.choice([0, 1])
                
                if label == 1:
                    # positive pair: get the same class but different image
                    other_img_path = random.choice([i for i in self.units[key] if i != anchor_img_path])
                    
                else:
                    # negative pair
                    other_classes = [i for i in list(self.units.keys()) if i != key]
                    other_img_path = random.choice(self.units[random.choice(other_classes)])
                    
                assert anchor_img_path != other_img_path, "Anchor and other image paths should not be the same"

                # append the pair
                self.pairs.append((anchor_img_path, other_img_path, label))
    

    def __getitem__(self, idx):
        
        anchor_img_path, other_img_path, label = self.pairs[idx]

        anchor_img = Image.open(anchor_img_path).convert('RGB')
        other_img = Image.open(other_img_path).convert('RGB')
        
        # Apply transformations (only normalize + to_tensor) only to the image and not the mask
        if self.transform:
            anchor_img = self.transform(anchor_img)
            other_img = self.transform(other_img)
            
        return anchor_img, other_img, label # anchor_img_path, other_img_path, label # Uncomment to return paths instead of images for evaluation
        
def my_train_loader(use_shuffle, augmented=True):
    
    transforms_train = transforms.Compose([
                transforms.Resize(size=(my_img_size, my_img_size)),
                transforms.ToTensor(),
                my_normalize,
    ])
        
    dataset = MyDataset(my_img_train_aug_dir if augmented else my_img_train_push_dir, transform=transforms_train)
    return DataLoader(dataset, batch_size=my_batch_size, shuffle=use_shuffle, num_workers=4)
    
def my_test_loader(use_shuffle, normalized):
    
    test_transforms = transforms.Compose([
        transforms.Resize(size=(my_img_size, my_img_size)),
        transforms.ToTensor(),
        my_normalize if normalized else transforms.Lambda(lambda x: x),
    ])
    
    dataset = MyDataset(my_img_test_dir, transform=test_transforms)
    return DataLoader(dataset, batch_size=my_batch_size, shuffle=use_shuffle, num_workers=4)
    
    
def preprocess_image(input_image, useNormalize = True):

    image = Image.open(input_image).convert('RGB')
        
    transform = transforms.Compose([
        transforms.Resize(size=(my_img_size, my_img_size)),
        transforms.ToTensor(),
        my_normalize if useNormalize else transforms.Lambda(lambda x: x),
    ])

    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension