import cv2
import matplotlib.pyplot as plt
from random import shuffle
from ppnet.src.utils.helpers import *
from protoSiamese import gen_local_prototypes
from input_psiamese import getLocalProtosAreas

model = ppnet_model_loader("../models/ppnet_v38_18push74.56.pth")

train_dataset = train_push_loader_helper(False).dataset

print('Number of images in the dataset:', len(train_dataset))

# Be careful with RGB vs BGR order in OpenCV!
color_encodings = {
    
    # Mouth
    'u_lip': (0, 0, 255),
    'l_lip': (255, 255, 255),
    'mouth': (255, 255, 0),
    
    # Ears
    'r_ear': (0, 255, 0),
    'l_ear': (0, 128, 0),
    
    # Eyes
    'eye_g': (255, 0, 0),
    'r_eye': (0, 255, 255),
    'l_eye': (255, 0, 255),
    
    # Brows
    'l_brow': (0, 0, 128),
    'r_brow': (128, 0, 0),
    
    # Nose
    'nose': (0, 128, 128),
}
    
def get_facial_elements_pixel_count(proto_mask):
    
    # Setup the color usage dictionary
    facial_element_pixel_count = {'mouth':0, 'ears':0, 'eyes':0, 'brows':0, 'nose':0}
        
    # Count the number of pixels for each color in the mask
    for i in range(proto_mask.shape[0]):
        for j in range(proto_mask.shape[1]):
            pixel = tuple(proto_mask[i][j])
            
            # Skip the black pixels
            if pixel == (0, 0, 0):
                continue
            
            # Check if the pixel color matches any of the facial elements
            for facial_elem, color_triplet in color_encodings.items():
                if pixel == color_triplet:
                    if facial_elem in ['mouth', 'u_lip', 'l_lip']:
                        facial_element_pixel_count['mouth'] += 1
                    elif facial_elem in ['r_ear', 'l_ear']:
                        facial_element_pixel_count['ears'] += 1
                    elif facial_elem in ['eye_g', 'r_eye', 'l_eye']:
                        facial_element_pixel_count['eyes'] += 1
                    elif facial_elem in ['l_brow', 'r_brow']:
                        facial_element_pixel_count['brows'] += 1
                    elif facial_elem == 'nose':
                        facial_element_pixel_count['nose'] += 1
                    else:
                        print(f'Unknown facial element: {facial_elem}')
                    
                    
    return facial_element_pixel_count # {color: number of pixels with that color}

percentages = [] 

def evaluate_local_proto_pairs(proto_masksA, proto_masksB):
    
     # it will hold 10 dictionaries, one for each local prototype
    presenceA = []
    presenceB = []
    
    # Get the pixel counts for each local prototype and convert them to element presences
    for i in range(len(proto_masksA)):
        facial_element_pixel_count = get_facial_elements_pixel_count(proto_masksA[i]) # {'mouth': 0, 'ears': 0, 'eyes': 0, 'brows': 120, 'nose': 0}, ...
        for facial_element in facial_element_pixel_count.keys():
            if facial_element_pixel_count[facial_element]  > 0:
                facial_element_pixel_count[facial_element] = 1 # make it binary presence
                
        presenceA.append(facial_element_pixel_count)
        
    for i in range(len(proto_masksB)):
        facial_element_pixel_count = get_facial_elements_pixel_count(proto_masksB[i])
        for facial_element in facial_element_pixel_count.keys():
            if facial_element_pixel_count[facial_element]  > 0:
                facial_element_pixel_count[facial_element] = 1
        presenceB.append(facial_element_pixel_count)
        
        
    # Compute the accuracy for each local prototype
    accuracy_list = []

    for a, b in zip(presenceA, presenceB):
        total = len(a)
        matches = sum(1 for key in a if a[key] == b[key])
        accuracy = matches / total  # or multiply by 100 for percentage
        accuracy_list.append(accuracy)
        
    # Store
    percentages.append(accuracy_list)

    
def getLocalMaskAreas(mask, picked_cordsA):
    # mask: (3, 224, 224) — torch tensor
    # picked_cordsA: [(i, j), (i, j), ...] in 7x7 grid where 0,0 is top-left and 6,6 is bottom-right
    
    local_masks = []
    _, H, W = mask.shape
    
    grid_size = 7
    patch_H, patch_W = H // grid_size, W // grid_size

    for i, j in picked_cordsA:
        top = i * patch_H
        left = j * patch_W
        bottom = top + patch_H
        right = left + patch_W
        
        patch = mask[:, top:bottom, left:right] # [3, 32, 32]
        local_masks.append(patch)
    
    return local_masks  


def singleProtoMaskGetter(image, mask):
    
    # Run ppnet on the image and get the local prototypes
    image = image.to(MY_GPU_DEVICE_NAME).unsqueeze(0) # [1, 3, 224, 224]
    extracted_features, similarities, _ = model.forward(image)
    picked_cordinates, _ = gen_local_prototypes(extracted_features, similarities, model.num_prototypes)
    
    # Get the 32x32 patches of the image and mask
    proto_masks = getLocalMaskAreas(mask, picked_cordinates) # list of 10 masks of shapes [1, 3, 32, 32]
    # local_protos = getLocalProtosAreas(image, picked_cordinates) # list of 10 protos of shapes [1, 3, 32, 32]

    # For each local prototype, get the analogous mask in numpy [32, 32, 3]
    for j in range(len(proto_masks)):
        proto_masks[j] = proto_masks[j].cpu().numpy().transpose((1, 2, 0))
    
    return proto_masks
        
# ======================== Single Image Prototype Type Evaluation ========================
pairs = []
# get all non repeating pairs in the dataset
for i in range(len(train_dataset)):
    for j in range(i + 1, len(train_dataset)):
        pairs.append((i, j))
        
shuffle(pairs) # shuffle the pairs to get random ones
print('Number of distinct pairs:', len(pairs))

pairs = pairs[:3000] # for testing purposes

k = 0
for i, j in pairs:
    imageA, maskA, _ = train_dataset[i]
    imageB, maskB, _ = train_dataset[j]
    
    proto_masksA = singleProtoMaskGetter(imageA, maskA)
    proto_masksB = singleProtoMaskGetter(imageB, maskB)

    evaluate_local_proto_pairs(proto_masksA, proto_masksB)
    
    if k % 50 == 0 and k > 0:
        print(f'Processed {k} images \t\t Acc: {[round(sum(x) / len(x), 3) for x in zip(*percentages)]}')
    k += 1
        
print(len(percentages))

# aggregate the results by elem pos [(10 elems), (10 elems), ...] to get only 10 values
accs = [round(sum(x) / len(x), 3) for x in zip(*percentages)]
print(accs)

# GOT THESE RESULTS:
# [0.749, 0.784, 0.636, 0.827, 0.659, 0.997, 0.996, 0.816, 0.842, 0.635]