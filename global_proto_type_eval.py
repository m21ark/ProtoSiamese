from calendar import c
import cv2
import matplotlib.pyplot as plt

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

color_activations = [{} for i in range(model.num_prototypes)] 
percentages = [[] for i in range(model.num_prototypes)] 
num_dead = [0 for i in range(model.num_prototypes)]

def evaluate_local_proto(local_protos, proto_masks):

    total_mask_pixels = proto_masks[0].shape[0] * proto_masks[0].shape[1] # 32x32 = 1024
    five_percent_mask_pixels = int(total_mask_pixels * 0.05) # 5% of the mask pixels
    
    for proto_index in range(len(local_protos)):
        # get the color usage
        facial_element_pixel_count = get_facial_elements_pixel_count(proto_masks[proto_index])

        # if the mask is all black, count it as a dead prototype
        if sum(facial_element_pixel_count.values()) == 0:
            num_dead[proto_index] += 1
            continue
        
        # store the current image's color usage info for later graphing
        for facial_element in facial_element_pixel_count.keys():
            if not facial_element in color_activations[proto_index]:
                color_activations[proto_index][facial_element] = 0
            
            # if the color is used, count it only if it represents more than 5% of the mask
            if facial_element_pixel_count[facial_element] >= five_percent_mask_pixels:
                color_activations[proto_index][facial_element] += facial_element_pixel_count[facial_element]
            
        # get the percentage of the mask that is activated (different form 0,0,0)
        gray_mask = cv2.cvtColor(proto_masks[proto_index], cv2.COLOR_BGR2GRAY)
        activated_pixels = cv2.countNonZero(gray_mask)

        percentage = (activated_pixels / total_mask_pixels) * 100
        percentages[proto_index].append(percentage)
        
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
        
# ======================== Single Image Prototype Type Evaluation ========================
    
for i in range(len(train_dataset)):
# for i in range(200):
    image, mask, _ = train_dataset[i]
    
    # Run ppnet on the image and get the local prototypes
    image = image.to(MY_GPU_DEVICE_NAME).unsqueeze(0) # [1, 3, 224, 224]
    extracted_features, similarities, _ = model.forward(image)
    picked_cordinates, _ = gen_local_prototypes(extracted_features, similarities, model.num_prototypes)
    
    # Get the 32x32 patches of the image and mask
    proto_masks = getLocalMaskAreas(mask, picked_cordinates) # list of 10 masks of shapes [1, 3, 32, 32]
    local_protos = getLocalProtosAreas(image, picked_cordinates) # list of 10 protos of shapes [1, 3, 32, 32]
    
    # Revert the images to numpy format
    image = image.cpu().numpy().squeeze(0).transpose((1, 2, 0)) # [224, 224, 3]
    for j in range(len(local_protos)):
        local_proto = local_protos[j][0] # [3, 32, 32]
        local_protos[j] = local_proto.cpu().numpy().transpose((1, 2, 0)) # [32, 32, 3]
    
    # For each local prototype, get the analogous mask in numpy [32, 32, 3]
    for j in range(len(proto_masks)):
        proto_masks[j] = proto_masks[j].cpu().numpy().transpose((1, 2, 0))
    
    evaluate_local_proto(local_protos, proto_masks)
    
    if i % 50 == 0 and i > 0:
        print(f'Processed {i} images  ({round(i / len(train_dataset) * 100, 2)}%)')


percentages = [round(sum(p) / len(p), 0) for p in percentages]

# convert num_dead to percentage
num_dead = [round((n / len(train_dataset)) * 100, 2) for n in num_dead]

labels_all = [
    'mouth', 'ears', 'eyes', 'brows', 'nose'
]

cmap = plt.colormaps['tab10']
label_colors = {label: cmap(i) for i, label in enumerate(labels_all)}

# Create a figure with 2 rows and 5 columns of subplots
fig, axes = plt.subplots(2, 5, figsize=(20, 8))

plt.subplots_adjust(wspace=0.0, hspace=0.2)  # Adjust spacing between columns and rows
axes = axes.flatten()

for i, ax in enumerate(axes):
    filtered_items = [(label, value) for label, value in color_activations[i].items() if value > 0]
    
    # if labels have less than 1% of activation, hide them
    total_values = sum(value for _, value in filtered_items)
    print(f'Prototype {i+1} : {[(label, value) for label, value in color_activations[i].items()]}')
    filtered_items = [(label, value) for label, value in filtered_items if value / total_values >= 0.02]

    if not filtered_items:
        ax.pie([1], colors=['black'])
        ax.set_title(f'Prototype {i+1}\n(All Zeros)\nActiv. Pixel Ratio: {percentages[i]}%\nDead: {num_dead[i]}%', fontsize=12)
    else:
        labels, sizes = zip(*filtered_items)
        colors = [label_colors.get(label, (0.7, 0.7, 0.7)) for label in labels]  # default gray if label missing
        max_label = labels[sizes.index(max(sizes))]
        ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.0f%%',
            textprops={'fontsize': 12, 'color': 'black'},  # Changed color to black
            startangle=140,
            wedgeprops={'edgecolor': 'black'}
        )
        ax.set_title(f'Prototype {i+1}\nActiv. Pixel Ratio: {percentages[i]}%\nDead: {num_dead[i]}%', fontsize=12)

for i, ax in enumerate(axes):
    filtered_items = [(label, value) for label, value in color_activations[i].items() if value > 0]
    
    # if labels have less than 1% of activation, hide them
    total_values = sum(value for _, value in filtered_items)
    filtered_items = [(label, value) for label, value in filtered_items if value / total_values >= 0.02]

    if not filtered_items:
        ax.pie([1], colors=['black'])
        ax.set_title(f'Prototype {i+1}\n(All Zeros)\nActiv. Pixel Ratio: {percentages[i]}%\nDead: {num_dead[i]}%', fontsize=12)
    else:
        labels, sizes = zip(*filtered_items)
        colors = [label_colors.get(label, (0.7, 0.7, 0.7)) for label in labels]  # default gray if label missing
        max_label = labels[sizes.index(max(sizes))]
        ax.pie(
            sizes,
            colors=colors,
            autopct='%1.0f%%',
            textprops={'fontsize': 12, 'color': 'white', 'weight': 'bold'},  # Changed color to black
            startangle=140,
            wedgeprops={'edgecolor': 'black'}
        )
        ax.set_title(f'Prototype {i+1}\nActiv. Pixel Ratio: {percentages[i]}%\nDead: {num_dead[i]}%', fontsize=12)


plt.savefig('global_proto_eval.png', dpi=1000, bbox_inches='tight')