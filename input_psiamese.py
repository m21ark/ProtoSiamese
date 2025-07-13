import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
import io
import torch
import numpy as np
from sklearn.metrics import precision_recall_curve

from protoSiamese import protoSiameseLoader
from PSiameseDataset import preprocess_image
from ppnet.src.utils.helpers import overlay_heatmap_on_img
from PSiameseDataset import my_test_loader

model = protoSiameseLoader("../output/versions/psiamese_75.80.pth") 
model.set_should_train_ppnet(False)
model.eval()

model_weights = model.final_layer.weight.tolist()[0]
# model_bias = model.final_layer.bias.item()

sims = []
conf_right = []
conf_wrong = []
probs = []
truths = []
tn, fp, fn, tp = 0, 0, 0, 0

def update_conf(pred, truth):
    global tn, fp, fn, tp, conf_right, conf_wrong
    # Update confusion matrix
    if pred == truth:
        conf_right.append(pred)
        if pred == 1:
            tp += 1
        else:
            tn += 1
    else:
        conf_wrong.append(pred)
        if pred == 1:
            fp += 1
        else:
            fn += 1

# ==================================== AUX FUNCS ====================================

def getImagesFilesInFolder(folder_path):
    image_paths = []
    valid_extensions = ('.png', '.jpg', '.jpeg')  # you can add more if needed
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(valid_extensions):
                image_paths.append(os.path.join(root, file))
    
    return image_paths

def plot_similarity_grid(image_paths, sims):

    # make a plt grid showing the images and their similarity scores
    fig, axes = plt.subplots(len(image_paths), len(image_paths), figsize=(10, 10))
    for i in range(len(image_paths)):
        for j in range(len(image_paths)):
            if i == j:
                img = Image.open(image_paths[i])
                axes[i, j].imshow(img)
                axes[i, j].axis('off')
                # add path as title
                file_name = f"{image_paths[i].split('/')[-2]}/{image_paths[i].split('/')[-1]}"
                axes[i, j].set_title(file_name, fontsize=10)
            
            else:
                axes[i, j].axis('off')
                sim = next((sim for sim in sims if sim[0] == i and sim[1] == j), None)
                if sim:
                    # Your plotting code with text color adjustment
                    color = "#AA0000" if sim[2] < 0.5 else "#00AA00"  # Set color based on similarity score

                    axes[i, j].text(0.5, 0.5, f"{sim[2]:.2f}", fontsize=12, ha='center', va='center', 
                                    color="white", fontweight='bold', bbox=dict(facecolor=color, edgecolor='white', boxstyle='square,pad=1'))
    plt.tight_layout()
    plt.savefig("similarity_grid.png", dpi=300)
    
def create_chosen_in_grid(heatmap_data):
    # Initialize a 7x7 grid with zeros
    board = np.zeros((7, 7))

    # Populate the grid with activation counts
    activation_counts = {}
    for r, c in heatmap_data:
        row_index = r
        col_index = c
        if 0 <= row_index < 7 and 0 <= col_index < 7:
            board[row_index, col_index] += 1
            key = (r, c)
            activation_counts[key] = activation_counts.get(key, 0) + 1
        else:
            print(f"Warning: Coordinates ({r}, {c}) are out of bounds for the 7x7 grid.")

    # Create the heatmap visualization
    fig, ax = plt.subplots(figsize=(20, 20))  # Set a square figure size
    im = ax.imshow(board, cmap='Greys', extent=[0, 7, 0, 7])

    # Set the x and y axis ticks and labels to be at the borders
    ax.set_xticks(np.arange(0, 7, 1))
    ax.set_xticklabels(np.arange(1, 8))
    ax.set_yticks(np.arange(0, 7, 1))
    ax.set_yticklabels(np.arange(1, 8))

    # Remove axis spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Add the number of activations as text labels inside each square
    for i in range(7):
        for j in range(7):
            activation = int(board[i, j])
            if activation > 0:
                # Flip the row index since image is displayed top-down
                ax.text(j + 0.5, 6 - i + 0.5, str(activation), ha='center', va='center', color='white', fontsize=120, fontweight='bold')

    # Ensure tight layout to remove padding
    plt.tight_layout()

    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)  # Close the figure explicitly
    return img

def visualizeExplanation(a, b, sim_pairs, pred, truth):
    
    (image1, simsA, picked_mapA, local_protosA) = a
    (image2, simsB, picked_mapB, local_protosB) = b

    image1_np = image1[0].cpu().numpy().transpose((1, 2, 0))
    image2_np = image2[0].cpu().numpy().transpose((1, 2, 0))
    num_protos = simsA.shape[1]

    fig, axes = plt.subplots(4, num_protos + 1, figsize=(15, 7))

    for i in range(num_protos):
        # Heatmap for the first pair
        heatmap1 = simsA[0, i].cpu().numpy()
        heatmap1_resized = cv2.resize(heatmap1, (image1_np.shape[1], image1_np.shape[0]), interpolation=cv2.INTER_CUBIC)
        overlayed1 = overlay_heatmap_on_img(image1_np, heatmap1_resized)
        axes[0, i].imshow(overlayed1)
        axes[0, i].axis('off')
        axes[0, i].set_title(f"Proto {i+1}")
        
        # Local prototypes for the first pair
        axes[1, i].imshow(local_protosA[i][0].cpu().numpy().transpose((1, 2, 0)))
        axes[1, i].axis('off')
        color = "red" if model_weights[i] < 0 else "green"
        axes[1, i].set_title(f"Weight: {model_weights[i]:.2f}", color=color)
        
        # Local prototypes for the second pair
        axes[2, i].imshow(local_protosB[i][0].cpu().numpy().transpose((1, 2, 0)))
        axes[2, i].axis('off')
        axes[2, i].set_title(f"Sim: {sim_pairs[0, i].item():.2f}")
        
        # Heatmap for the second pair
        heatmap2 = simsB[0, i].cpu().numpy()
        heatmap2_resized = cv2.resize(heatmap2, (image2_np.shape[1], image2_np.shape[0]), interpolation=cv2.INTER_CUBIC)
        overlayed2 = overlay_heatmap_on_img(image2_np, heatmap2_resized)
        axes[3, i].imshow(overlayed2)
        axes[3, i].axis('off')
        
    # Add in the last column the picked features
    axes[0, num_protos].imshow(picked_mapA)
    axes[0, num_protos].axis('off')
    axes[0, num_protos].set_title("Picked Features")
    axes[3, num_protos].imshow(picked_mapB)
    axes[3, num_protos].axis('off')
    
    axes[1, num_protos].axis('off')
    axes[2, num_protos].axis('off')
    color = "green" if pred == truth else "red"
    axes[2, num_protos].set_title(f"Pred: {pred}\nTruth: {truth}", color=color)
    
    plt.tight_layout()
    plt.show()

def getLocalProtosAreas(image, picked_cordsA):
    # image: (1, 3, 224, 224) — torch tensor
    # picked_cordsA: [(i, j), (i, j), ...] in 7x7 grid where 0,0 is top-left and 6,6 is bottom-right
    
    local_protos = []
    _, _, H, W = image.shape
    grid_size = 7
    patch_H, patch_W = H // grid_size, W // grid_size

    for i, j in picked_cordsA:
        top = i * patch_H
        left = j * patch_W
        bottom = top + patch_H
        right = left + patch_W
        
        patch = image[:, :, top:bottom, left:right]  # shape: (1, 3, 32, 32)
        local_protos.append(patch)
    
    return local_protos

# ==================================== TEST ====================================

THRESHOLD_POINT = 0.5

def process_pair(path1, path2, verbose=False):

    image1 = preprocess_image(path1, True).to("cuda" if torch.cuda.is_available() else "mps")
    image2 = preprocess_image(path2, True).to("cuda" if torch.cuda.is_available() else "mps")

    logits, extracted_features, simsA, simsB, sim_pairs, picked_cordsA, picked_cordsB = model(image1, image2)
    conf = torch.sigmoid(logits).item()
    probs.append(conf)
    
    # Compare prediction with ground truth
    trueAclass = path1.split("/")[-2]
    trueBclass = path2.split("/")[-2]
    pred = 1 if conf > THRESHOLD_POINT else 0
    truth = 1 if trueAclass == trueBclass else 0
    truths.append(truth)
    
    # See internal logic
    if verbose:
        picked_mapA = create_chosen_in_grid(picked_cordsA)
        picked_mapB = create_chosen_in_grid(picked_cordsB)
        
        no_norm_img1 = preprocess_image(path1, False)  # [1, 3, 224, 224]
        no_norm_img2 = preprocess_image(path2, False)

        local_protosA = getLocalProtosAreas(no_norm_img1, picked_cordsA)
        local_protosB = getLocalProtosAreas(no_norm_img2, picked_cordsB)
                
        a = (no_norm_img1, simsA, picked_mapA, local_protosA)
        b = (no_norm_img2, simsB, picked_mapB, local_protosB)
        
        visualizeExplanation(a, b, sim_pairs, pred, truth)
        
    return pred, truth, conf


if __name__ == "__main__":
    
    # ===================================== SINGLE ====================================

    path1 = "../data/examples/amy_2.png"
    path2 = "../data/examples/Mayim_Bialik.jpg"

    test_loader = my_test_loader(use_shuffle=True, normalized=True)

    pairs = []
    for i in range(0,32):
        pairs.append(test_loader.dataset[i])
    np.random.shuffle(pairs)

    for i in range(0,32):
        path1, path2, label = pairs[i]
        print(pairs[i])
        process_pair(path1, path2, verbose=True)
    exit(0)

    # ===================================== FOLDER ====================================

    image_paths = getImagesFilesInFolder("../data/examples/")# [:100] # THE TRUTH CLASSES DONT WORK ON ../data/examples/ 
    num_images = len(image_paths)  
    if num_images < 2:
        raise ValueError(f"Please provide at least two images for comparison and got: {num_images}")

    # get the upper triangle of pairs
    for i in range(num_images):
        for j in range(i + 1, num_images):
            pred, truth, conf = process_pair(image_paths[i], image_paths[j])
            update_conf(pred, truth)
            sims.append((i, j, conf))
    
    # plot the similarity grid     
    if num_images < 20:
        plot_similarity_grid(image_paths, sims)
        print("Similarity grid saved as 'similarity_grid.png'")
                
    print(f"Confusion matrix ({THRESHOLD_POINT}):")
    print(f"{'':<10}{'Pred 0':<15}{'Pred 1':<15}")
    print(f"{'Actual 0':<10}{tn:<15}{fp:<15}")
    print(f"{'Actual 1':<10}{fn:<15}{tp:<15}")

    print("\nAccuracy: {:.2f}".format((tp + tn) / (tp + tn + fp + fn)))
    print("Precision: {:.2f}".format(tp / (tp + fp) if (tp + fp) > 0 else 0))
    print("Recall: {:.2f}".format(tp / (tp + fn) if (tp + fn) > 0 else 0))
    print("F1: {:.2f}".format(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0))

    prec, rec, thresholds = precision_recall_curve(truths, probs)

    plt.figure(figsize=(10, 6))
    plt.plot(prec, rec, marker='o', linestyle='-', color='b')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision-Recall Curve')
    plt.grid()
    plt.savefig("precision_recall_curve.png", dpi=300)