import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time

from ppnet.src.utils.helpers import *

def _push_save_prototypes(save_proto_dir, original_img_j, upsampled_act_img_j, proto_img_j, j, k):
    
    # save the whole image containing the prototype as png
    file_name = os.path.join(save_proto_dir, f'source-{j}-{k}.png')
    plt.imsave(file_name, original_img_j, vmin=0.0, vmax=1.0)
    
    # overlay (upsampled) self activation on original image and save the result
    overlayed_original_img_j = overlay_heatmap_on_img(original_img_j, upsampled_act_img_j)
    
    file_name = os.path.join(save_proto_dir, f'heatmap-{j}-{k}.png')
    plt.imsave(file_name, overlayed_original_img_j, vmin=0.0, vmax=1.0)
    
    # save the prototype image (highly activated region of the whole image)
    file_name = os.path.join(save_proto_dir, f'prototype-{j}-{k}.png')
    plt.imsave(file_name, proto_img_j, vmin=0.0, vmax=1.0)
    

# update each prototype for current search batch
def _update_prototypes_on_batch(search_batch_input, multi_ppnet, save_proto_dir, preprocess_func, topk_seen_proto_sim, best_proto_match_fmap, top_k):
    
    # ================================================ SETUP ================================================

    multi_ppnet.eval()
    model = multi_ppnet.module
    n_prototypes = my_prototype_shape[0]

    if preprocess_func is not None:
        search_batch = preprocess_func(search_batch_input)
    else:
        search_batch = search_batch_input

    with torch.no_grad():
        search_batch = search_batch.to(MY_GPU_DEVICE_NAME)
        extracted_features, proto_sims, _ = model.forward(search_batch)

    extracted_features = np.copy(extracted_features.detach().cpu().numpy())
    proto_sims = np.copy(proto_sims.detach().cpu().numpy())

    # ================================================ FIND PROTOTYPES ================================================

    # for each prototype, scan through all patches in the given batch to 
    # find the closest patch to blit the prototype into it as an image
    for j in range(n_prototypes):
        
        proto_sim_j = proto_sims[:,j,:,:] # 7x7, 1 channel - similarity of each patch to prototype j
        
        # ================================================================================================

        best_patch_j = np.amax(proto_sim_j) # max similarity of all patches to prototype j
        
        # if the current prototype is not the in the top-k match yet, skip 
        # Else, update the lowest of the top-k similarity seen so far and continue to push the prototype to the closest patch
        # Identify that position in the top-k as k
    
        # K is the worst of the top-k seen so far
        k = np.argmin(topk_seen_proto_sim[j])
        
        # if the current patch is not better than the worst of the top-k seen so far, skip
        if best_patch_j <= topk_seen_proto_sim[j][k]:
            continue
        
        # Else, replace the worst of the top-k seen so far with the current best patch similarity
        topk_seen_proto_sim[j][k] = best_patch_j
        
        # ================================================================================================
        
        # get the tensor cordinates of the patch that generates the representation closest to the prototype
        max_sim_patch_cords_j = list(np.unravel_index(np.argmax(proto_sim_j, axis=None), proto_sim_j.shape))

        # retrieve the corresponding patch info from feature map 
        img_index_in_batch, fmap_height_start_idx, fmap_width_start_idx = max_sim_patch_cords_j[0:3]
        
        proto_h = my_prototype_shape[2]
        proto_w = my_prototype_shape[3]

        fmap_height_end_idx = fmap_height_start_idx + proto_h
        fmap_width_end_idx = fmap_width_start_idx + proto_w

        # get the closest patch feature representation with all channels
        batch_best_patch_j = extracted_features[img_index_in_batch,:, fmap_height_start_idx:fmap_height_end_idx, fmap_width_start_idx:fmap_width_end_idx]

        # set the best patch representation for the current prototype
        best_proto_match_fmap[j, :, :, :, k] = batch_best_patch_j
        
        # ================================================================================================
        
        # get the whole original image that contains the patch closest to the prototype
        original_img_j = search_batch_input[img_index_in_batch].numpy()
        original_img_j = np.transpose(original_img_j, (1, 2, 0))
        og_image_size = original_img_j.shape[0]
        
        # ================================================================================================

        # find the highly activated similarity region on the original image
        proto_act_img_j = proto_sims[img_index_in_batch, j, :, :]
         
        # upsample the activation map to the original image size
        upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(og_image_size, og_image_size), interpolation=cv2.INTER_CUBIC)
        proto_bound_j = find_high_activation_crop(upsampled_act_img_j)
        
        # crop out the image patch with high activation as the final prototype image
        proto_img_j = extract_patch(original_img_j, proto_bound_j)

        # ================================================ SAVE PROTOTYPES ================================================

        _push_save_prototypes(save_proto_dir, original_img_j, upsampled_act_img_j, proto_img_j, j, k)
        
    return topk_seen_proto_sim, best_proto_match_fmap

# push each prototype to the nearest patch in the training set
def push_prototypes(dataloader, # pytorch dataloader (must be unnormalized in [0,1])
                    multi_ppnet, # pytorch network with prototype_vectors
                    preprocess_func=None, # normalize if needed
                    root_dir_for_saving_prototypes=None, # if not None, prototypes will be saved here
                    epoch_number=None, # if not provided, prototypes saved previously will be overwritten
                    log=print):

    multi_ppnet.eval()
    start = time.time()
    log('\tStarting Push')
    
    # ======================== SETUP DIR FOR SAVING PROTOTYPES ========================
    
    if root_dir_for_saving_prototypes != None:
        if epoch_number != None:
            save_proto_dir = os.path.join(root_dir_for_saving_prototypes, f'epoch-{epoch_number}')
            makedir(save_proto_dir)
        else:
            save_proto_dir = root_dir_for_saving_prototypes
    else:
        save_proto_dir = None
        
    # ======================== VARS SETUP ========================

    top_k = 5 # number of top patches to save for each prototype
    
    num_p, p_channels, p_h, p_w = my_prototype_shape
    
    # to save the best top-k similarities seen so far per prototype
    topk_seen_proto_sim = np.full((num_p, top_k), -np.inf)
    
    # saves the patch representation that gives the current best patch
    best_proto_match_fmap = np.zeros((num_p, p_channels, p_h, p_w, top_k)) # clones the shape of the prototypes 
    
    # ======================== PUSH OF PROTOTYPES REPRESENTATIONS TO IMAGE ========================

    for push_iter, (image, mask, label) in enumerate(dataloader):
        
        # start_index_of_search keeps track of the index of the image assigned to serve as prototype
        search_batch_start_idx = push_iter * dataloader.batch_size
        
        # Search for prototype patches in the current batch
        topk_seen_proto_sim, best_proto_match_fmap = _update_prototypes_on_batch(image, multi_ppnet, save_proto_dir, preprocess_func, 
                                                                                topk_seen_proto_sim, best_proto_match_fmap, top_k)

    end = time.time()
    log(f'\tPush time: \t{end -  start:.3f}')
