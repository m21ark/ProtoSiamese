import time
import torch
import torch.nn.functional as F

from ppnet.src.utils.helpers import *

# ==================================== AUX FUNCS ====================================

def train(model, dataloader, optimizer, log=print):
    assert(optimizer is not None)
    log('\ttrain')
    model.train()
    return _train_or_test(model, dataloader, optimizer, log)

def test(model, dataloader, log=print):
    log('\ttest')
    model.eval()
    return _train_or_test(model, dataloader, None, log)

def _unwrap_model(model):
    # unwrap model if necessary and convert to GPU parallel
    while type(model) == torch.nn.DataParallel:
        model = model.module
    model = model.to(MY_GPU_DEVICE_NAME)
    return torch.nn.DataParallel(model)

def _avg_proto_distance(model):
    prototypes = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.sum((torch.unsqueeze(prototypes, dim=2) - torch.unsqueeze(prototypes.t(), dim=0)) ** 2, dim=1)
    return torch.mean(p_avg_pair_dist).item()

def getActivationMask(mask, similarities):

   # make mask binary, 1 if different from 0,0,0 
    mask = mask / 255
    mask = torch.sum(mask, dim=1) # [b_size, 224, 224]
    mask = mask > 0
    mask = mask.float() # [b_size, 224, 224]
    
    # get the activation map of the prototypes, flattening the number of prototypes with max
    activation_map = torch.max(similarities, dim=1)[0] # [b_size, 7, 7]    
    
    # upsample the activation map to the original image size for comparison
    upsampled_activation_map = F.interpolate(activation_map.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False).squeeze(1) # [b_size, 224, 224]
    overlap_activations = upsampled_activation_map * mask
    
    return overlap_activations, upsampled_activation_map

# ==================================== MAIN FUNCS ====================================

def _model_evaluate(model, input, target, mask):
    
    # input: [b_size, 3, 224, 224] with channels normalized to [0, 1]
    # target: [b_size]
    # mask: [b_size, 3, 224, 224] with channels in [0, 255]
    
    # get model output
    _, similarities, logits = model.forward(input)
    _, predicted = torch.max(logits.data, dim=1)
    num_prototypes = my_prototype_shape[0]
    
    # ======================================== COMPUTE LOSS ========================================
    
    # compute loss
    cross_entropy_cost = torch.nn.functional.cross_entropy(logits, target)
    
    # Encourage images to match at least one prototype
    distances = 2 - similarities # [b_size, 7, 7] in range [0, 2]
    min_dists, _ = torch.min(distances, dim=1)  # [b_size, 7, 7]
    cluster_cost = torch.mean(min_dists)          
            
    # Punish prototypes if they activate outside the mask
    overlap_activations, upsampled_activation_map = getActivationMask(mask, similarities)
    outside_mask_cost = torch.mean(F.relu(upsampled_activation_map - overlap_activations))
    
    # Encourage prototypes to focus on different features
    max_sim = torch.max(similarities, dim=1)[0] # [b_size, 7, 7]
    min_sim = torch.min(similarities, dim=1)[0]
    spread_candy = torch.mean(F.relu(max_sim - min_sim)) # the higher the better
        
    # Encourage prototypes to focus on different features
    proto_vectors = model.prototype_vectors.view(num_prototypes, -1)
    proto_vectors = F.normalize(proto_vectors, p=2, dim=1)
    proto_sim = torch.mm(proto_vectors, proto_vectors.T)
    proto_sim = proto_sim - torch.eye(num_prototypes, device=proto_sim.device)  # Remove self-diagonal
    cos_sim_cost = torch.mean(proto_sim)  

    # ========================================= FINAL LOSS ========================================
    
    total_loss = 0
    total_loss += 0.70 * cross_entropy_cost # v39 was 0.6
    total_loss += 0.90 * cluster_cost # v39 was 0.7
    total_loss += 0.45 * cos_sim_cost
    total_loss -= 0.05 * spread_candy
    total_loss += 1 * outside_mask_cost # v39 was 1.5

    return total_loss, (cross_entropy_cost, cluster_cost, cos_sim_cost, spread_candy, outside_mask_cost), predicted

def _train_or_test(model, dataloader, optimizer, log):

    is_train = optimizer is not None 
    model = _unwrap_model(model)
    start_time = time.time()
    
    # statistics to collect
    stats_num_examples = 0
    stats_num_correct = 0
    stats_num_batches = 0
    stats_loss = 0
    
    # Loss metrics
    total_cross_entropy_cost = 0
    total_cluster_cost = 0
    total_cos_sim_cost = 0
    total_spread_candy = 0
    total_outside_mask_cost = 0
    
    # ======================================== FORWARD PASS ========================================
    
    for _, (image, mask, label) in enumerate(dataloader):
        input = image.to(MY_GPU_DEVICE_NAME) # [b_size, 3, 224, 224]
        label = label.to(MY_GPU_DEVICE_NAME) # [b_size]
        mask = mask.to(MY_GPU_DEVICE_NAME) # [b_size, 3, 224, 224]

        with torch.set_grad_enabled(is_train):
            
            # ======================================== EVAlUATION STATISTICS ========================================
            
            total_loss, loss_metrics, predicted = _model_evaluate(model.module, input, label, mask)                    
            
            stats_num_examples += label.size(0)
            stats_num_batches += 1
            stats_num_correct += (predicted == label).sum().item()
            stats_loss += total_loss.item()
            
            # Loss metrics
            cross_entropy_cost, cluster_cost, cos_sim_cost, spread_candy, outside_mask_cost = loss_metrics
            total_cross_entropy_cost += cross_entropy_cost.item()
            total_cluster_cost += cluster_cost.item()
            total_cos_sim_cost += cos_sim_cost.item()
            total_spread_candy += spread_candy.item()
            total_outside_mask_cost += outside_mask_cost.item()

        # ======================================== TRAINING OPTIMIZATION ========================================

        # backpropagate the loss and update the model weights
        if is_train:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Only for local debugging
            if MY_GPU_DEVICE_NAME == 'mps':
                print(f'batch {stats_num_batches}/{len(dataloader)}    loss: {total_loss.item():.3f}', end='\r')
    
    # ======================================== LOG RESULTS ========================================

    end_time = time.time()
    
    avg_print = lambda name, metric: log(f'\t{name}: \t{(metric / stats_num_batches):.3f}')
    
    log(f'\ttime: \t{(end_time - start_time):.3f}')
    log(f'\taccu ↑: \t{(stats_num_correct / stats_num_examples * 100):.3f}%')
    avg_print('loss ↓: ', stats_loss)
    avg_print('cross ent ↓', total_cross_entropy_cost)
    avg_print('cluster ↓', total_cluster_cost)
    avg_print('cos sim ↓', total_cos_sim_cost)
    avg_print('spread ↑', total_spread_candy) # The max will be 7x7xb_size
    avg_print('outside mask ↓', total_outside_mask_cost)
    
    log(f'\tavg p dist ↑: \t{(_avg_proto_distance(model)):.3f}')
    log('-' * 50)
    
    return stats_num_correct / stats_num_examples