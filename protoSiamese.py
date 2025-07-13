import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from ppnet.src.utils.helpers import *

def gen_local_prototypes(extracted_features, similarities, num_prototypes):

        # for each global prototype channel in similarities, we take the patch cordinates of the largest activation
        # and use that cordinates to extract the local prototype from the extracted features
        
        local_prototypes = [] # [bsize, num_protos, 128]
        
        picked_cordinates = []
        
        for b in range(similarities.shape[0]):
                proto_list = []
                for p in range(num_prototypes):
                    
                    # Get the similarity map for this prototype
                    sim_map = similarities[b, p]  # [7, 7]
                    
                    # Find the index of the maximum activation and get (i, j) from flattened index
                    max_idx = sim_map.view(-1).argmax()
                    i, j = divmod(max_idx.item(), sim_map.shape[1])
                    
                    picked_cordinates.append((i, j))

                    # Extract the corresponding feature vector from extracted_features
                    local_feat = extracted_features[b, :, i, j]  # [128]
                    proto_list.append(local_feat)
                    
                # Stack the prototypes for this batch item
                local_prototypes.append(torch.stack(proto_list))  # [num_protos, 128]

        # Stack over the batch
        return picked_cordinates, torch.stack(local_prototypes)  # [bsize, num_protos, 128]
    
class ProtoSiamese(nn.Module):

    def __init__(self):
        super(ProtoSiamese, self).__init__()
        self.epsilon = 1e-8
        
        # Load the pre-trained model
        self.ppnet = ppnet_model_loader("../models/ppnet_v38.pth")
        self.num_prototypes = self.ppnet.num_prototypes
        
        self.trainPPnet = False
        self.set_should_train_ppnet(self.trainPPnet)
        
        # final layer to classify the local prototypes
        # Gets num_prototypes distances as input and outputs 1 similarity value
        self.final_layer = nn.Linear(self.num_prototypes, 1, bias=True)
        
    def set_should_train_ppnet(self, should_train):
        for p in self.ppnet.features.parameters():
            p.requires_grad = should_train
            
        for p in self.ppnet.add_on_layers.parameters():
            p.requires_grad = should_train
            
        for p in self.ppnet.last_layer.parameters():
            p.requires_grad = should_train
        
        self.ppnet.prototype_vectors.requires_grad = should_train
    
    def forward(self, anchor_img, other_img):
        
        # extracted_features: [bsize, 128, 7, 7]
        # similarities: [bsize, num_protos, 7, 7]
        
        extracted_features, similaritiesA, _ = self.ppnet(anchor_img)
        picked_cordinatesA, local_prototypesA = gen_local_prototypes(extracted_features, similaritiesA, self.num_prototypes) # [bsize, num_protos, 128] (1x1)
        
        extracted_features, similaritiesB, _ = self.ppnet(other_img)
        picked_cordinatesB, local_prototypesB = gen_local_prototypes(extracted_features, similaritiesB, self.num_prototypes)
        
        # Compute the cosine similarity between the pairs of local prototypes
        similarity_pairs = F.cosine_similarity(local_prototypesA, local_prototypesB, dim=2) # [bsize, num_protos]

        # Compute the final similarity score based on the pairwise similarities between local prototypes
        logits = self.final_layer(similarity_pairs).view(-1) # [bsize]
        
        return logits, extracted_features, similaritiesA, similaritiesB, similarity_pairs, picked_cordinatesA, picked_cordinatesB

# ==================================== TRAINING AND TESTING ====================================

def _protoSiameseTrainTest(model: ProtoSiamese, dataset_loader, optimizer, log):
    
    isTrain = optimizer is not None
    log(f'\t{"train" if isTrain else "test"}')
    
    if isTrain:
        model.train()
    else:
        model.eval() 
        
    start_time = time.time()

    tn, fp, fn, tp = 0, 0, 0, 0
    
    classif_loss_func = torch.nn.BCEWithLogitsLoss()
    
    for batch_idx, (anchor_img, other_img, label) in enumerate(dataset_loader):
        
        anchor_img = anchor_img.to(MY_GPU_DEVICE_NAME) # [bsize, 3, 224, 224]
        other_img = other_img.to(MY_GPU_DEVICE_NAME) # [bsize, 3, 224, 224]
        label = label.to(MY_GPU_DEVICE_NAME) # [bsize]
        
        # Forward pass
        logits = model(anchor_img, other_img)[0] # [bsize]
        
        # Compute tn, fp, fn, tp
        predicted = torch.sigmoid(logits).round() # [b_size] --> Either 0.0 or 1.0 (0.5 threshold)

        tp += ((predicted == 1) & (label == 1)).sum().item()
        tn += ((predicted == 0) & (label == 0)).sum().item()
        fp += ((predicted == 1) & (label == 0)).sum().item()
        fn += ((predicted == 0) & (label == 1)).sum().item()

        # Compute loss with bcelogits
        loss = classif_loss_func(logits, label.float())
        
        # Backpropagation
        if isTrain:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    # Logging
    acc100 = round(((tp + tn) / (tp + tn + fp + fn)) * 100, 2)
    log(f'\n\ttime: \t{(time.time() - start_time):.3f}')
    log(f'\tloss: \t{loss.item():.3f}')

    # Performance Metrics
    log(f'\tacc: \t{acc100}%')
    
    log(f'\tpre_1: \t{(tp / (tp + fp + model.epsilon) * 100):.3f}%')
    log(f'\trec_1: \t{(tp / (tp + fn + model.epsilon) * 100):.3f}%')
    log(f'\tf1_1: \t{(2 * tp) / (2 * tp + fp + fn + model.epsilon) * 100:.3f}%')
    
    log(f'\tpre_0: \t{(tn / (tn + fn + model.epsilon) * 100):.3f}%')
    log(f'\trec_0: \t{(tn / (tn + fp + model.epsilon) * 100):.3f}%')
    log(f'\tf1_0: \t{(2 * tn) / (2 * tn + fn + fp + model.epsilon) * 100:.3f}%')
    
    # Confusion matrix
    log(f'\ttn: \t{tn}')
    log(f'\tfp: \t{fp}')
    log(f'\tfn: \t{fn}')
    log(f'\ttp: \t{tp}')
    log('-' * 50)
    
    return acc100, loss
    
def protoSiameseTrain(model: ProtoSiamese, train_loader, optimizer, log):
    return _protoSiameseTrainTest(model, train_loader, optimizer, log)

def protoSiameseTest(model: ProtoSiamese, test_loader, log):
    return _protoSiameseTrainTest(model, test_loader, None, log)

def protoSiameseSave(model: ProtoSiamese, save_path, acc100):
    torch.save(model.state_dict(), f=os.path.join(save_path, f'psiamese_{acc100:.2f}.pth'))

def protoSiameseLoader(load_path):
    model = ProtoSiamese()
    model.load_state_dict(torch.load(load_path, map_location=torch.device(MY_GPU_DEVICE_NAME), weights_only=False))
    model = model.to(MY_GPU_DEVICE_NAME)
    while type(model) == torch.nn.DataParallel:
        model = model.module
    model.eval()
    return model