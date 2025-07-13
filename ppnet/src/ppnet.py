import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F

from ppnet.src.models.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from ppnet.src.models.densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features
from ppnet.src.models.vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features,\
                         vgg19_features, vgg19_bn_features
from ppnet.src.utils.receptive_field import compute_proto_layer_rf_info_v2

from ppnet.src.utils.settings import *

base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'densenet121': densenet121_features,
                                 'densenet161': densenet161_features,
                                 'densenet169': densenet169_features,
                                 'densenet201': densenet201_features,
                                 'vgg11': vgg11_features,
                                 'vgg11_bn': vgg11_bn_features,
                                 'vgg13': vgg13_features,
                                 'vgg13_bn': vgg13_bn_features,
                                 'vgg16': vgg16_features,
                                 'vgg16_bn': vgg16_bn_features,
                                 'vgg19': vgg19_features,
                                 'vgg19_bn': vgg19_bn_features}

class PPNet(nn.Module):

    def __init__(self, features, proto_layer_rf_info):
        super(PPNet, self).__init__()
        self.num_prototypes = my_prototype_shape[0]
        self.num_classes = my_num_classes
        self.epsilon = 1e-7
        self.prototype_activation_function = my_prototype_activation_function
        self.proto_layer_rf_info = proto_layer_rf_info # [7, 32, 268, 16.0]
        self.max_patch_2_proto_dist = my_prototype_shape[1] * my_prototype_shape[2] * my_prototype_shape[3]

        # ==================================== 1. Feature Extractor ====================================

        self.features = features

        features_name = str(self.features).upper()
        
        if features_name.startswith('VGG') or features_name.startswith('RES'):
            temp = [i for i in features.modules() if isinstance(i, nn.Conv2d)]
            first_add_on_layer_in_channels = temp[-1].out_channels
            
        elif features_name.startswith('DENSE'):
            temp = [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)]
            first_add_on_layer_in_channels = temp[-1].num_features
        else:
            raise Exception('Other base architectures besides (RESNET, VGG, DENSE) are NOT implemented')
        
        # ==================================== 2. Add-on Layers ====================================

        if my_add_on_layers_type == 'bottleneck':
            # progressive downsampling via 1x1 convolutions and ReLU activations over many layers
            add_on_layers = construct_bottle_neck_layer(first_add_on_layer_in_channels)
            self.add_on_layers = nn.Sequential(*add_on_layers)
            
        else: # simple: directly converts in a single step from the feature extractor to the 
            # prototype layer channel sizes (less train parameters but less flexible feature transformation)
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=my_prototype_shape[1], kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=my_prototype_shape[1], out_channels=my_prototype_shape[1], kernel_size=1),
                nn.Sigmoid()
                )

        # ==================================== 3. Prototype Layer ====================================
        
        # intialize the prototype vectors as spread out as possible
        random_matrix = torch.randn(my_prototype_shape[1], my_prototype_shape[1])
        Q, _ = torch.linalg.qr(random_matrix)
        orthogonal_vectors = Q[:my_prototype_shape[0]]
        orthogonal_vectors = orthogonal_vectors.view(*my_prototype_shape)
        
        self.prototype_vectors = nn.Parameter(orthogonal_vectors, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(my_prototype_shape), requires_grad=False)
        
        # ==================================== 5. Final Linear Layer ====================================

        p_channel_size = my_prototype_shape[1]
        feature_width = self.proto_layer_rf_info[0]

        self.last_layer = nn.Linear(p_channel_size * feature_width**2, self.num_classes, bias=False)

        self._initialize_weights()

    def _calculate_distance_squared(self, features):
        
        # features: [b_size, p_channel_count, 7, 7]
        # L2 Distance formula: (x - p)^2 = x^2 - 2xp + p^2
        
        # self.ones: [num_prototypes, proto_channel_size, 1, 1]
        x2_patch_sum = F.conv2d(input=features**2, weight=self.ones) # [b_size, num_prototypes, 7, 7]
        
        p2 = self.prototype_vectors ** 2 # [num_prototypes, proto_channel_size, 1, 1]
        p2 = torch.sum(p2, dim=(1, 2, 3)) # [num_prototypes]
        p2_reshape = p2.view(-1, 1, 1) # [num_prototypes, 1, 1]

        xp = F.conv2d(input=features, weight=self.prototype_vectors)  # [b_size, num_prototypes, 7, 7]
        
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast [b_size, num_prototypes, 7, 7]
        
        # use ReLU to avoid negative distances
        distances = F.relu(x2_patch_sum + intermediate_result) 
        return distances # [b_size, num_prototypes, 7, 7]
    
    def _calculate_distance_linear(self, features):
        # Compute distances between feature maps and global prototypes
        distances = torch.norm(features.unsqueeze(1) - self.prototype_vectors, dim=2, p=2)
        return distances
    
    def _calculate_cos_sim(self, norm_features: torch.Tensor):
     
        norm_protos = F.normalize(self.prototype_vectors, dim=1)  # [num_protos, 128, 1, 1]

        # Compute cosine similarities offset to [0, 2]
        return 1 + F.cosine_similarity(
            norm_features.unsqueeze(1), # [b_size, 1, 128, 7, 7]
            norm_protos.unsqueeze(0), # [1, num_protos, 128, 7, 7]
            dim=2
        ) # [b_size, num_protos, 7, 7]
    
    def _distance_2_similarity(self, distances):
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return - distances # doing self.max_patch_2_proto_dist - distances will not work for some reason
        else:
            raise Exception('Invalid activation function: ' + self.prototype_activation_function)
        
    def forward(self, input_img):
        
        # Extract feature maps
        extracted_features = self.features(input_img)
        extracted_features = self.add_on_layers(extracted_features) # normalized features [b_size, p_channel_count, 7, 7]

        # Get proto similarity scores to the patches
        similarities = self._calculate_cos_sim(extracted_features) # [b_size, num_protos, 7, 7] ; in range [0, 2] (where 2 is max sim)

        # get a attention map for each patch --> works worse than the uncomented way below
        # similarities = similarities.max(dim=1)[0] # [b_size, 7, 7]
        # similarities = similarities.sum(dim=1) # [b_size, 7, 7]
        # # multiply the attention map with the feature map across the channel dimension
        # weighted_features = similarities.unsqueeze(1) * extracted_features # [b_size, p_channel_count, 7, 7]

        # Use similarity scores to weight feature maps (this version preserves per-prototype feature weighting)
        weighted_features = extracted_features.unsqueeze(1) * similarities.unsqueeze(2) # [b_size, num_protos, p_channel_count, 7, 7]
        weighted_features = weighted_features.sum(dim=1) # [b_size, p_channel_count, 7, 7]
        
        # Flatten for final classification and compute logits
        last_layer_input = weighted_features.view(weighted_features.shape[0], -1) # [b_size, p_channel_count * 49]
        logits = self.last_layer(last_layer_input) # [b_size, num_classes]

        return extracted_features, similarities, logits

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # randomly initialize last layer's weights and no bias
        nn.init.kaiming_normal_(self.last_layer.weight, mode='fan_out', nonlinearity='relu')

def construct_PPNet():
    features = base_architecture_to_features[my_base_architecture](pretrained=True)
    layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()
    proto_layer_rf_info = compute_proto_layer_rf_info_v2(my_img_size, layer_filter_sizes, layer_strides, layer_paddings, my_prototype_shape[2])
    return PPNet(features=features, proto_layer_rf_info=proto_layer_rf_info)

def construct_bottle_neck_layer(first_add_on_layer_in_channels):
    
    add_on_layers = []
    current_in_channels = first_add_on_layer_in_channels
            
    # progressively taper down the channel depth from the feature extractor to the prototype layer
    while (current_in_channels > my_prototype_shape[1]) or (len(add_on_layers) == 0):

        # halve the channel depth at each loop iteration, until reaching prototype_shape channel size
        # e.g. input 512 & prototype_shape 64  : 512 -> 256 -> 128 -> 64
        current_out_channels = max(my_prototype_shape[1], (current_in_channels // 2)) 
        add_on_layers.append(nn.Conv2d(in_channels=current_in_channels, out_channels=current_out_channels, kernel_size=1))
        add_on_layers.append(nn.ReLU())
        
        # (25% neurons randomly turned off --> forces model to learn more robust representations to prevent overfitting)
        add_on_layers.append(nn.Dropout(0.25))  
        
        add_on_layers.append(nn.Conv2d(in_channels=current_out_channels, out_channels=current_out_channels, kernel_size=1))
        
        # if it is the last layer, use sigmoid instead of ReLU
        if current_out_channels > my_prototype_shape[1]:
            add_on_layers.append(nn.ReLU())
        else:
            assert(current_out_channels == my_prototype_shape[1])
            add_on_layers.append(nn.Sigmoid())
            
        # update the channel depth for the next iteration
        current_in_channels = current_in_channels // 2 
        
    return add_on_layers