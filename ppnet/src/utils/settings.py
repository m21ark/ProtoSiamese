# ==================================== Model Config ====================================

my_base_architecture = 'vgg19'
my_img_size = 224

my_num_classes = 245
my_num_face_prototypes = 10
my_prototype_shape = (my_num_face_prototypes, 128, 1, 1) # number of prototypes, channels, height, width

# ==================================== Data Paths ====================================

my_data_path = 'celebmask_245/'

# Images
my_img_train_aug_dir = my_data_path + 'images/train_augmented/'      # augmented
my_img_train_push_dir = my_data_path + 'images/train/'               # no augmentation (used for pushing to patches)
my_img_test_dir = my_data_path + 'images/test/' 

# Ground truth masks
my_mask_train_aug_dir = my_data_path + 'masks/train_augmented/'
my_mask_train_push_dir = my_data_path + 'masks/train/' 
my_mask_test_dir = my_data_path + 'masks/test/'

# ==================================== Training Scheme ====================================

my_batch_size = 32
my_num_train_epochs = 200
my_num_warm_epochs = 5

my_early_stopping_patience = 15
my_early_stopping_delta = 0.01

my_acc_threshold_to_save_model = 0.7

my_push_start = 5
my_push_epochs = [i for i in range(my_num_train_epochs) if i % 2 == 0]
my_push_iterations = 10

my_prototype_activation_function = 'linear' # log or linear
my_add_on_layers_type = 'regular' # bottleneck or regular

my_joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}

my_warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

my_last_layer_optimizer_lr = 1e-4
my_joint_lr_step_size = 5
