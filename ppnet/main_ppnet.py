import os
import re

import ppnet.src.ppnet as ppnet
from ppnet.src.trainer import *

# py -m ppnet.main_ppnet        (in code/ dir)

if __name__ == "__main__":
    
    # ==================================== INITIAL SETUP ====================================

    base_architecture_type = re.match('^[a-z]*', my_base_architecture).group(0)

    model_dir = './ppnet_train_output/'
    makedir(model_dir)
    save_code_state_ppnet(model_dir + "code/")

    log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
    img_dir = os.path.join(model_dir, 'img')
    makedir(img_dir)

    # ==================================== LOAD DATASET ====================================
    
    train_loader = train_aug_loader_helper(True)
    train_push_loader = train_push_loader_helper(False)
    test_loader = test_loader_helper(False, normalized=True)

    log(f'training set size: {len(train_loader.dataset)}')
    log(f'push set size: {len(train_push_loader.dataset)}')
    log(f'test set size: {len(test_loader.dataset)}')
    log(f'batch size: {my_batch_size}')

    # ==================================== CONSTRUCT MODEL ====================================

    model = ppnet.construct_PPNet()
    model = model.to(MY_GPU_DEVICE_NAME)
    ppnet_multi = torch.nn.DataParallel(model)

    # ==================================== SETUP OPTIMIZER ====================================

    warm_optimizer = torch.optim.Adam([
        {'params': model.add_on_layers.parameters(), 'lr': my_warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
        {'params': model.prototype_vectors, 'lr': my_warm_optimizer_lrs['prototype_vectors']},
    ])

    joint_optimizer = torch.optim.Adam([
        {'params': model.features.parameters(), 'lr': my_joint_optimizer_lrs['features'], 'weight_decay': 1e-3},
        {'params': model.add_on_layers.parameters(), 'lr': my_joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
        {'params': model.prototype_vectors, 'lr': my_joint_optimizer_lrs['prototype_vectors']},
    ])

    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=my_joint_lr_step_size, gamma=0.5)
    last_layer_optimizer = torch.optim.Adam([{'params': model.last_layer.parameters(), 'lr': my_last_layer_optimizer_lr}])

    # ==================================== TRAIN MODEL ====================================
    
    paths = (model_dir, img_dir)
    loaders = (train_loader, train_push_loader, test_loader)
    optimizers = (warm_optimizer, joint_optimizer, last_layer_optimizer, joint_lr_scheduler)
    
    trainer = Trainer(ppnet_multi, log, loaders, optimizers, paths)

    for epoch in range(my_num_train_epochs):
        log('epoch: \t{0}'.format(epoch))

        # Train the model
        if epoch < my_num_warm_epochs: 
           trainer.warm_train()
        else: 
            trainer.joint_train()

        # Test the model and save it if it performs better
        acc, early_stop = trainer.test_save(epoch)
        
        if early_stop:
            break
            
        # if we are at the epoch when we want to push the prototypes to file
        if epoch >= my_push_start and epoch in my_push_epochs:
          trainer.push_epoch(epoch, acc)
    
    logclose()