import os
import re

from ppnet.src.utils.helpers import *
from ppnet.src.trainer import EarlyStopping
from PSiameseDataset import my_train_loader, my_test_loader
from protoSiamese import ProtoSiamese, protoSiameseTrain, protoSiameseTest, protoSiameseSave

if __name__ == "__main__":
    
    # ==================================== INITIAL SETUP ====================================

    base_architecture_type = re.match('^[a-z]*', my_base_architecture).group(0)

    model_dir = './psiamese_train_output/'
    makedir(model_dir)
    save_code_state_psiamese(model_dir + "code/")

    log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))

    # ==================================== LOAD DATASET ====================================
    
    train_loader = my_train_loader(True)
    test_loader = my_test_loader(False, normalized=True)

    log(f'training set size: {len(train_loader.dataset)}')
    log(f'test set size: {len(test_loader.dataset)}')
    log(f'batch size: {my_batch_size}')

    # ==================================== CONSTRUCT MODEL ====================================

    model = ProtoSiamese()
    model = model.to(MY_GPU_DEVICE_NAME)
    
    # # ==================================== SETUP OPTIMIZER ====================================

    optimizer = torch.optim.Adam([
        {'params': model.final_layer.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4},
        # {'params': model.ppnet.features.parameters(), 'lr': my_joint_optimizer_lrs['features'], 'weight_decay': 1e-3},
        # {'params': model.ppnet.add_on_layers.parameters(), 'lr': my_joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
        # {'params': model.ppnet.prototype_vectors, 'lr': my_joint_optimizer_lrs['prototype_vectors']},
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.75)

    # # ==================================== TRAIN MODEL ====================================
    
    early_stopper = EarlyStopping(patience=my_early_stopping_patience, min_delta=my_early_stopping_delta, acc_mode=False)
    max_seen_acc = 0
    
    for epoch in range(my_num_train_epochs):
        log('epoch: \t{0}'.format(epoch))

        # Train the model
        protoSiameseTrain(model, train_loader, optimizer, log)

        # Test the model and save it if it performs better
        acc100, test_loss = protoSiameseTest(model, test_loader, log)
        
        # If new best accuracy, save the model
        acc100 = round(acc100, 1)
        if acc100 > max_seen_acc:
            max_seen_acc = acc100
            protoSiameseSave(model, model_dir, acc100)
        
        # Stop training if early stopping criteria is met
        if early_stopper(test_loss):
            break
    
    logclose()