from ppnet.src.utils.helpers import *
import ppnet.src.train_and_test as tnt
import ppnet.src.push as push

def set_train_mode_last_only(model, log=print):
    
    # Freeze CNN, add-on layers, and prototype vectors
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    
    # Unfreeze last layer
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tTraining only the last layer')

def set_train_mode_warm_only(model, log=print):
    
    # Freeze CNN
    for p in model.module.features.parameters():
        p.requires_grad = False
        
    # Unfreeze add-on layers, prototype vectors, and last layer
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tTraining all layers except the CNN (warm start)')

def set_train_mode_joint(model, log=print):
    
    # Unfreeze all layers
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tTraining all layers')

class Trainer:       
 
    def __init__(self, ppnet_multi, log, loaders, optimizers, paths):
        
        train_loader, train_push_loader, test_loader = loaders
        warm_optimizer, joint_optimizer, last_layer_optimizer, joint_lr_scheduler = optimizers
        
        self.model = ppnet_multi
        self.log = log
        
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_push_loader = train_push_loader
        
        self.optimizer_warm = warm_optimizer
        self.optimizer_joint = joint_optimizer
        self.optimizer_last = last_layer_optimizer
        self.joint_lr_scheduler = joint_lr_scheduler
        
        self.model_dir, self.img_dir = paths
        
        self.max_nopush_acc = 0
        self.max_iter_acc = 0
        
        self.early_stopper = EarlyStopping(patience=my_early_stopping_patience, min_delta=my_early_stopping_delta)
    
    def warm_train(self):
        # Train only part of the model
        set_train_mode_warm_only(model=self.model, log=self.log)
        _ = tnt.train(model=self.model, dataloader=self.train_loader, optimizer=self.optimizer_warm, log=self.log)
        
    def joint_train(self):
        # Train the model jointly
        set_train_mode_joint(model=self.model, log=self.log)
        _ = tnt.train(model=self.model, dataloader=self.train_loader, optimizer=self.optimizer_joint, log=self.log)
        self.joint_lr_scheduler.step()
        
    def last_train(self, epoch):
        # Train only the last layer for a few iterations
        if my_prototype_activation_function == 'linear':
            return
            
        set_train_mode_last_only(model = self.model, log=self.log) 

        for i in range(my_push_iterations): 
            self.log('iteration: \t{0}'.format(i))
            _ = tnt.train(model=self.model, dataloader=self.train_loader, optimizer=self.optimizer_last, log=self.log)
            acc, early_stop = self.test_save(epoch, i)
            
    # Test the model, see if performance improved and save the model if it did
    def _save_model(self, model_name, acc, unconditionalSave=False):
        if acc >= my_acc_threshold_to_save_model or unconditionalSave:
            self.log('\tSaving model with {0:.2f}%'.format(acc * 100))
            torch.save(self.model.module.state_dict(), f=os.path.join(self.model_dir, f'{model_name}{acc*100:.2f}.pth'))
        
    def test_save(self, epoch, i=-1):
        
        acc = tnt.test(model=self.model, dataloader=self.test_loader, log=self.log)
        
        if i != -1:
            if acc > self.max_iter_acc:
                self.max_iter_acc = acc
                self. _save_model(f'{epoch}_{i}iter', acc) # Push Iteration
        else:
            if acc > self.max_nopush_acc:
                self._save_model(f'{epoch}nopush', acc) # No Push Epoch
                self.max_nopush_acc = acc       
                
            # Stop training early if test accuracy isn't improving on non-push epochs
            return acc, self.early_stopper(acc)
        
        return acc, False 
    
    def push_epoch(self, epoch, acc):
        
        push.push_prototypes(
            self.train_push_loader, # must be unnormalized in [0,1]
            multi_ppnet=self.model,
            preprocess_func=preprocess_input_function, # normalize if needed
            root_dir_for_saving_prototypes=self.img_dir,
            epoch_number=epoch, log=self.log)
                
        self._save_model(f'{epoch}push', acc, unconditionalSave=True) # Push Epoch
           
        self.last_train(epoch)

class EarlyStopping:
    # Defines an early stopping mechanism to stop training 
    # if the model isn't improvingby at least a certain  
    # threshold amount after a certain number of epochs
    
    def __init__(self, patience, min_delta, acc_mode=True):
        self.patience = patience
        self.min_delta = min_delta
        self.best_accu = 0 
        self.min_test_loss = float('inf')
        self.counter = 0
        self.acc_mode = acc_mode

    def __call__(self, curr_val):
        if self.acc_mode and (curr_val > self.best_accu + self.min_delta):
            self.best_accu = curr_val
            self.counter = 0  # Reset patience
        elif not self.acc_mode and (curr_val < self.min_test_loss - self.min_delta):
            self.min_test_loss = curr_val
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping triggered! Stopping training.")
                return True
        return False
