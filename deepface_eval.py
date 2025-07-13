from deepface import DeepFace
from PSiameseDataset import my_test_loader

model_name = "Dlib"

test_loader = my_test_loader(False, normalized=True)

fp = 0
fn = 0
tp = 0
tn = 0

eps = 1e-7  # to avoid division by zero

if __name__ == "__main__":

    for batch_idx, (anchor_img_path, other_img_path, label) in enumerate(test_loader):
        print(f"Batch {batch_idx + 1}/{len(test_loader)}")
        
        for i in range(len(anchor_img_path)):
            decision = DeepFace.verify(anchor_img_path[i], other_img_path[i], model_name=model_name, enforce_detection=False)
            if decision["verified"]:
                if label[i] == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if label[i] == 0:
                    tn += 1
                else:
                    fn += 1
                
        acc = (tp + tn) / (tp + tn + fp + fn)
        print(f'\tacc: \t{(acc * 100):.3f}%')
        
    print(f'\tpre: \t{(tp / (tp + fp + eps) * 100):.3f}%')
    print(f'\trec: \t{(tp / (tp + fn + eps) * 100):.3f}%')
    print(f'\tf1: \t{(2 * tp) / (2 * tp + fp + fn + eps) * 100:.3f}%')
    
    # Confusion matrix
    print(f'\ttn: \t{tn}')
    print(f'\tfp: \t{fp}')
    print(f'\tfn: \t{fn}')
    print(f'\ttp: \t{tp}')
    
    print(decision)
