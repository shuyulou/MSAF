import torch
from model import MSAF
from tqdm import tqdm
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from data_process import data_processing
import random

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def test(args):
 
    set_seed(3407)
    with torch.no_grad():
        test_loader, test_len = data_processing(args, dataset_split_type=3)

        top_model = TopModel_coca(args)
        pth_path = [f for f in os.listdir(args.save_model) if f.endswith('.pth')][0]
        checkpoint = torch.load(os.path.join(args.save_model, pth_path))
        # print(checkpoint.keys())
        top_model.load_state_dict(checkpoint, strict = False) # , strict = False

        if args.cuda is True:
            top_model = top_model.cuda()
        top_model.eval()

        y_true = []
        y_pre = []
        test_loss = 0
        total_its = 0

        test_loader_tqdm = tqdm(test_loader, desc='Test')
        for _, data in enumerate(test_loader_tqdm):
            text_ids, text_masks, images, image_masks, labels = data
            if args.cuda is True:
                text_ids = text_ids.cuda()
                text_masks = text_masks.cuda()
                images = images.cuda()
                image_masks = image_masks.cuda()
                labels = labels.cuda()

            pre, cls_loss, cl_loss, _, _ = top_model(text_ids, text_masks, images, image_masks, labels)
            loss = cls_loss + cl_loss * 0.1
            test_loss += loss.item()
            total_its += 1
            y_true.extend(labels.cpu())
            y_pre.extend(pre.cpu())
            test_loader_tqdm.set_description(f"Test, loss: {loss:.6f}")

        test_loss /= total_its
        y_true = np.array(y_true)
        y_pre = np.array(y_pre)
        test_acc = accuracy_score(y_true, y_pre)
        test_F1_weighted = f1_score(y_true, y_pre, average='weighted')
        test_R_weighted = recall_score(y_true, y_pre, average='weighted')
        test_precision_weighted = precision_score(y_true, y_pre, average='weighted')

        test_F1_micro = f1_score(y_true, y_pre, average='micro')
        test_R_micro = recall_score(y_true, y_pre, average='micro')
        test_precision_micro = precision_score(y_true, y_pre, average='micro')

        test_F1_macro = f1_score(y_true, y_pre, average='macro')
        test_R_macro = recall_score(y_true, y_pre, average='macro')
        test_precision_macro = precision_score(y_true, y_pre, average='macro')

        test_status = f'''   Test Status   
        Accuracy: {test_acc:.6f}
        F1(weighted): {test_F1_weighted:.6f}
        R(weighted): {test_R_weighted:.6f}
        Precision(weighted): {test_precision_weighted:.6f}
        F1(micro): {test_F1_micro:.6f}
        R(micro): {test_R_micro:.6f}
        Precision(micro): {test_precision_micro:.6f}
        F1(macro): {test_F1_macro:.6f}
        R(macro): {test_R_macro:.6f}
        Precision(macro): {test_precision_macro:.6f}
        loss: {test_loss:.6f}\n\n\n'''

        print(test_status)
