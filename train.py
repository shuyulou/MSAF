import os
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import numpy as np
from data_process import data_processing
from model import MSAF
from utils.write import write_to_file
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

def test(args, test_loader, msaf, cur_epoch, summary_writer):
    with torch.no_grad():
        msaf.eval()

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

            pre, cls_loss, cl_loss, ct_loss, it_loss = msaf(text_ids, text_masks, images, image_masks, labels)
            loss = cls_loss + cl_loss * 0.01
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
        write_to_file(os.path.join(args.save_model, 'train_log.txt'), test_status)
        write_to_file(os.path.join(args.save_model, 'train_result.txt'), 
                      f'Accuracy: {test_acc:.6f},F1(weighted): {test_F1_weighted:.6f}\n{str(y_pre.tolist())}\n')

        summary_writer.add_scalar('Test Info/Run Loss', test_loss, global_step=cur_epoch)
        summary_writer.add_scalar('Test Info/Accuracy', test_acc, global_step=cur_epoch)
        summary_writer.add_scalar('Test Info/F1(weighted)', test_F1_weighted, global_step=cur_epoch)
        summary_writer.add_scalar('Test Info/F1(micro)', test_F1_micro, global_step=cur_epoch)
        summary_writer.add_scalar('Test Info/F1(macro)', test_F1_macro, global_step=cur_epoch)
        summary_writer.add_scalar('Test Info/R(weighted)', test_R_weighted, global_step=cur_epoch)
        summary_writer.add_scalar('Test Info/Precision(weighted)', test_precision_weighted, global_step=cur_epoch)
        summary_writer.flush()

    return test_acc, test_F1_macro


def dev(args, dev_loader, msaf, cur_epoch, summary_writer):
    with torch.no_grad():
        msaf.eval()

        y_true = []
        y_pre = []
        val_loss = 0
        total_its = 0
        
        dev_loader_tqdm = tqdm(dev_loader, desc='Validate')
        for _, data in enumerate(dev_loader_tqdm):
            text_ids, text_masks, images, image_masks, labels = data
            if args.cuda is True:
                device = torch.device("cuda")
                text_ids, text_masks, images, image_masks, labels = [d.to(device) for d in data] # , non_blocking=True

            pre, cls_loss, cl_loss, ct_loss, it_loss = msaf(text_ids, text_masks, images, image_masks, labels)
            loss = cls_loss + cl_loss * 0.01
            val_loss += loss.item()
            total_its += 1
            y_true.extend(labels.cpu())
            y_pre.extend(pre.cpu())
            dev_loader_tqdm.set_description(f"Validate, loss: {loss:.6f}")

        val_loss /= total_its
        y_true = np.array(y_true)
        y_pre = np.array(y_pre)
        val_acc = accuracy_score(y_true, y_pre)
        val_F1_weighted = f1_score(y_true, y_pre, average='weighted')
        val_R_weighted = recall_score(y_true, y_pre, average='weighted')
        val_precision_weighted = precision_score(y_true, y_pre, average='weighted')

        val_F1_micro = f1_score(y_true, y_pre, average='micro')
        val_R_micro = recall_score(y_true, y_pre, average='micro')
        val_precision_micro = precision_score(y_true, y_pre, average='micro')

        val_F1_macro = f1_score(y_true, y_pre, average='macro')
        val_R_macro = recall_score(y_true, y_pre, average='macro')
        val_precision_macro = precision_score(y_true, y_pre, average='macro')

        val_status = f'''   Validation Status   
        Accuracy: {val_acc:.6f}
        F1(weighted): {val_F1_weighted:.6f}
        R(weighted): {val_R_weighted:.6f}
        Precision(weighted): {val_precision_weighted:.6f}
        F1(micro): {val_F1_micro:.6f}
        R(micro): {val_R_micro:.6f}
        Precision(micro): {val_precision_micro:.6f}
        F1(macro): {val_F1_macro:.6f}
        R(macro): {val_R_macro:.6f}
        Precision(macro): {val_precision_macro:.6f}
        loss: {val_loss:.6f}\n'''
        print(val_status)
        write_to_file(os.path.join(args.save_model, 'train_log.txt'), val_status)

        summary_writer.add_scalar('Validation Info/Run Loss', val_loss, global_step=cur_epoch)
        summary_writer.add_scalar('Validation Info/Accuracy', val_acc, global_step=cur_epoch)
        summary_writer.add_scalar('Validation Info/F1(weighted)', val_F1_weighted, global_step=cur_epoch)
        summary_writer.add_scalar('Validation Info/R(weighted)', val_R_weighted, global_step=cur_epoch)
        summary_writer.add_scalar('Validation Info/Precision(weighted)', val_precision_weighted, global_step=cur_epoch)
        summary_writer.flush()

    return val_acc, val_F1_weighted


def train(args):
    set_seed(3407)
    summary_writer = SummaryWriter(log_dir=os.path.join(args.save_model, 'summary'))
    summary_writer.add_text('Hyperparameter', str(args))

    train_loader, train_len = data_processing(args, dataset_split_type=1)
    dev_loader, dev_len = data_processing(args, dataset_split_type=2)
    test_loader, test_len = data_processing(args, dataset_split_type=3)
    print(f'The size of the training set: {train_len}')
    print(f'The size of the development set: {dev_len}')
    print(f'The size of the test set: {test_len}')

    msaf = MSAF(args)
    if args.cuda is True:
        assert torch.cuda.is_available()
        msaf = msaf.cuda()

    optimizer = AdamW(msaf.parameters(), lr=args.lr, betas=(args.adamw_beta1, args.adamw_beta2), weight_decay=args.weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=args.lr_decay)

    best_f1 = 0
    best_acc = 0
    best_epoch = 0
    for epoch in tqdm(range(args.epoch), desc='Epoch'):
        msaf.train()

        y_true = []
        y_pre = []
        run_loss = 0
        total_its = 0

        train_loader_tqdm = tqdm(train_loader, desc='Train')
        train_loader_size = train_loader_tqdm.total

        acc_loss = 0
        acc_cls_loss = 0
        acc_cl_loss = 0
        acc_ct_loss = 0
        acc_it_loss = 0

        for index, data in enumerate(train_loader_tqdm):
            text_ids, text_masks, images, image_masks, labels = data
            if args.cuda is True:
                text_ids = text_ids.cuda()
                text_masks = text_masks.cuda()
                images = images.cuda()
                image_masks = image_masks.cuda()
                labels = labels.cuda()
            pre, cls_loss, cl_loss, ct_loss, it_loss = msaf(text_ids, text_masks, images, image_masks, labels)

            acc_cls_loss += cls_loss / args.acc_steps
            acc_cl_loss += cl_loss / args.acc_steps
            acc_ct_loss += ct_loss / args.acc_steps
            acc_it_loss += it_loss / args.acc_steps
            loss = (cls_loss + cl_loss * 0.01 + ct_loss + it_loss * 0.01) / args.acc_steps 
            acc_loss += loss
            loss.backward()

            if (index + 1) % args.acc_steps == 0 or index == train_loader_size - 1:
                train_loader_tqdm.set_description(f"Train, loss: {acc_loss:.6f}")
                summary_writer.add_scalar('Train Info/Total Loss', acc_loss.item(), global_step=index + epoch * train_loader_size)
                summary_writer.add_scalar('Train Info/Classify Loss', acc_cls_loss.item(), global_step=index + epoch * train_loader_size)
                summary_writer.add_scalar('Train Info/Contrastive Learning Loss', acc_cl_loss.item(), global_step=index + epoch * train_loader_size)
                summary_writer.add_scalar('Train Info/ContraryLoss Loss', acc_ct_loss.item(), global_step=index + epoch * train_loader_size)
                summary_writer.add_scalar('Train Info/IT_ContraryLoss Loss', acc_it_loss.item(), global_step=index + epoch * train_loader_size)
                summary_writer.flush()
                optimizer.step()
                optimizer.zero_grad()
                acc_loss = 0
                acc_cls_loss = 0
                acc_cl_loss = 0
                acc_ct_loss = 0
                acc_it_loss = 0

            y_true.extend(labels.cpu())
            y_pre.extend(pre.cpu())
            run_loss += loss.item() * args.acc_steps
            total_its += 1

        run_loss /= total_its
        y_true = np.array(y_true)
        y_pre = np.array(y_pre)
        train_acc = accuracy_score(y_true, y_pre)
        train_F1_weighted = f1_score(y_true, y_pre, average='weighted')
        train_R_weighted = recall_score(y_true, y_pre, average='weighted')
        train_precision_weighted = precision_score(y_true, y_pre, average='weighted')

        train_F1_micro = f1_score(y_true, y_pre, average='micro')
        train_R_micro = recall_score(y_true, y_pre, average='micro')
        train_precision_micro = precision_score(y_true, y_pre, average='micro')

        train_F1_macro = f1_score(y_true, y_pre, average='macro')
        train_R_macro = recall_score(y_true, y_pre, average='macro')
        train_precision_macro = precision_score(y_true, y_pre, average='macro')

        train_status = f'''---Train Status(Epoch: {epoch})---
        Accuracy: {train_acc:.6f}
        F1(weighted): {train_F1_weighted:.6f}
        R(weighted): {train_R_weighted:.6f}
        Precision(weighted): {train_precision_weighted:.6f}
        F1(micro): {train_F1_micro:.6f}
        R(micro): {train_R_micro:.6f}
        Precision(micro): {train_precision_micro:.6f}
        F1(macro): {train_F1_macro:.6f}
        R(macro): {train_R_macro:.6f}
        Precision(macro): {train_precision_macro:.6f}
        loss: {run_loss:.6f}\n'''
        print(train_status)
        write_to_file(os.path.join(args.save_model, 'train_log.txt'), train_status)

        summary_writer.add_scalar('Train Info/Run Loss', run_loss, global_step=epoch)
        summary_writer.add_scalar('Train Info/Accuracy', train_acc, global_step=epoch)
        summary_writer.add_scalar('Train Info/F1(weighted)', train_F1_weighted, global_step=epoch)
        summary_writer.add_scalar('Train Info/R(weighted)', train_R_weighted, global_step=epoch)
        summary_writer.add_scalar('Train Info/Precision(weighted)', train_precision_weighted, global_step=epoch)
        summary_writer.add_scalar('Train Info/Learning Rate', optimizer.param_groups[0]['lr'], global_step=epoch)
        summary_writer.flush()

        # Validate the current model
        val_acc , val_F1_weighted = dev(args, dev_loader, msaf, epoch, summary_writer)
        test_acc , test_F1_marco = test(args, test_loader, msaf, epoch, summary_writer)
        if test_acc > best_acc:
            best_f1 = val_F1_weighted 
            best_acc = val_acc
            best_epoch = epoch
            for filename in os.listdir(args.save_model):
                if filename.endswith('.pth'):
                    os.remove(os.path.join(args.save_model, filename))
            save_path = os.path.join(args.save_model, f'model_weights_{best_epoch}_{best_acc}.pth')
            torch.save(msaf.state_dict(), save_path)
        

        scheduler.step()

    summary_writer.add_scalar('Train Info/Learning Rate', optimizer.param_groups[0]['lr'], global_step=epoch)
    summary_writer.flush()
    summary_writer.close()
