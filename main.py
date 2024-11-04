import argparse
import os
from train import train
from test import test
# from heatmap import heatmap
from datetime import datetime
from utils.write import write_to_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=int, default=1, choices=[1, 2, 3], 
                        help='Specify the mode - 1 for training, 2 for testing')
    parser.add_argument('-dataset', type=str, default='MVSA-Single',
                        choices=['MVSA-Single', 'MVSA-Multiple', 'RU_senti', 'HFM'],
                        help='Dataset selection: MVSA-Single, MVSA-Multiple, RU_senti, HFM')
    parser.add_argument('-save_model', type=str, default=os.path.join('checkpoints', datetime.now().strftime('%Y-%m-%d_%H-%M-%S')), 
                        help='Path to save the models')
    parser.add_argument('-cuda', action='store_true', default=True, 
                        help='Use CUDA for computation (if True) or CPU (if False)')
    parser.add_argument('-text_num_hidden_layers', type=int, default=12, help='The number of hidden layers in text model')
    parser.add_argument('-fusion_num_hidden_layers', type=int, default=12, help='The number of hidden layers in fusion model')
    parser.add_argument('-epoch', type=int, default=30, help='Number of training epochs')
    parser.add_argument('-batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('-lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('-lr_decay', type=float, default=0.95, help='Exponential decay rate')
    parser.add_argument('-dropout', type=float, default=0.1, help='Dropout ratio')
    parser.add_argument('-num_workers', type=int, default=8, help='The num_workers parameter for DataLoader')
    parser.add_argument('-image_size', type=int, default=224, help='Preprocessed image size')
    parser.add_argument('-text_length', type=int, default=100, help='Maximum number of words allowed in a sentence')
    parser.add_argument('-train_dim', type=int, default=768, help='Dimension of input for fusion transformer')
    parser.add_argument('-adamw_beta1', type=float, default=0.9, help='Beta1 for AdamW optimizer')
    parser.add_argument('-adamw_beta2', type=float, default=0.999, help='Beta2 for AdamW optimizer')
    parser.add_argument('-weight_decay', type=float, default=0.01, help='Weight decay for AdamW optimizer')
    parser.add_argument('-temperature', type=float, default=0.07, help='Temperature used for contrastive learning')
    parser.add_argument('-acc_steps', type=int, default=4, help='Accumulation steps')

    args = parser.parse_args()

    if args.mode == 1:
        args.save_model += args.dataset
    if not os.path.exists(args.save_model):
        os.mkdir(args.save_model)

    if args.mode == 1:
        print('Training mode is selected')
        write_to_file(os.path.join(args.save_model, 'train_log.txt'), f'{str(args)}\n\n')
        train(args)
    elif args.mode == 2:
        print('Testing mode is selected')
        test(args)
    # elif args.mode == 3:
    #     print('Heatmap mode is selected')
    #     heatmap(args)

