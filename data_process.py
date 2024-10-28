import os
import math
import json
import torch
import numpy as np
from PIL import Image
from PIL import ImageFile
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
ImageFile.LOAD_TRUNCATED_IMAGES = True


class EmoDataset(Dataset):
    def __init__(self, args, dataset_split_type):
        # Data transformation pipeline
        self.train_transform = transforms.Compose(
            [
                transforms.RandAugment(num_ops=2, magnitude=14),
                transforms.Resize(self.calculate_resize_dimension(args.image_size)),
                transforms.CenterCrop(args.image_size),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                # Mean and std values are randomly sampled and calculated from the ImageNet dataset
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        self.val_test_transform = transforms.Compose(
            [
                transforms.Resize(self.calculate_resize_dimension(args.image_size)),
                transforms.CenterCrop(args.image_size),
                transforms.ToTensor(),
                # Mean and std values are randomly sampled and calculated from the ImageNet dataset
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        if dataset_split_type == 1:
            self.image_transform = self.train_transform
        else:
            self.image_transform = self.val_test_transform

        # read data
        self.data_path = os.path.join('datasets', args.dataset)
        self.photo_path = os.path.join(self.data_path, 'data')
        if dataset_split_type == 1:
            self.data_path = os.path.join(self.data_path, 'train.json')
        elif dataset_split_type == 2:
            self.data_path = os.path.join(self.data_path, 'dev.json')
        else:
            self.data_path = os.path.join(self.data_path, 'test.json')
        
        self.id_list = []
        self.text_list = []
        self.label_list = []
        with open(self.data_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for item in data:
                self.id_list.append(item['photo_id'])
                self.text_list.append(item['text'])
                self.label_list.append(item['label'])

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.text_token_list = [self.tokenizer.tokenize('[CLS]' + text + '[SEP]') for text in self.text_list]
        self.text_token_list = [text_token if len(text_token) < args.text_length else text_token[0: args.text_length - 1] + [text_token[-1]]
                                for text_token in self.text_token_list]
        self.text_id_list = [self.tokenizer.convert_tokens_to_ids(text_token) for text_token in self.text_token_list]

    def calculate_resize_dimension(self, image_size):
        return int(2**math.ceil(math.log2(image_size)))
    
    def __len__(self):
        return len(self.text_id_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.photo_path, f'{self.id_list[index]}.jpg')
        image_data = Image.open(image_path)
        image_data.load()
        if image_data.mode != 'RGB':
            image_data = image_data.convert('RGB')
        image_transform = self.image_transform(image_data)
        return self.text_id_list[index], image_transform, self.label_list[index]


class Collate():
    def __call__ (self, batch_data):
        text_ids = [torch.LongTensor(b[0]) for b in batch_data]
        # images = torch.FloatTensor(np.array([b[1] for b in batch_data]))
        images = torch.stack([b[1] for b in batch_data])
        labels = torch.LongTensor([b[2] for b in batch_data])

        text_lengths = [text_id.size(0) for text_id in text_ids]
        max_length = max(text_lengths)
        text_ids = pad_sequence(text_ids, batch_first=True, padding_value=0)

        text_masks = []
        image_masks = []
        for length in text_lengths:
            text_mask = [1] * length + [0] * (max_length - length)
            text_masks.append(text_mask)
            # 49 represents the features of the swin transformer output
            image_mask = [1] * 49
            image_masks.append(image_mask) 
        return text_ids, torch.LongTensor(text_masks), images, torch.LongTensor(image_masks), labels


def data_processing(args, dataset_split_type):
    """
    Parameters:
        - args (argparse.Namespace): Arguments containing configuration settings.
        - dataset_split_type (int): 1 represents the training set, 2 represents the development set, 
                              and 3 represents the test set.
    """
    emo_dataset = EmoDataset(args, dataset_split_type)
    data_loader = DataLoader(emo_dataset, batch_size=args.batch_size, shuffle=True if dataset_split_type == 1 else False,
                            collate_fn=Collate(), pin_memory=True if args.cuda else False, num_workers=32)  # add num_worker
    return data_loader, emo_dataset.__len__()
