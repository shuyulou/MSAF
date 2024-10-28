import json
import os
import random


def shuffle_dataset(dataset):
    train_path = os.path.join('datasets', dataset, 'train.json')
    dev_path = os.path.join('datasets', dataset, 'dev.json')
    test_path = os.path.join('datasets', dataset, 'test.json')
    
    all_data = []
    with open(train_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for item in data:
            photo_id = item['photo_id']
            text = item['text']
            label = item['label']
            all_data.append({"photo_id": photo_id, "text": text, "label": label})
    
    with open(dev_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for item in data:
            photo_id = item['photo_id']
            text = item['text']
            label = item['label']
            all_data.append({"photo_id": photo_id, "text": text, "label": label})

    with open(test_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for item in data:
            photo_id = item['photo_id']
            text = item['text']
            label = item['label']
            all_data.append({"photo_id": photo_id, "text": text, "label": label})
    
    random.shuffle(all_data)

    num_total_samples = len(all_data)
    num_train_samples = int(0.8 * num_total_samples)
    num_dev_samples = int(0.1 * num_total_samples)

    train_set = all_data[:num_train_samples]
    dev_set = all_data[num_train_samples:num_train_samples + num_dev_samples]
    test_set = all_data[num_train_samples + num_dev_samples:]

    train_path = os.path.join('datasets', dataset, 'train.json')
    dev_path = os.path.join('datasets', dataset, 'dev.json')
    test_path = os.path.join('datasets', dataset, 'test.json')
    with open(train_path, 'w', encoding='utf-8') as file:
        json.dump(train_set, file, ensure_ascii=False, indent=4)
    with open(dev_path, 'w', encoding='utf-8') as file:
        json.dump(dev_set, file, ensure_ascii=False, indent=4)
    with open(test_path, 'w', encoding='utf-8') as file:
        json.dump(test_set, file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    shuffle_dataset('MVSA-Single')
