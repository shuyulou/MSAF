import json
import os
import random
from transformers import BertTokenizer
import matplotlib.pyplot as plt


def print_distribution(dataset):
    train_path = os.path.join('datasets', dataset, 'train.json')
    dev_path = os.path.join('datasets', dataset, 'dev.json')
    test_path = os.path.join('datasets', dataset, 'test.json')
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    text_lengths = []
    with open(train_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for item in data:
            text = item['text']
            text_token_list = [tokenizer.tokenize('[CLS]' + text + '[SEP]')]
            text_lengths.append(len(text_token_list[0]))
    
    with open(dev_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for item in data:
            text = item['text']
            text_token_list = [tokenizer.tokenize('[CLS]' + text + '[SEP]')]
            text_lengths.append(len(text_token_list[0]))

    with open(test_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for item in data:
            text = item['text']
            text_token_list = [tokenizer.tokenize('[CLS]' + text + '[SEP]')]
            text_lengths.append(len(text_token_list[0]))
    
    plt.figure(figsize=(8, 6))
    plt.hist(text_lengths, bins=10, color='skyblue', edgecolor='black')
    plt.xlabel('Sentence Length')
    plt.ylabel('Frequency')
    plt.title('Histogram of Sentence Length')
    plt.grid(True)
    plt.show()


def RU_senti_tiny_process():
    json_file = os.path.join('datasets', 'RU_senti_tiny', 'RU_senti_tiny.json')
    all_data = []
    with open(json_file, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            photo_id = random.choice(data['Photos'])['_id']
            text = data['Text']
            label = int(data['Rating']) - 1
            all_data.append({"photo_id": photo_id, "text": text, "label": label})

    random.shuffle(all_data)

    num_total_samples = len(all_data)
    num_train_samples = int(0.8 * num_total_samples)
    num_dev_samples = int(0.1 * num_total_samples)

    train_set = all_data[:num_train_samples]
    dev_set = all_data[num_train_samples:num_train_samples + num_dev_samples]
    test_set = all_data[num_train_samples + num_dev_samples:]

    train_path = os.path.join('datasets', 'RU_senti_tiny', 'train.json')
    dev_path = os.path.join('datasets', 'RU_senti_tiny', 'dev.json')
    test_path = os.path.join('datasets', 'RU_senti_tiny', 'test.json')
    with open(train_path, 'w', encoding='utf-8') as file:
        json.dump(train_set, file, ensure_ascii=False, indent=4)
    with open(dev_path, 'w', encoding='utf-8') as file:
        json.dump(dev_set, file, ensure_ascii=False, indent=4)
    with open(test_path, 'w', encoding='utf-8') as file:
        json.dump(test_set, file, ensure_ascii=False, indent=4)


def RU_senti_process():
    json_file = os.path.join('datasets', 'RU_senti', 'RU_senti.json')
    all_data = []
    with open(json_file, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            photo_id = random.choice(data['Photos'])['_id']
            text = data['Text']
            label = int(data['Rating']) - 1
            all_data.append({"photo_id": photo_id, "text": text, "label": label})

    random.shuffle(all_data)

    num_total_samples = len(all_data)
    num_train_samples = int(0.8 * num_total_samples)
    num_dev_samples = int(0.1 * num_total_samples)

    train_set = all_data[:num_train_samples]
    dev_set = all_data[num_train_samples:num_train_samples + num_dev_samples]
    test_set = all_data[num_train_samples + num_dev_samples:]

    train_path = os.path.join('datasets', 'RU_senti', 'train.json')
    dev_path = os.path.join('datasets', 'RU_senti', 'dev.json')
    test_path = os.path.join('datasets', 'RU_senti', 'test.json')
    with open(train_path, 'w', encoding='utf-8') as file:
        json.dump(train_set, file, ensure_ascii=False, indent=4)
    with open(dev_path, 'w', encoding='utf-8') as file:
        json.dump(dev_set, file, ensure_ascii=False, indent=4)
    with open(test_path, 'w', encoding='utf-8') as file:
        json.dump(test_set, file, ensure_ascii=False, indent=4)


def TumEmo_process():
    emotion_dict = {'Angry': 0, 'Bored': 1, 'Calm': 2, 'Fear': 3, 'Happy': 4, 'Love': 5, 'Sad': 6}
    label_list = []
    tum_file = os.path.join('datasets', 'TumEmo', 'all_data_id_and_label.txt')
    with open(tum_file, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if i == 0:
                continue
            emo = emotion_dict[line.split()[1]]
            label_list.append(emo)

    all_data = []
    num_total_samples = len(label_list)
    for i in range(num_total_samples):
        photo_id = i
        label = label_list[i]
        text_path = os.path.join('datasets', 'TumEmo', 'data', f'{i}.txt')
        with open(text_path, 'r', encoding='utf-8') as file:
            text = file.read()
        all_data.append({"photo_id": photo_id, "text": text, "label": label})

    random.shuffle(all_data)

    num_train_samples = int(0.8 * num_total_samples)
    num_dev_samples = int(0.1 * num_total_samples)

    train_set = all_data[:num_train_samples]
    dev_set = all_data[num_train_samples:num_train_samples + num_dev_samples]
    test_set = all_data[num_train_samples + num_dev_samples:]

    train_path = os.path.join('datasets', 'TumEmo', 'train.json')
    dev_path = os.path.join('datasets', 'TumEmo', 'dev.json')
    test_path = os.path.join('datasets', 'TumEmo', 'test.json')
    with open(train_path, 'w', encoding='utf-8') as file:
        json.dump(train_set, file, ensure_ascii=False, indent=4)
    with open(dev_path, 'w', encoding='utf-8') as file:
        json.dump(dev_set, file, ensure_ascii=False, indent=4)
    with open(test_path, 'w', encoding='utf-8') as file:
        json.dump(test_set, file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # RU_senti_tiny_process()
    # print_distribution('RU_senti_tiny')
    RU_senti_process()
    print_distribution('RU_senti')
    # TumEmo_process()
    # print_distribution('TumEmo')
