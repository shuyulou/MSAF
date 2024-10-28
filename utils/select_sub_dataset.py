import os
import json
import random

def select_sub_dataset():
    """
    To improve the training efficiency, select 9000 pieces of data from RU_Senti,
    form the new dataset RU_Senti_tiny
    """
    text_path = os.path.join('datasets', 'RU_senti', 'RU_senti.json')

    # 10 months from March to December
    data_list = [[[] for _ in range(3)] for _ in range(10)]
    with open(text_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            time = data['date']
            month =  int(time.split('-')[1])
            emo = int(data['Rating'])
            data_list[month - 3][emo - 1].append(data)

    # select 300 pieces of data per rating per month
    selected_data_list = []
    for row in data_list:
        for element in row:
            selected_data_list.extend(random.sample(element, 300))
    print(f'length for RU_senti_tiny: {len(selected_data_list)}')

    with open(os.path.join('datasets', 'RU_senti_tiny', 'RU_senti_tiny.json'), "w", encoding='utf-8') as file:
        for selected_data in selected_data_list:
            file.write(json.dumps(selected_data) + '\n')


if __name__ == '__main__':
    select_sub_dataset()
