# test
dataset=RU-Senti
save_model=./checkpoints/2024-08-03_14-54-04RU_senti_base
text_num_hidden_layers=12
fusion_num_hidden_layers=2
epoch=15
batch_size=16
lr=4e-5
lr_decay=0.7
dropout=0.4
text_length=80
weight_decay=0
temperature=0.1

python main.py -mode 2 -save_model ${save_model} -dataset ${dataset} -cuda -text_num_hidden_layers ${text_num_hidden_layers}\
        -fusion_num_hidden_layers ${fusion_num_hidden_layers} -epoch ${epoch} -batch_size ${batch_size}\
        -lr ${lr} -lr_decay ${lr_decay} -dropout ${dropout} -text_length ${text_length}\
        -weight_decay ${weight_decay} -temperature ${temperature}

