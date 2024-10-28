dataset=MVSA-Multiple
text_num_hidden_layers=12
fusion_num_hidden_layers=4
epoch=15
batch_size=16 #16
lr=2e-5
lr_decay=0.70
dropout=0.4
text_length=80
weight_decay=0 # 1e-3
temperature=0.1
acc_steps=4 #2

CUDA_VISIBLE_DEVICES=3 python main.py -dataset ${dataset} -cuda -text_num_hidden_layers ${text_num_hidden_layers}\
                                       -fusion_num_hidden_layers ${fusion_num_hidden_layers} -epoch ${epoch} -batch_size ${batch_size}\
                                       -lr ${lr} -lr_decay ${lr_decay} -dropout ${dropout} -text_length ${text_length}\
                                       -weight_decay ${weight_decay} -temperature ${temperature} -acc_steps ${acc_steps}
