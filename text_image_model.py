import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig
from swin_transformer import SwinTransformer
from xbert import BertModel
from scipy.optimize import linear_sum_assignment


class TextEncoder(nn.Module):
    def __init__(self, args, pretrained_path=None):
        super(TextEncoder, self).__init__()
        self.bert_config = BertConfig.from_pretrained('bert-base-uncased')
        self.bert_config.num_hidden_layers = args.text_num_hidden_layers
        self.bert_model = None
        if pretrained_path:
            self.bert_model = BertModel(config=self.bert_config)
            pretrained_dict = torch.load(pretrained_path, map_location='cuda' if args.cuda else 'cpu')
            self.bert_model.load_state_dict(pretrained_dict)
        else:
            self.bert_model = BertModel.from_pretrained('bert-base-uncased', config=self.bert_config)

    def forward(self, text_ids, text_masks):
        text_output = self.bert_model(text_ids, attention_mask=text_masks)
        text_embed = text_output.last_hidden_state
        text_pooler = text_output.pooler_output
        return text_embed, text_pooler


class ImageEncoder(nn.Module):
    def __init__(self, args, pretrained_path='swin_tiny_patch4_window7_224.pth'):
        super(ImageEncoder, self).__init__()
        print(pretrained_path)
        pretrained_dict = torch.load(pretrained_path, map_location='cuda' if args.cuda else 'cpu')
        if pretrained_path == 'swin_tiny_patch4_window7_224.pth':
            pretrained_dict = pretrained_dict['model']
        self.image_model = SwinTransformer()
        self.image_model.load_state_dict(pretrained_dict, strict=False)

    def forward(self, images):
        image_embed, image_pooler, image_layer_output, layer_0_output = self.image_model(images)
        self.xi = image_layer_output[1]
        self.cu = image_layer_output[3]
        return image_embed, image_pooler, image_layer_output, layer_0_output


class FusionModel(nn.Module):
    def __init__(self, args, pretrained_path=None):
        super(FusionModel, self).__init__()
        self.fusion_config = BertConfig.from_pretrained('bert-base-uncased')
        self.fusion_config.num_hidden_layers = args.fusion_num_hidden_layers
        self.fusion_config.is_decoder = True
        self.fusion_config.add_cross_attention = True
        # self.fusion_config.hidden_dropout_prob = 0.4  # 设置隐藏层的dropout率
        # self.fusion_config.attention_probs_dropout_prob = 0.4  # 设置注意力层的dropout率
        self.fusion_model = BertModel(config=self.fusion_config)
        if pretrained_path:
            pretrained_dict = torch.load(pretrained_path, map_location='cuda' if args.cuda else 'cpu')
            self.fusion_model.load_state_dict(pretrained_dict)

    def forward(self, text_embed, text_masks, image_embed, image_masks):
        fusion_output = self.fusion_model(inputs_embeds=text_embed,
                                          attention_mask=text_masks,
                                          encoder_hidden_states=image_embed,
                                          encoder_attention_mask = image_masks,
                                          return_dict=True)
        fusion_output = fusion_output.last_hidden_state
        return fusion_output

