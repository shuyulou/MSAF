import torch
import torch.nn as nn
import torch.nn.functional as F
from text_image_model import TextEncoder, ImageEncoder, FusionModel
from einops import rearrange, repeat
from torch import einsum

def exists(val):
    return val is not None
def default(val, d):
    return val if exists(val) else d
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)
class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim=None,
        dim_head=64,
        heads=8,
        parallel_ff=False,
        ff_mult=4,
        norm_context=False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        context_dim = default(context_dim, dim)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(context_dim) if norm_context else nn.Identity()

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether to have parallel feedforward

        ff_inner_dim = ff_mult * dim

        self.ff = nn.Sequential(
            nn.Linear(dim, ff_inner_dim * 2, bias=False),
            SwiGLU(),
            nn.Linear(ff_inner_dim, dim, bias=False)
        ) if parallel_ff else None

    def forward(self, x, context):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        # pre-layernorm, for queries and context

        x = self.norm(x)
        context = self.context_norm(context)

        # get queries

        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        # scale

        q = q * self.scale

        # get key / values

        k, v = self.to_kv(context).chunk(2, dim=-1)

        # query / key similarity

        sim = einsum('b h i d, b j d -> b h i j', q, k)

        # attention

        sim = sim - sim.amax(dim=-1, keepdim=True)
        attn = sim.softmax(dim=-1)

        # aggregate

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        # merge and combine heads

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        # add parallel feedforward (for multimodal layers)

        if exists(self.ff):
            out = out + self.ff(x)

        return out

class SentimentAnalysis(nn.Module):
    def __init__(self, args):
        super(SentimentAnalysis, self).__init__()
        self.num_labels = 3 + 1
        if args.dataset == 'HFM':
            self.num_labels = 2
        elif args.dataset == 'TumEmo':
            self.num_labels = 7

        if args.dataset == 'RU_senti':
            self.sentiment_analysis = nn.Sequential(
                nn.Dropout(args.dropout),
                nn.Linear(args.train_dim, args.train_dim // 2),
                nn.GELU(),
                nn.Linear(args.train_dim // 2, self.num_labels)
            )
        else:
            self.sentiment_analysis = nn.Sequential(
                nn.Linear(args.train_dim, args.train_dim),
                nn.GELU(),
                nn.Linear(args.train_dim, self.num_labels)
            )           

    def forward(self, fusion_output):
        cls_res = self.sentiment_analysis(fusion_output)
        return cls_res

class ContraryLoss(nn.Module):
    def __init__(self, gamma=1e-6):
        super(ContraryLoss, self).__init__()
        self.gamma = gamma  

    def forward(self, fnEin, fnEwr):
        
        vin = fnEin.view(fnEin.size(0), -1) 
        vwr = fnEwr.view(fnEwr.size(0), -1) 

        vin_norm = F.normalize(vin, p=2, dim=1) 
        vwr_norm = F.normalize(vwr, p=2, dim=1) 

        loss = torch.sum(vin_norm * vwr_norm, dim=1) 

        # 计算正则项
        lambda1 = torch.abs(torch.norm(vin_norm, p=2, dim=1) - 1)
        lambda2 = torch.abs(torch.norm(vwr_norm, p=2, dim=1) - 1)

        total_loss = torch.mean(loss + lambda1 + lambda2)

        return total_loss

class MSAF(nn.Module):
    def __init__(self, args):
        super(MSAF, self).__init__()
        self.train_dim = args.train_dim
        self.temperature = args.temperature
        self.cls_cross_entropy_loss = nn.CrossEntropyLoss()
        self.cl_cross_entropy_loss = nn.CrossEntropyLoss()

        self.text_encoder = TextEncoder(args, pretrained_path=None)
        self.image_encoder = ImageEncoder(args, pretrained_path='swin_tiny_patch4_window7_224.pth')
        self.fusion_model = FusionModel(args, pretrained_path=None)

        self.sentiment_analysis = SentimentAnalysis(args)
        self.contrary_loss = ContraryLoss()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fusion_ln = nn.LayerNorm(args.train_dim)
        self.fusiondr = nn.Dropout(p = 0.4)

        self.num_img_queries = 49
        self.img_queries = nn.Parameter(torch.randn(self.num_img_queries + 1, args.train_dim)) # num image queries for multimodal, but 1 extra CLS for contrastive learning
        self.img_attn_pool = CrossAttention(dim=args.train_dim, context_dim=768, dim_head=64, heads=8, norm_context=True)
        self.img_attn_pool_norm = LayerNorm(args.train_dim)

    def embed_image(self, image_tokens=None):
        img_queries = repeat(self.img_queries, 'n d -> b n d', b=image_tokens.shape[0])
        img_queries = self.img_attn_pool(img_queries, image_tokens)
        img_queries = self.img_attn_pool_norm(img_queries)

        return img_queries[:, 0], img_queries[:, 1:]

    def forward(self, text_ids, text_masks, images, image_masks, labels):
        device = text_ids.device
        text_embed, _ = self.text_encoder(text_ids, text_masks)
        _, _, image_layer_output, layer_0_output = self.image_encoder(images)

        cu_feature, cu_embed = self.embed_image(image_tokens=image_layer_output[0])
        xi_feature, xi_embed = self.embed_image(image_tokens=image_layer_output[3])

        image_masks_ad = torch.ones(1, self.num_img_queries).to(device) 
        cu_fusion_outputs = self.fusion_model(text_embed, text_masks, cu_embed, image_masks_ad)
        xi_fusion_outputs = self.fusion_model(cu_fusion_outputs, text_masks, xi_embed, image_masks_ad)[:,0,:] 

        xi_fusion_feature = self.fusiondr(self.fusion_ln(xi_fusion_outputs))
        cls_res = self.sentiment_analysis(xi_fusion_feature) 
        _, pre = torch.max(cls_res, 1)
        cls_loss = self.cls_cross_entropy_loss(cls_res, labels)

        text_feature = F.normalize(text_embed[:,0,:],dim=-1) 

        targets = torch.arange(cu_feature.size(0)).to(device)
        sim_cx = torch.matmul(cu_feature, xi_feature.T) / self.temperature
        sim_xc = torch.matmul(xi_feature, cu_feature.T) / self.temperature           
        cl_loss = F.cross_entropy(sim_cx, targets) + F.cross_entropy(sim_xc, targets)

        ct_loss = self.contrary_loss(cu_embed, xi_embed)

        sim_ti = torch.matmul(text_feature, xi_feature.T) / self.temperature
        it_loss = F.cross_entropy(sim_ti, targets)

        return pre, cls_loss, cl_loss, ct_loss, it_loss, cls_res
