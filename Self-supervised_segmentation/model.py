
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from dino.vision_transformer import VisionTransformer
import os
import dino.vision_transformer as vits
from torch.nn import functional as F
from functools import partial

class VisionTransformerForSimMIM(VisionTransformer):
    def __init__(self,interpolate_encoding = False,img_size = 224, **kwargs):
        super().__init__(**kwargs)

        # assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.img_size=img_size
        self._trunc_normal_(self.mask_token, std=.02)
        self.interpolate_encoding = interpolate_encoding

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def forward(self, x, mask):
        x = self.patch_embed(x)
        
        assert mask is not None
        B, L, _ = x.shape

        mask_token = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        
        if self.img_size[0] != 224: #and self.interpolate_encoding
            x  = x  + self.interpolate_pos_encoding(x , self.img_size[0], self.img_size[0])
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        # rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x) #, rel_pos_bias=rel_pos_bias)
        x = self.norm(x)

        x = x[:, 1:]
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x

class MIM(nn.Module):
    def __init__(self, encoder, encoder_stride):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder.num_features,
                out_channels=self.encoder_stride ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),
        )

        self.in_chans = 3
        self.patch_size = 8

    def forward(self, x, mask):
        z = self.encoder(x, mask)
        x_rec = self.decoder(z)
        mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).unsqueeze(1).contiguous()
        loss_recon = F.l1_loss(x, x_rec, reduction='none')
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        return loss, x_rec, mask

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}

def build_model(args):

    encoder = VisionTransformerForSimMIM(
            patch_size=args.MODEL.PATCH_SIZE, 
            embed_dim=384, 
            depth=12, 
            num_heads=6, 
            mlp_ratio=4,
            img_size=[args.DATA.IMG_SIZE],
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            interpolate_encoding=True,
            )

    state_dict = get_state_dict(args)
    encoder.load_state_dict(state_dict, strict=False)

    return encoder

class VisionTransformerForFinetune(VisionTransformer):
    def __init__(self,interpolate_encoding = False,img_size = 224, **kwargs):
        super().__init__(**kwargs)

        # assert self.num_classes == 0
        self.img_size=img_size
        self.interpolate_encoding = interpolate_encoding

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def forward(self, x):
        x = self.patch_embed(x)
        
        B, L, _ = x.shape

        # w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
        # x = x * (1 - w) + mask_token * w

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        
        if self.img_size[0] != 224: #and self.interpolate_encoding
            x  = x  + self.interpolate_pos_encoding(x , self.img_size[0], self.img_size[0])
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        # rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x) #, rel_pos_bias=rel_pos_bias)
        x = self.norm(x)

        x = x[:, 1:]
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x
    

class LinearProbing(nn.Module):
    def __init__(self, encoder,encoder_stride, layer_num=1):
        super().__init__()
        self.encoder = encoder
        self.layer_num = layer_num
        self.encoder_stride = encoder_stride
        self.one_layer_decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder.num_features,
                out_channels=self.encoder_stride ** 2, kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),
        )
        self.two_layer_decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder.num_features,
                out_channels=self.encoder_stride ** 2 * 4,
                kernel_size=3, padding=1),
            nn.BatchNorm2d(self.encoder_stride ** 2 * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=self.encoder_stride ** 2 * 4,
                out_channels=self.encoder_stride ** 2,
                kernel_size=3, padding=1),
            nn.PixelShuffle(self.encoder_stride),
        )

    def forward(self, x):
        z = self.encoder(x)
        if self.layer_num == 2:
            x_rec = self.two_layer_decoder(z)
        else:
            x_rec = self.one_layer_decoder(z)
        return x_rec
def build_finetune_model(args):

    encoder = VisionTransformerForFinetune(
            patch_size=args.MODEL.PATCH_SIZE, 
            embed_dim=384, 
            depth=12, 
            num_heads=6, 
            mlp_ratio=4,
            img_size=[args.DATA.IMG_SIZE],
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            interpolate_encoding=True,
            )

    state_dict = get_state_dict(args)
    encoder.load_state_dict(state_dict, strict=False)

    return encoder


def get_state_dict(args):
    if os.path.isfile(args.PRETRAINED_WEIGHTS):
        state_dict = torch.load(args.PRETRAINED_WEIGHTS, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}  
        # msg = encoder.load_state_dict(state_dict)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.PRETRAINED_WEIGHTS, "")) #, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if args.MODEL.NAME == "vit_small" and args.MODEL.PATCH_SIZE == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif args.MODEL.NAME == "vit_small" and args.MODEL.PATCH_SIZE == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
        elif args.MODEL.NAME == "vit_base" and args.MODEL.PATCH_SIZE == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif args.MODEL.NAME == "vit_base" and args.MODEL.PATCH_SIZE == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
        else:
            print("There is no reference weights available for this model => We use random weights.")
    return state_dict

