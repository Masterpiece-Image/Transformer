import torch
import torch.nn as nn
import InputProj
import OutputProj

from TransformerModule import *
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


## Creating num WTM block and one STM block if USE_CROSS
class BasicIDTLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, win_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 k_bound=8, num=2, USE_CROSS=True):
        super(BasicIDTLayer, self).__init__()

        #0 - INIT
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        blocks = []
        #1 - Creating all the WTM blocks
        for i in range(num):
            blocks.append(WTM(dim=dim, input_resolution=input_resolution,
                                  num_heads=num_heads, win_size=win_size,
                                  shift_size=0,
                                  mlp_ratio=mlp_ratio,
                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                                  drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                                  norm_layer=norm_layer))

        #2 - If USE_CROSS is define, we add a last STM layer
        if USE_CROSS:
            blocks.append(SpatialTransformerLayer(input_resolution, num_heads, channel=dim,
                                mlp_ratio=mlp_ratio,
                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                                  drop_path=drop_path[-1] if isinstance(drop_path, list) else drop_path,
                                  norm_layer=norm_layer, k_bound=k_bound))
        # build blocks
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, mask=None):
        #If checkpoints are activated, then we save the states of the blocks, else we just continue
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, mask)
        return x




class IDT(nn.Module):
    def __init__(self, img_size=128, in_chans=3,
                 embed_dim=32, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2], num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, dowsample=Downsample, upsample=Upsample):
        super(IDT, self).__init__()

        #0 - Init
        self.num_enc_layers = len(depths) // 2
        self.num_dec_layers = len(depths) // 2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.win_size = win_size
        self.reso = img_size

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate] * depths[4]
        dec_dpr = enc_dpr[::-1]

        # build layers
        # Input/Output
        self.input_proj = InputProj(in_channel=in_chans, out_channel=embed_dim, kernel_size=3, stride=1,
                                    act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=2 * embed_dim, out_channel=in_chans, kernel_size=3, stride=1)

        # Encoder
        # check Fig 1.
        # 4 first blocks are encoders
        # 1 block of 'transition' is the Bottleneck
        # 4 others blocks of decoders
        # all blocks are similar, just the dimension and resolution are modified
        self.encoderlayer_0 = BasicIDTLayer(dim=embed_dim,
                                                input_resolution=(img_size,
                                                                  img_size),
                                                depth=depths[0],
                                                num_heads=num_heads[0],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint, k_bound=16, num=3)
        self.dowsample_0 = dowsample(embed_dim, embed_dim * 2)

        self.encoderlayer_1 = BasicIDTLayer(dim=embed_dim * 2,
                                                input_resolution=(img_size // 2,
                                                                  img_size // 2),
                                                depth=depths[1],
                                                num_heads=num_heads[1],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint, k_bound=8, num=3)
        self.dowsample_1 = dowsample(embed_dim * 2, embed_dim * 4)
        self.encoderlayer_2 = BasicIDTLayer(dim=embed_dim * 4,
                                                input_resolution=(img_size // (2 ** 2),
                                                                  img_size // (2 ** 2)),
                                                depth=depths[2],
                                                num_heads=num_heads[2],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint, k_bound=4, num=2)
        self.dowsample_2 = dowsample(embed_dim * 4, embed_dim * 8)
        self.encoderlayer_3 = BasicIDTLayer(dim=embed_dim * 8,
                                                input_resolution=(img_size // (2 ** 3),
                                                                  img_size // (2 ** 3)),
                                                depth=depths[3],
                                                num_heads=num_heads[3],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop_path=enc_dpr[sum(depths[:3]):sum(depths[:4])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint, k_bound=2, num=2)
        self.dowsample_3 = dowsample(embed_dim * 8, embed_dim * 16)

        # Bottleneck
        # No STM in this layer
        self.bottle = BasicIDTLayer(dim=embed_dim * 16,
                                      input_resolution=(img_size // (2 ** 4),
                                                        img_size // (2 ** 4)),
                                      depth=depths[4],
                                      num_heads=num_heads[4],
                                      win_size=win_size,
                                      mlp_ratio=self.mlp_ratio,
                                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                                      drop_path=conv_dpr,
                                      norm_layer=norm_layer,
                                      use_checkpoint=use_checkpoint,
                                      k_bound=1, num=1, USE_CROSS=False)

        # Decoder
        self.upsample_0 = upsample(embed_dim * 16, embed_dim * 8)
        self.decoderlayer_0 = BasicIDTLayer(dim=embed_dim * 16,
                                                input_resolution=(img_size // (2 ** 3),
                                                                  img_size // (2 ** 3)),
                                                depth=depths[5],
                                                num_heads=num_heads[5],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop_path=dec_dpr[:depths[5]],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint, k_bound=2, num=1)
        self.upsample_1 = upsample(embed_dim * 16, embed_dim * 4)
        self.decoderlayer_1 = BasicIDTLayer(dim=embed_dim * 8,
                                                input_resolution=(img_size // (2 ** 2),
                                                                  img_size // (2 ** 2)),
                                                depth=depths[6],
                                                num_heads=num_heads[6],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop_path=dec_dpr[sum(depths[5:6]):sum(depths[5:7])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint, k_bound=4, num=2)

        self.upsample_2 = upsample(embed_dim * 8, embed_dim * 2)
        self.decoderlayer_2 = BasicIDTLayer(dim=embed_dim * 4,
                                                input_resolution=(img_size // 2,
                                                                  img_size // 2),
                                                depth=depths[7],
                                                num_heads=num_heads[7],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop_path=dec_dpr[sum(depths[5:7]):sum(depths[5:8])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint, k_bound=8, num=2)

        self.upsample_3 = upsample(embed_dim * 4, embed_dim)
        self.decoderlayer_3 = BasicIDTLayer(dim=embed_dim * 2,
                                                input_resolution=(img_size,
                                                                  img_size),
                                                depth=depths[8],
                                                num_heads=num_heads[8],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop_path=dec_dpr[sum(depths[5:8]):sum(depths[5:9])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint, k_bound=16, num=3)

        self.apply(self._init_weights)



    #Init of the weights
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x, mask=None):
        # Input Projection
        y = self.input_proj(x)
        # Encoder
        encoder0 = self.encoderlayer_0(y, mask=mask) #128x128  32
        pool0 = self.dowsample_0(encoder0)

        encoder1 = self.encoderlayer_1(pool0, mask=mask) #64x64 64
        pool1 = self.dowsample_1(encoder1)

        encoder2 = self.encoderlayer_2(pool1, mask=mask) #32x32 128
        pool2 = self.dowsample_2(encoder2)

        encoder3 = self.encoderlayer_3(pool2, mask=mask) #16x16 256
        pool3 = self.dowsample_3(encoder3)

        # Bottleneck
        bottle = self.bottle(pool3, mask=mask) #8x8 512

        # Decoder
        up0 = self.upsample_0(bottle) #16x16 256
        decoder0 = torch.cat([up0, encoder3], -1) #16x16 512
        decoder0 = self.decoderlayer_0(decoder0, mask=mask) #16x16 512

        up1 = self.upsample_1(decoder0) #32x32 128
        decoder1 = torch.cat([up1, encoder2], -1) #32x32 256
        decoder1 = self.decoderlayer_1(decoder1, mask=mask) #32x32 256

        up2 = self.upsample_2(decoder1) #64x64 64
        decoder2 = torch.cat([up2, encoder1], -1) #64x64 128
        decoder2 = self.decoderlayer_2(decoder2, mask=mask) #64x64 128

        up3 = self.upsample_3(decoder2) #128x128 32
        decoder3 = torch.cat([up3, encoder0], -1) #128x128 64
        decoder3 = self.decoderlayer_3(decoder3, mask=mask)

        # Output Projection
        y = self.output_proj(decoder3)
        return x + y
