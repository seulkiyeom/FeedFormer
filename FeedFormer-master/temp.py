from mmseg.models.decode_heads.feedformer_head import FeedFormerHead
from mmseg.models.decode_heads.segformer_head import SegFormerHead

norm_cfg = dict(type='SyncBN', requires_grad=True)

head = FeedFormerHead(in_channels=[32, 64, 160, 256],
            in_index=[0, 1, 2, 3],
            feature_strides=[4, 8, 16, 32],
            channels=128, 
            dropout_ratio=0.1,
            num_classes=150,
            norm_cfg=norm_cfg,
            align_corners=False,
            decoder_params=dict(embed_dim=128),
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))