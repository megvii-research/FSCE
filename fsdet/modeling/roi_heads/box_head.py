# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from fsdet.layers import Conv2d, ShapeSpec, get_norm
from fsdet.utils.registry import Registry
from fsdet.modeling.backbone.resnet import BasicBlock, BottleneckBlock

ROI_BOX_HEAD_REGISTRY = Registry("ROI_BOX_HEAD")
ROI_BOX_HEAD_REGISTRY.__doc__ = """
Registry for box heads, which make box predictions from per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


@ROI_BOX_HEAD_REGISTRY.register()
class FastRCNNConvFCHead(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and
    several fc layers (each followed by relu).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv, num_fc: the number of conv/fc layers
            conv_dim/fc_dim: the dimension of the conv/fc layers
            norm: normalization for the conv layers
        """
        super().__init__()

        # fmt: off
        num_conv   = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim   = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc     = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim     = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        norm       = cfg.MODEL.ROI_BOX_HEAD.NORM
        # fmt: on
        assert num_conv + num_fc > 0

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        self.conv_norm_relus = []
        for k in range(num_conv):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not norm,
                norm=get_norm(norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        for k in range(num_fc):
            fc = nn.Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for layer in self.fcs:
                x = F.relu(layer(x))
        return x

    @property
    def output_size(self):
        return self._output_size


@ROI_BOX_HEAD_REGISTRY.register()
class FastRcnnNovelHead(nn.Module):
#     """
#     A head that has separate 1024 fc for regression and classification branch
#     """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv, num_fc: the number of conv/fc layers
            conv_dim/fc_dim: the dimension of the conv/fc layers
            norm: normalization for the conv layers
        """
        super().__init__()

        # fmt: off
        fc_dim        = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        sub_fc_dim    = cfg.MODEL.ROI_BOX_HEAD.SUB_FC_DIM
        norm          = cfg.MODEL.ROI_BOX_HEAD.NORM
        # fmt: on
        box_feat_shape = (input_shape.channels, input_shape.height, input_shape.width)

        self.fc_main = nn.Linear(np.prod(box_feat_shape), fc_dim)
        self.fc_reg = nn.Linear(fc_dim, sub_fc_dim)
        self.fc_cls = nn.Linear(fc_dim, sub_fc_dim)

        self._output_size = sub_fc_dim

        for layer in [self.fc_main, self.fc_reg, self.fc_cls]:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        main_feat = F.relu(self.fc_main(x))
        loc_feat = F.relu(self.fc_reg(main_feat))
        cls_feat = F.relu(self.fc_cls(main_feat))
        return loc_feat, cls_feat

    @property
    def output_size(self):
        return self._output_size


@ROI_BOX_HEAD_REGISTRY.register()
class FastRCNNDoubleHead(nn.Module):
    """
    Double Head as described in https://arxiv.org/pdf/1904.06493.pdf
    The Conv Head composed of 1 (BasicBlock) + x (BottleneckBlock) and average pooling
    for bbox regression. From config: num_conv = 1 + x
    The FC Head composed of 2 fc layers for classification.
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv, num_fc: the number of conv/fc layers
            conv_dim/fc_dim: the dimension of the conv/fc layers
            norm: normalization for the conv layers
        """
        super().__init__()

        # fmt: off
        num_conv   = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim   = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc     = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim     = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        norm       = cfg.MODEL.ROI_BOX_HEAD.NORM
        # fmt: on
        assert num_conv + num_fc > 0

        self.convs = []
        for k in range(num_conv):
            if k == 0:
                # import pdb; pdb.set_trace()
                conv = BasicBlock(input_shape.channels, conv_dim, norm=norm)
                # for name, param in conv.named_parameters():
                #     print(name, param.requires_grad)

                # bottleneck_channels = conv_dim // 4
                # conv = BottleneckBlock(input_shape.channels, conv_dim,
                #                        bottleneck_channels=bottleneck_channels, norm=norm)
                # import pdb; pdb.set_trace()
                # for name, param in conv.named_parameters():
                #     print(name, param)
            else:
                bottleneck_channels = conv_dim // 4
                conv = BottleneckBlock(conv_dim, conv_dim,
                                       bottleneck_channels=bottleneck_channels, norm=norm)
            self.add_module("conv{}".format(k + 1), conv)
            self.convs.append(conv)
        # this is a @property, see line 153, will be used as input_size for box_predictor
        # here when this function return, self._output_size = fc_dim (=1024)
        self._output_size = input_shape.channels * input_shape.height * input_shape.width
        self.fcs = []
        for k in range(num_fc):
            fc = nn.Linear(self._output_size, fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        # init has already been done in BasicBlock and BottleneckBlock
        # for layer in self.conv_norm_relus:
        #     weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        loc_feat = x
        for layer in self.convs:
            loc_feat = layer(loc_feat)

        loc_feat = F.adaptive_avg_pool2d(loc_feat, (1,1))
        loc_feat = torch.flatten(loc_feat, start_dim=1)

        cls_feat = torch.flatten(x, start_dim=1)
        for layer in self.fcs:
            cls_feat = F.relu(layer(cls_feat))
        return loc_feat, cls_feat

    @property
    def output_size(self):
        return self._output_size


def build_box_head(cfg, input_shape):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_BOX_HEAD.NAME
    return ROI_BOX_HEAD_REGISTRY.get(name)(cfg, input_shape)
