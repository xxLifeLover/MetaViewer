import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from torch.nn.functional import normalize
from methods.backbones import NNEncoder, NNDecoder


def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
    if isinstance(m, nn.Linear) and m.bias is not None:
        nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Conv2d):
        trunc_normal_(m.weight, std=.02)
    if isinstance(m, nn.Conv2d) and m.bias is not None:
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_maxpool=True, kernel_size=5, padding=2):
        super(ConvBlock, self).__init__()
        self.use_maxpool = use_maxpool
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2) if use_maxpool else nn.Identity()

    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))


class RWConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        self.groups = groups
        super(RWConv1d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)

    def forward(self, input_x):
        [x, v] = input_x
        weights = self.weight[:, v, :]
        y = nn.functional.conv1d(
            x, weights, self.bias, self.stride, self.padding,
            self.dilation, self.groups)
        return y


class RWConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_maxpool=True, kernel_size=5, padding=2):
        super(RWConvBlock, self).__init__()
        self.use_maxpool = use_maxpool
        self.conv = RWConv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2) if use_maxpool else nn.Identity()

    def forward(self, x, v):
        return self.pool(self.relu(self.bn(self.conv([x, v]))))


class MetaNet(nn.Module):
    def __init__(self,
                 args,
                 encoder_dim,
                 activation='relu',
                 batchnorm=True):
        super(MetaNet, self).__init__()
        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm

        padding_dicts = {'1': 0, '3': 1, '5': 2, '7': 3, '9': 4}
        kernel_size = args.meta_kernels
        padding = padding_dicts[str(kernel_size)]

        self.conv_first = RWConvBlock(encoder_dim[0], encoder_dim[1], use_maxpool=False, kernel_size=kernel_size, padding=padding)
        self.conv_last = ConvBlock(encoder_dim[-1], 1, use_maxpool=False, kernel_size=kernel_size, padding=padding)
        encoder_layers = []
        for i in range(1, self._dim):
            encoder_layers.append(ConvBlock(encoder_dim[i], encoder_dim[i+1], use_maxpool=False, kernel_size=kernel_size, padding=padding))
        self._encoder = nn.Sequential(*encoder_layers)

    def forward(self, x, v):
        x = self.conv_first(x, v)
        x = self._encoder(x)
        h = self.conv_last(x)
        return h


class MetaViewer(nn.Module):
    def __init__(self, args, sample_data):
        super().__init__()
        self.args = args
        self.batch_size = sample_data[0].shape[0]
        self.num_views = len(sample_data)

        # AutoEncoder (encoder s_enc: embedding module, decoder s_dec: reconstruction head)
        self.s_enc = torch.nn.ModuleList()
        self.s_dec = torch.nn.ModuleList()
        for v in range(self.num_views):
            view_shape = sample_data[v].shape
            self.s_enc.append(
                NNEncoder(args=self.args, view_shape=view_shape, v=v))
            self.s_dec.append(
                NNDecoder(args=self.args, view_shape=view_shape, v=v))
        # MetaNet: MetaViewer
        meta_channel = args.meta_channels
        meta_channel[0] = self.num_views
        self.meta_net = MetaNet(args, meta_channel)

        self.apply(_init_vit_weights)
        self.lossF_mse = torch.nn.MSELoss(reduction='sum')

    def forward_base(self, x, v):
        x = x.squeeze()
        x_emb = self.s_enc[v](x)
        x_meta = self.meta_net(normalize(x_emb, dim=1).unsqueeze(dim=1), [v])
        x_rec = self.s_dec[v](x_meta.squeeze())
        return [x, x_rec]

    def loss_base(self, logits):
        [x, x_rec] = logits
        loss_rec = 0.5 * self.lossF_mse(x_rec, x)
        return loss_rec

    def forward_meta(self, data, views):
        data_embs = []
        data = [data[v].squeeze() for v in range(len(data))]
        for v in range(self.num_views):
            data_embs.append(self.s_enc[v](data[v]))

        data_embs_cat = torch.transpose(torch.stack([normalize(emb, dim=1) for emb in data_embs]), 1, 0)
        metaviews = self.meta_net(data_embs_cat, views).squeeze()
        data_recs = []
        for v in range(len(views)):
            data_recs.append(self.s_dec[v](metaviews))
        return [data, data_embs, metaviews, data_recs]

    def loss_meta(self, logits):
        [data, data_embs, metaviews, data_recs] = logits
        loss_rec = 0.0
        for v in range(len(data)):
            loss_rec += 0.5 * self.lossF_mse(data_recs[v], data[v])
        return loss_rec
