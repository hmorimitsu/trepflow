import torch
import torch.nn as nn
import torch.nn.functional as F

from .next import NeXtStage
from .repnext import RepNeXtStage
from .repvgg import RepConv2d
from .utils import get_norm


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, info_pred=False):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        out_ch = 6 if info_pred else 2
        self.conv2 = nn.Conv2d(hidden_dim, out_ch, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class NeXTDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        input_dim=192 + 128,
        kernel_size=7,
        next_num_conv_dw_layers=1,
        next_use_conv_dw_relu=True,
        next_dilation=1,
        next_use_act_relu=True,
        depth=1,
        mlp_ratio=4.0,
        norm_layer=None,
        mlp_use_norm=False,
        mlp_middle_conv=False,
        mlp_in_kernel_size=1,
        mlp_out_kernel_size=1,
        rep=False,
        use_residual_shortcut=True,
        use_layer_scale=True,
        use_ms_bn=False,
        bn_num_stats=1,
        use_tanh=True,
        deploy=False,
    ):
        super(NeXTDecoder, self).__init__()
        self.rep = rep
        self.use_tanh = use_tanh

        in_ch = hidden_dim + input_dim
        if rep:
            self.conv = RepNeXtStage(
                in_ch,
                hidden_dim,
                kernel_sizes=(kernel_size, 1, 0),
                stride=1,
                depth=depth,
                mlp_ratio=mlp_ratio,
                mlp_middle_conv=mlp_middle_conv,
                mlp_in_kernel_size=mlp_in_kernel_size,
                mlp_out_kernel_size=mlp_out_kernel_size,
                num_conv_dw_layers=next_num_conv_dw_layers,
                use_conv_dw_relu=next_use_conv_dw_relu,
                use_residual_shortcut=use_residual_shortcut,
                use_layer_scale=use_layer_scale,
                use_ms_bn=use_ms_bn,
                bn_num_stats=bn_num_stats,
                use_act_relu=next_use_act_relu,
                deploy=deploy,
            )
        else:
            self.conv = NeXtStage(
                in_ch,
                hidden_dim,
                kernel_size=kernel_size,
                stride=1,
                dilation=next_dilation,
                depth=depth,
                norm_layer=norm_layer,
                mlp_use_norm=mlp_use_norm,
                mlp_ratio=mlp_ratio,
                mlp_middle_conv=mlp_middle_conv,
                mlp_in_kernel_size=mlp_in_kernel_size,
                mlp_out_kernel_size=mlp_out_kernel_size,
                num_conv_dw_layers=next_num_conv_dw_layers,
                use_conv_dw_relu=next_use_conv_dw_relu,
                use_residual_shortcut=use_residual_shortcut,
                use_layer_scale=use_layer_scale,
                use_act_relu=next_use_act_relu,
            )

    def forward(self, h, x, it):
        inp = torch.cat([h, x], dim=1)
        if self.rep:
            h = self.conv(inp, istat=it)
        else:
            h = self.conv(inp)
        if self.use_tanh:
            h = torch.tanh(h)
        return h


class MotionEncoder(nn.Module):
    def __init__(self, args):
        super(MotionEncoder, self).__init__()

        c_hidden = args.dec_motenc_corr_hidden_chs
        c_out = args.dec_motenc_corr_out_chs
        f_hidden = args.dec_motenc_flow_hidden_chs
        f_out = args.dec_motenc_flow_out_chs

        cor_planes = args.corr_levels * (2 * args.corr_range + 1) ** 2
        self.convc1 = nn.Conv2d(cor_planes, c_hidden, 1, padding=0)
        self.convc2 = nn.Conv2d(c_hidden, c_out, 3, padding=1)
        self.convf1 = nn.Conv2d(
            2,
            f_hidden,
            args.dec_flow_kernel_size,
            padding=args.dec_flow_kernel_size // 2,
        )
        self.convf2 = nn.Conv2d(f_hidden, f_out, 3, padding=1)
        self.conv = nn.Conv2d(f_out + c_out, args.dec_motion_chs - 2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class RepMotionEncoder(nn.Module):
    def __init__(self, args):
        super(RepMotionEncoder, self).__init__()

        c_hidden = args.dec_motenc_corr_hidden_chs
        c_out = args.dec_motenc_corr_out_chs
        f_hidden = args.dec_motenc_flow_hidden_chs
        f_out = args.dec_motenc_flow_out_chs

        cor_planes = args.corr_levels * (2 * args.corr_range + 1) ** 2
        self.convc1 = RepConv2d(
            cor_planes,
            c_hidden,
            (1, 0),
            use_ms_bn=args.dec_use_ms_bn,
            bn_num_stats=args.iters,
            deploy=args.deploy,
        )
        self.convc2 = RepConv2d(
            c_hidden,
            c_out,
            (3, 1, 0),
            use_ms_bn=args.dec_use_ms_bn,
            bn_num_stats=args.iters,
            deploy=args.deploy,
        )
        self.convf1 = RepConv2d(
            2,
            f_hidden,
            (args.next_kernel_size, 1, 0),
            use_ms_bn=args.dec_use_ms_bn,
            bn_num_stats=args.iters,
            deploy=args.deploy,
        )
        self.convf2 = RepConv2d(
            f_hidden,
            f_out,
            (3, 1, 0),
            use_ms_bn=args.dec_use_ms_bn,
            bn_num_stats=args.iters,
            deploy=args.deploy,
        )
        self.conv = RepConv2d(
            f_out + c_out,
            args.dec_motion_chs - 2,
            (3, 1, 0),
            use_ms_bn=args.dec_use_ms_bn,
            bn_num_stats=args.iters,
            deploy=args.deploy,
        )

    def forward(self, flow, corr, it):
        cor = self.convc1(corr, istat=it)
        cor = self.convc2(cor, istat=it)
        flo = self.convf1(flow, istat=it)
        flo = self.convf2(flo, istat=it)

        cor_flo = torch.cat([cor, flo], dim=1)
        out = self.conv(cor_flo, istat=it)
        return torch.cat([out, flow], dim=1)


class UpdateBlock(nn.Module):
    def __init__(self, args):
        super(UpdateBlock, self).__init__()
        self.args = args

        if args.dec_use_ms_bn:
            assert args.use_rep_gru or args.motenc == "rep"

        if args.motenc == "base":
            encoder_fn = MotionEncoder
        elif args.motenc == "rep":
            encoder_fn = RepMotionEncoder

        self.encoder = encoder_fn(args)

        self.decoder = NeXTDecoder(
            hidden_dim=args.dec_net_chs,
            input_dim=args.dec_motion_chs + args.dec_inp_chs,
            kernel_size=args.next_kernel_size,
            next_num_conv_dw_layers=args.next_num_conv_dw_layers,
            next_use_conv_dw_relu=args.next_use_conv_dw_relu,
            next_dilation=args.next_dilation,
            next_use_act_relu=args.next_use_act_relu,
            depth=args.dec_depth,
            mlp_ratio=args.dec_mlp_ratio,
            norm_layer=get_norm(args.dec_norm),
            mlp_use_norm=args.mlp_use_norm,
            mlp_middle_conv=args.use_mlp_middle_conv,
            mlp_in_kernel_size=args.dec_mlp_in_kernel_size,
            mlp_out_kernel_size=args.dec_mlp_out_kernel_size,
            rep=args.use_rep_gru,
            use_residual_shortcut=args.dec_use_residual_shortcut,
            use_layer_scale=args.dec_use_layer_scale,
            use_ms_bn=args.dec_use_ms_bn,
            bn_num_stats=args.iters,
            use_tanh=args.net_use_tanh,
            deploy=args.deploy,
        )

        self.flow_head = FlowHead(
            args.dec_net_chs,
            hidden_dim=args.dec_flow_head_chs,
        )

        pred_stride = min(min(self.args.pyramid_ranges), 8)
        self.mask = nn.Sequential(
            nn.Conv2d(args.dec_net_chs, args.dec_net_chs * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(args.dec_net_chs * 2, pred_stride**2 * 9, 1, padding=0),
        )

    def forward(self, net, inp, corr, flow, get_mask=False, it=None):
        if self.args.motenc == "rep":
            motion_features = self.encoder(flow, corr, it=it)
        else:
            motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.decoder(net, inp, it=it)
        delta_flow = self.flow_head(net)

        mask = None
        if get_mask:
            mask = self.mask(net)

        return delta_flow, net, mask
