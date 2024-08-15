from .trepflow import TrepFlow


class TrepFlow_S(TrepFlow):
    def __init__(self, args):
        args.enc_rec = True
        args.enc_use_layer_scale = False
        args.use_rep_encoder = True
        args.dec_norm = "none"
        args.dec_flow_kernel_size = 3
        args.dec_use_layer_scale = False
        args.dec_mlp_out_kernel_size = 3
        args.next_kernel_size = 3
        args.next_num_conv_dw_layers = 3
        args.next_use_conv_dw_relu = False
        args.enc_use_ms_bn = True
        args.corr_range = 3
        args.dec_motion_chs = 96
        args.enc_mlp_ratio = 2
        args.dec_mlp_ratio = 2
        args.enc_hidden_chs = 72
        args.enc_out_chs = 2 * args.enc_hidden_chs
        args.enc_feat_chs = 2 * args.enc_out_chs // 3
        args.dec_inp_chs = args.enc_out_chs // 3
        args.dec_net_chs = args.enc_out_chs // 3
        args.dec_motenc_corr_hidden_chs = 192
        args.dec_motenc_corr_out_chs = 160
        args.dec_motenc_flow_hidden_chs = 96
        args.dec_motenc_flow_out_chs = 64
        args.dec_flow_head_chs = 192
        args.iters = 9
        super().__init__(args)


class TrepFlow_M(TrepFlow):
    def __init__(self, args):
        args.enc_rec = True
        args.enc_use_layer_scale = False
        args.use_rep_encoder = True
        args.dec_norm = "none"
        args.dec_flow_kernel_size = 3
        args.dec_use_layer_scale = False
        args.dec_mlp_out_kernel_size = 3
        args.next_kernel_size = 3
        args.next_num_conv_dw_layers = 3
        args.next_use_conv_dw_relu = False
        args.enc_use_ms_bn = True
        super().__init__(args)


class TrepFlow_L(TrepFlow):
    def __init__(self, args):
        args.enc_rec = True
        args.enc_use_layer_scale = False
        args.use_rep_encoder = True
        args.dec_norm = "none"
        args.dec_flow_kernel_size = 3
        args.dec_use_layer_scale = False
        args.dec_mlp_out_kernel_size = 3
        args.next_kernel_size = 3
        args.next_num_conv_dw_layers = 3
        args.next_use_conv_dw_relu = False
        args.enc_use_ms_bn = True
        args.enc_hidden_chs = 192
        args.enc_out_chs = 2 * args.enc_hidden_chs
        args.enc_feat_chs = 2 * args.enc_out_chs // 3
        args.dec_inp_chs = args.enc_out_chs // 3
        args.dec_net_chs = args.enc_out_chs // 3
        super().__init__(args)


class TrepFlow_Table2_NoNorm(TrepFlow):
    def __init__(self, args):
        args.enc_rec = True
        args.enc_use_layer_scale = False
        args.use_rep_encoder = False
        args.enc_norm = "none"
        args.dec_norm = "none"
        args.dec_flow_kernel_size = 3
        args.dec_use_layer_scale = False
        args.dec_mlp_out_kernel_size = 3
        args.next_kernel_size = 3
        args.next_num_conv_dw_layers = 3
        args.next_use_conv_dw_relu = False
        super().__init__(args)


class TrepFlow_Table2_BatchNorm(TrepFlow):
    def __init__(self, args):
        args.enc_rec = True
        args.enc_use_layer_scale = False
        args.use_rep_encoder = True
        args.dec_norm = "none"
        args.dec_flow_kernel_size = 3
        args.dec_use_layer_scale = False
        args.dec_mlp_out_kernel_size = 3
        args.next_kernel_size = 3
        args.next_num_conv_dw_layers = 3
        args.next_use_conv_dw_relu = False
        super().__init__(args)


class TrepFlow_Table2_LayerNorm(TrepFlow):
    def __init__(self, args):
        args.enc_rec = True
        args.enc_use_layer_scale = False
        args.use_rep_encoder = False
        args.enc_norm = "layer"
        args.dec_norm = "none"
        args.dec_flow_kernel_size = 3
        args.dec_use_layer_scale = False
        args.dec_mlp_out_kernel_size = 3
        args.next_kernel_size = 3
        args.next_num_conv_dw_layers = 3
        args.next_use_conv_dw_relu = False
        super().__init__(args)


class TrepFlow_Table2_RecBatchNorm(TrepFlow):
    def __init__(self, args):
        args.enc_rec = True
        args.enc_use_layer_scale = False
        args.use_rep_encoder = True
        args.dec_norm = "none"
        args.dec_flow_kernel_size = 3
        args.dec_use_layer_scale = False
        args.dec_mlp_out_kernel_size = 3
        args.next_kernel_size = 3
        args.next_num_conv_dw_layers = 3
        args.next_use_conv_dw_relu = False
        args.enc_use_ms_bn = True
        args.use_rep_branchless = True
        super().__init__(args)


class TrepFlow_Table2_TrepEncDec(TrepFlow):
    def __init__(self, args):
        args.enc_rec = True
        args.enc_use_layer_scale = False
        args.use_rep_encoder = True
        args.dec_norm = "none"
        args.dec_flow_kernel_size = 3
        args.dec_use_layer_scale = False
        args.dec_mlp_out_kernel_size = 3
        args.next_kernel_size = 3
        args.next_num_conv_dw_layers = 3
        args.next_use_conv_dw_relu = False
        args.use_rep_gru = True
        args.enc_use_ms_bn = True
        args.dec_use_ms_bn = True
        args.net_use_tanh = False
        args.motenc = "rep"
        super().__init__(args)


class TrepFlow_Table3_ConvNeXt(TrepFlow):
    def __init__(self, args):
        args.enc_rec = True
        args.enc_use_layer_scale = True
        args.use_rep_encoder = False
        args.enc_norm = "layer"
        args.dec_norm = "none"
        args.dec_flow_kernel_size = 3
        args.dec_use_layer_scale = True
        args.dec_mlp_out_kernel_size = 1
        args.next_kernel_size = 7
        args.next_num_conv_dw_layers = 1
        args.next_use_conv_dw_relu = False
        args.next_use_act_relu = False
        super().__init__(args)


class TrepFlow_Table3_GELU_ReLU(TrepFlow):
    def __init__(self, args):
        args.enc_rec = True
        args.enc_use_layer_scale = True
        args.use_rep_encoder = False
        args.enc_norm = "layer"
        args.dec_norm = "none"
        args.dec_flow_kernel_size = 3
        args.dec_use_layer_scale = True
        args.dec_mlp_out_kernel_size = 1
        args.next_kernel_size = 7
        args.next_num_conv_dw_layers = 1
        args.next_use_conv_dw_relu = False
        super().__init__(args)


class TrepFlow_Table3_NoLayerScale(TrepFlow):
    def __init__(self, args):
        args.enc_rec = True
        args.enc_use_layer_scale = False
        args.use_rep_encoder = False
        args.enc_norm = "layer"
        args.dec_norm = "none"
        args.dec_flow_kernel_size = 3
        args.dec_use_layer_scale = False
        args.dec_mlp_out_kernel_size = 1
        args.next_kernel_size = 7
        args.next_num_conv_dw_layers = 1
        args.next_use_conv_dw_relu = False
        super().__init__(args)


class TrepFlow_Table3_7x7_3x3x3(TrepFlow):
    def __init__(self, args):
        args.enc_rec = True
        args.enc_use_layer_scale = False
        args.use_rep_encoder = False
        args.enc_norm = "layer"
        args.dec_norm = "none"
        args.dec_flow_kernel_size = 3
        args.dec_use_layer_scale = False
        args.dec_mlp_out_kernel_size = 1
        args.next_kernel_size = 3
        args.next_num_conv_dw_layers = 3
        args.next_use_conv_dw_relu = False
        args.enc_use_ms_bn = True
        super().__init__(args)


class TrepFlow_Table3_LayerNm_Trep(TrepFlow):
    def __init__(self, args):
        args.enc_rec = True
        args.enc_use_layer_scale = False
        args.use_rep_encoder = True
        args.dec_norm = "none"
        args.dec_flow_kernel_size = 3
        args.dec_use_layer_scale = False
        args.dec_mlp_out_kernel_size = 1
        args.next_kernel_size = 3
        args.next_num_conv_dw_layers = 3
        args.next_use_conv_dw_relu = False
        args.enc_use_ms_bn = True
        super().__init__(args)


class TrepFlow_Table4_28M_NonRec(TrepFlow):
    def __init__(self, args):
        args.enc_rec = False
        args.enc_use_layer_scale = False
        args.use_rep_encoder = True
        args.dec_norm = "none"
        args.dec_flow_kernel_size = 3
        args.dec_use_layer_scale = False
        args.dec_mlp_out_kernel_size = 3
        args.next_kernel_size = 3
        args.next_num_conv_dw_layers = 3
        args.next_use_conv_dw_relu = False
        super().__init__(args)


class TrepFlow_Table4_29M_Rec(TrepFlow):
    def __init__(self, args):
        args.enc_rec = True
        args.enc_use_layer_scale = False
        args.use_rep_encoder = True
        args.dec_norm = "none"
        args.dec_flow_kernel_size = 3
        args.dec_use_layer_scale = False
        args.dec_mlp_out_kernel_size = 3
        args.next_kernel_size = 3
        args.next_num_conv_dw_layers = 3
        args.next_use_conv_dw_relu = False
        args.enc_use_ms_bn = True
        args.enc_hidden_chs = 132
        args.enc_out_chs = 2 * args.enc_hidden_chs
        args.enc_feat_chs = 2 * args.enc_out_chs // 3
        args.dec_inp_chs = args.enc_out_chs // 3
        args.dec_net_chs = args.enc_out_chs // 3
        super().__init__(args)


class TrepFlow_Table4_51M_NonRec(TrepFlow):
    def __init__(self, args):
        args.enc_rec = False
        args.enc_use_layer_scale = False
        args.use_rep_encoder = True
        args.dec_norm = "none"
        args.dec_flow_kernel_size = 3
        args.dec_use_layer_scale = False
        args.next_kernel_size = 3
        args.next_num_conv_dw_layers = 3
        args.next_use_conv_dw_relu = False
        args.enc_hidden_chs = 144
        args.enc_out_chs = 2 * args.enc_hidden_chs
        args.enc_feat_chs = 2 * args.enc_out_chs // 3
        args.dec_inp_chs = args.enc_out_chs // 3
        args.dec_net_chs = args.enc_out_chs // 3
        args.dec_motion_chs = 128
        args.dec_mlp_out_kernel_size = 3
        super().__init__(args)


models_dict = {
    "trepflow_s": (TrepFlow_S, "ckpts/trepflow_s_things.ckpt"),
    "trepflow_m": (TrepFlow_M, "ckpts/trepflow_m_things_s1.ckpt"),
    "trepflow_l": (TrepFlow_L, "ckpts/trepflow_l_things.ckpt"),
    "trepflow_t2_nonorm": (TrepFlow_Table2_NoNorm, None),
    "trepflow_t2_bnorm": (TrepFlow_Table2_BatchNorm, "ckpts/table2_batchnorm_s1.ckpt"),
    "trepflow_t2_lnorm": (TrepFlow_Table2_LayerNorm, "ckpts/table2_layernorm_s1.ckpt"),
    "trepflow_t2_recbnorm": (
        TrepFlow_Table2_RecBatchNorm,
        "ckpts/table2_recbatchnorm_s1.ckpt",
    ),
    "trepflow_t2_trepencdec": (
        TrepFlow_Table2_TrepEncDec,
        "ckpts/table2_trepencdec_s1.ckpt",
    ),
    "trepflow_t3_next": (TrepFlow_Table3_ConvNeXt, "ckpts/table3_convnext_s1.ckpt"),
    "trepflow_t3_relu": (TrepFlow_Table3_GELU_ReLU, "ckpts/table3_relu_s1.ckpt"),
    "trepflow_t3_nols": (
        TrepFlow_Table3_NoLayerScale,
        "ckpts/table3_nolayerscale_s1.ckpt",
    ),
    "trepflow_t3_333": (TrepFlow_Table3_7x7_3x3x3, "ckpts/table3_conv333_s1.ckpt"),
    "trepflow_t3_trep": (TrepFlow_Table3_LayerNm_Trep, "ckpts/table3_trep_s1.ckpt"),
    "trepflow_t4_nonrec_28m": (
        TrepFlow_Table4_28M_NonRec,
        "ckpts/table4_nonrec_28m.ckpt",
    ),
    "trepflow_t4_rec_29m": (TrepFlow_Table4_29M_Rec, "ckpts/table4_rec_29m.ckpt"),
    "trepflow_t4_nonrec_51m": (
        TrepFlow_Table4_51M_NonRec,
        "ckpts/table4_nonrec_51m.ckpt",
    ),
}
