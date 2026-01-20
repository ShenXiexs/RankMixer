# @File : train_config.py
import copy
import os
import sys

# 保持与原项目一致的路径层级（config/<model_version>/...）
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _extract_train_date(argv):
    for flag in ("--end_time_str", "--time_str"):
        for i, arg in enumerate(argv):
            if arg == flag and i + 1 < len(argv):
                return str(argv[i + 1])[:8]
            if arg.startswith(flag + "="):
                return arg.split("=", 1)[1][:8]
    return None


def _resolve_learning_rate(default_lr=2e-4, cutoff_mmdd="1102", before_lr=5e-4):
    train_date = _extract_train_date(sys.argv)
    if not train_date or len(train_date) < 8:
        return default_lr
    cutoff_date = train_date[:4] + cutoff_mmdd
    return before_lr if train_date <= cutoff_date else default_lr

class TrainConfig:
    """
    关键改动：
    1) 模型入口切换为 RankMixer（GPU 优先）
    2) 显式配置 seq_length（固定长度），供 RankMixer 使用
    3) 在 train_params 中新增 rankmixer 超参，并将 d_model 与动态 Embedding 维度对齐
    """

    # ======================= 基本信息 =======================
    model_version = "RankMixer_Shen0118"                       # 必填：版本名（影响配置/输出路径）
    model_modul   = "models.rankmixer_shen0118.model_fn"       # RankMixer_Shen0118 Estimator 入口
    dataset_modul = "dataset.dataset_seq.input_fn"             # 仍采用现有 TF 数据管道

    ### GPU训练参数配置
    device = "GPU"  # Device to use: cpu, gpu, or multi_gpu
    gpu_list = "1"  # Comma-separated list of GPU IDs for multi-GPU mode
    gpu_memory_limit = 0  # GPU memory limit in MB (0 for no limit)
    gpu_memory_growth = True  # Allow GPU memory growth
    keep_date_ranges = [("1101", "1110")]
    
    # ======================= 训练参数（传入 model_fn 的 params） =======================
    train_params = {
        # 优化器配置（供 RankMixer 主干使用）
        "optimize_config": {
            "learning_rate": _resolve_learning_rate(),
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
            "warmup_steps": 4000,
            "decay_type": "none",
            "decay_steps": 0,
            "min_learning_rate": 0.0,
            "grad_clip_norm": 5.0,
        },
        # ★RankMixer_Shen0118 超参（论文 100M 配置）
        "rankmixer": {
            # ===== 模型规模 =====
            "d_model": 128,
            "num_layers": 8,
            "num_tokens": 64,
            "num_heads": 64,  # 论文要求 H = T
            "ffn_mult": 8,

            # ===== Tokenization =====
            "tokenization_strategy": "semantic",
            "tokenization_version": "v2",
            "semantic_target_tokens": 64,
            "semantic_groups": {},
            "token_projection": "linear",
            "include_seq_in_tokenization": True,

            # ===== Token Mixing (paper strict) =====
            "token_mixing_type": "paper_strict",
            "token_mixing_dropout": 0.0,

            # ===== Per-token FFN =====
            "per_token_ffn": True,
            "ffn_activation": "gelu",
            "ffn_dropout": 0.0,

            # ===== LN 风格 =====
            "ln_style": "pre",
            "input_ln": False,
            "use_final_ln": True,

            # ===== Pooling / Head =====
            "output_pooling": "mean",
            "add_cls_token": False,
            "head_dropout": 0.0,

            # ===== MoE（100M 关闭） =====
            "use_moe": False,
            "moe_num_experts": 8,
            "moe_sparsity_ratio": 0.125,
            "moe_routing_type": "relu_dtsi",
            "moe_l1_lambda": 0.01,
            "moe_use_dtsi": True,

            # ===== 其他基础配置 =====
            "use_other_features": True,
            "seq_pool": "mean",
            "embedding_size": 9,
            "input_dropout": 0.0,

            # ===== Loss =====
            "use_ctcvr_loss": True,
            "use_conditional_cvr": False,
        },

        # 资源与动态表策略（与 baseline 对齐）
        "ps_num": 1,                  # TFRA 动态表挂至 GPU:0，避免回落 CPU
        "restrict": True,             # 控制动态表规模，显存可控
        "l2_reg": 1e-6,
    }

    # ======================= 数据 Schema =======================
    data_schema = ["user_id", "requestid", "combination_un_id", "is_click", "is_conversion", "features", "app_pkg_src", "app_pkg", "app_first_type", "seq_features" ] #必填
    label_schema = {"is_click": "ctr_label",
                    "is_conversion": "ctcvr_label"}                                                        #必填
    # seq_feature配置
    seq_features_config = [
        {"name": "user_awake_90d_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户90天唤醒包序列"},
        {"name": "user_awake_90d_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户90天唤醒包一级分类序列"},
        {"name": "user_awake_30d_recent_10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户30日最近10次唤醒序列"},
        {"name": "user_awake_30d_freq_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户30日唤醒包频次最多10个序列"},
        {"name": "user_awake_30d_sticky_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户30日粘性最高10个唤醒包序列"},
        {"name": "user_first_awke_90d_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户90天首唤包序列"},
        {"name": "user_first_awke_90d_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户90天首唤包一级分类序列"},
        {"name": "user_first_awke_30d_recent_10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户30日最近10次首唤序列"},
        {"name": "user_first_awke_30d_freq_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户30日首唤包频次最多10个序列"},
        {"name": "user_first_awke_30d_sticky_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户30日粘性最高10个首唤包序列"},
        {"name": "user_imp_app_90d_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户90天曝光包序列"},
        {"name": "user_imp_app_90d_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户90天曝光包应用一级分类序列"},
        {"name": "user_imp_app_30d_recent_10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户30最近10次曝光包序列"},
        {"name": "user_imp_app_30d_freq_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户30日曝光包频次最多10个序列"},
        {"name": "user_imp_app_30d_sticky_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户30日粘性最高10个曝光包序列"},
        {"name": "user_clk_app_90d_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户90天点击包序列"},
        {"name": "user_clk_app_90d_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户90天点击包应用一级分类序列"},
        {"name": "user_clk_app_30d_recent_10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户30最近10次点击包序列"},
        {"name": "user_clk_app_30d_freq_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户30日点击包频次最多10个序列"},
        {"name": "user_clk_app_30d_sticky_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户30日粘性最高10个点击包序列"},
        {"name": "user_imp_launch_recent_5_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "用户最近5次曝光启动序列"},
        {"name": "user_imp_launch_30d_freq_top10_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "用户30日曝光启动频次最多10个序列"},
        {"name": "user_imp_launch_30d_sticky_top10_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "用户30日粘性最高10个曝光启动序列"},
        {"name": "user_clk_launch_recent_5_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "用户最近5次点击启动序列"},
        {"name": "user_clk_launch_30d_freq_top10_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "用户30日点击启动频次最多10个序列"},
        {"name": "user_clk_launch_30d_sticky_top10_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "用户30日粘性最高10个点击启动序列"},
        {"name": "user_awake_7d_recent_5_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户7日最近5次唤醒序列"},
        {"name": "user_awake_15d_freq_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户15日唤醒频次最多10个序列"},
        {"name": "user_awake_7d_freq_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户7日唤醒频次最多10个序列"},
        {"name": "user_awake_1d_freq_top5_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户1日唤醒频次最多5个序列"},
        {"name": "user_awake_15d_freq_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户15日唤醒包应用一级分类频次最多10个序列"},
        {"name": "user_awake_7d_freq_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户7日唤醒包应用一级分类频次最多10个序列"},
        {"name": "user_awake_1d_freq_top5_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户1日唤醒包应用一级分类频次最多5个序列"},
        {"name": "user_awake_15d_sticky_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户15日粘性最高10个唤醒序列"},
        {"name": "user_awake_7d_sticky_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户7日粘性最高10个唤醒序列"},
        {"name": "user_awake_1d_sticky_top5_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户1日粘性最高5个唤醒序列"},
        {"name": "user_awake_15d_sticky_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户15日粘性最高10个唤醒包应用一级分类序列"},
        {"name": "user_awake_7d_sticky_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户7日粘性最高10个唤醒包应用一级分类序列"},
        {"name": "user_awake_1d_sticky_top5_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户1日粘性最高5个唤醒包应用一级分类序列"},
        {"name": "user_first_awke_7d_recent_5_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户7日最近5次首唤序列"},
        {"name": "user_first_awke_15d_freq_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户15日首唤频次最多10个序列"},
        {"name": "user_first_awke_7d_freq_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户7日首唤频次最多10个序列"},
        {"name": "user_first_awke_1d_freq_top5_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户1日首唤频次最多5个序列"},
        {"name": "user_first_awke_15d_freq_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20,
         "pad": "0", "description": "用户15日首唤包应用一级分类频次最多10个序列"},
        {"name": "user_first_awke_7d_freq_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20,
         "pad": "0", "description": "用户7日首唤包应用一级分类频次最多10个序列"},
        {"name": "user_first_awke_1d_freq_top5_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户1日首唤包应用一级分类频次最多5个序列"},
        {"name": "user_first_awke_15d_sticky_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户15日粘性最高10个首唤序列"},
        {"name": "user_first_awke_7d_sticky_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户7日粘性最高10个首唤序列"},
        {"name": "user_first_awke_1d_sticky_top5_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户1日粘性最高5个首唤序列"},
        {"name": "user_first_awke_15d_sticky_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20,
         "pad": "0", "description": "用户15日粘性最高10个首唤包应用一级分类序列"},
        {"name": "user_first_awke_7d_sticky_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20,
         "pad": "0", "description": "用户7日粘性最高10个首唤包应用一级分类序列"},
        {"name": "user_first_awke_1d_sticky_top5_app_type_seq", "is_download": 1, "index": "27", "length": 20,
         "pad": "0", "description": "用户1日粘性最高5个首唤包应用一级分类序列"},
        {"name": "user_imp_app_7d_recent_5_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户7日最近5次曝光包序列"},
        {"name": "user_imp_app_15d_freq_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户15日曝光包频次最多10个序列"},
        {"name": "user_imp_app_7d_freq_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户7日曝光包频次最多10个序列"},
        {"name": "user_imp_app_1d_freq_top5_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户1日曝光包频次最多5个序列"},
        {"name": "user_imp_app_15d_freq_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户15日曝光包应用一级分类频次最多10个序列"},
        {"name": "user_imp_app_7d_freq_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户7日曝光包应用一级分类频次最多10个序列"},
        {"name": "user_imp_app_1d_freq_top5_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户1日曝光包应用一级分类频次最多5个序列"},
        {"name": "user_imp_app_15d_sticky_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户15日粘性最高10个曝光包序列"},
        {"name": "user_imp_app_7d_sticky_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户7日粘性最高10个曝光包序列"},
        {"name": "user_imp_app_1d_sticky_top5_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户1日粘性最高5个曝光包序列"},
        {"name": "user_imp_app_15d_sticky_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20,
         "pad": "0", "description": "用户15日粘性最高10个曝光包应用一级分类序列"},
        {"name": "user_imp_app_7d_sticky_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户7日粘性最高10个曝光包应用一级分类序列"},
        {"name": "user_imp_app_1d_sticky_top5_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户1日粘性最高5个曝光包应用一级分类序列"},
        {"name": "user_clk_app_7d_recent_5_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户7日最近5次点击包序列"},
        {"name": "user_clk_app_15d_freq_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户15日点击包频次最多10个序列"},
        {"name": "user_clk_app_7d_freq_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户7日点击包频次最多10个序列"},
        {"name": "user_clk_app_1d_freq_top5_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户1日点击包频次最多5个序列"},
        {"name": "user_clk_app_15d_freq_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户15日点击包应用一级分类频次最多10个序列"},
        {"name": "user_clk_app_7d_freq_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户7日点击包应用一级分类频次最多10个序列"},
        {"name": "user_clk_app_1d_freq_top5_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户1日点击包应用一级分类频次最多5个序列"},
        {"name": "user_clk_app_15d_sticky_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户15日粘性最高10个点击包序列"},
        {"name": "user_clk_app_7d_sticky_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户7日粘性最高10个点击包序列"},
        {"name": "user_clk_app_1d_sticky_top5_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户1日粘性最高5个点击包序列"},
        {"name": "user_clk_app_15d_sticky_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20,
         "pad": "0", "description": "用户15日粘性最高10个点击包应用一级分类序列"},
        {"name": "user_clk_app_7d_sticky_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户7日粘性最高10个点击包应用一级分类序列"},
        {"name": "user_clk_app_1d_sticky_top5_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户1日粘性最高5个点击包应用一级分类序列"},
        {"name": "user_imp_launch_15d_freq_top20_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "用户15日曝光启动频次最多20个序列"},
        {"name": "user_imp_launch_7d_freq_top10_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "用户7日曝光启动频次最多10个序列"},
        {"name": "user_imp_launch_1d_freq_top5_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "用户1日曝光启动频次最多5个序列"},
        {"name": "user_imp_launch_15d_sticky_top20_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "用户15日粘性最高20个曝光启动序列"},
        {"name": "user_imp_launch_7d_sticky_top10_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "用户7日粘性最高10个曝光启动序列"},
        {"name": "user_imp_launch_1d_sticky_top5_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "用户1日粘性最高5个曝光启动序列"},
        {"name": "user_clk_launch_15d_freq_top20_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "用户15日点击启动频次最多20个序列"},
        {"name": "user_clk_launch_7d_freq_top10_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "用户7日点击启动频次最多10个序列"},
        {"name": "user_clk_launch_1d_freq_top5_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "用户1日点击启动频次最多5个序列"},
        {"name": "user_clk_launch_15d_sticky_top20_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "用户15日粘性最高20个点击启动序列"},
        {"name": "user_clk_launch_7d_sticky_top10_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "用户7日粘性最高10个点击启动序列"},
        {"name": "user_clk_launch_1d_sticky_top5_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "用户1日粘性最高5个点击启动序列"},
    ]

    # ======================= 标签映射 / 预测输出列 =======================
    # 跟模型中prediction中out的输出对其，给模型输出补key，组成json格式，用于推理模型的效果存储
    predict_columns = [k for k,v in label_schema.items() if v.endswith("_label")] \
                    + [v.replace("_label", "_pred") for k,v in label_schema.items() if v.endswith("_label")]  # 必填

    # ======================= 解析/压缩配置 =======================
    field_sep = "\003"  # 字段的分隔符
    features_sep = "\002"  # features特征的分隔符
    compression_type = "GZIP"  # 数据压缩格式

    # ======================= 线下特征/分桶等配置 =======================
    # 定义分桶且特征选择表
    binning_table_name = "tmp_ad_rank_cvr_activation_sample_data_v2"
    partitions = "ds_date='{day}',durations='1',model_type='TO5'"
    downodps_datas = ['20250901']

    # ======================= 本地/OSS 配置路径 =======================
    schema_path = f"{dirname}/config/{model_version}/schema.conf"  # 该文件必须要
    slot_path = f"{dirname}/config/{model_version}/slot.conf"  # 该文件必须要
    sel_feat_path = f"{dirname}/config/{model_version}/select_feature.conf"
    boundaries_map_path = f"{dirname}/config/{model_version}/boundaries_map.json"  # 该文件必须要
    fg_path = f"{dirname}/config/{model_version}/fg.json"  # 该文件必须要
    feature_config_path = f"{dirname}/config/{model_version}/feature_config.json"
    body_json_name = f"{dirname}/config/{model_version}/body.json"

    # ======================= Estimator 运行配置（与 baseline 一致） =======================
    es_run_config = {
        "keep_checkpoint_max": 1,
        "save_checkpoints_steps": 100000,
        "log_step_count_steps": 5000,
        "save_summary_steps": 10000
    }

    # ======================= Dataset input_fn 配置 =======================
    data_nm = "TO5"
    inp_fn_config = {
        "train_spec": {
            "max_steps": None
        },
        "eval_spec": {
            "start_delay_secs": 1e20,
            "steps": None
        },
        "train_batch_size": 2048,
        "train_epoch": 1,
        "batch_size": 2048
    }

    # ======================= 写回/导出/指标 =======================
    # infer数据写入的结果表
    infer_table_name = 'adx_dmp.ads_algorithm_yoyo_model_offline_shallow_predict'
    ### upload
    oss_bucket_name = "adx-oss"
    upload_oss_path = "rankmixer_shen0118_model_test"  #定义模型导出OSS路径
    oss_offline_root_path = "deep_model/offline"  # 离线特征推送OSS路径，判断特征是否推线上，再推模型到OSS供线上推理使用
    # 模型训练指标写入表, yoyo_model独有
    metric_table = 'adx_dmp.ads_algorithm_yoyo_model_eval_metric_table_dm'
    # 当前模型类型，ctr,cvr,ctcvr..
    eval_type = "ctcvr"
    # deep_model/offline/{}/20250924/_FEATURE_SUCCESS is exists
    oss_offline_model_ver = "rankmixer_shen0118"


class TrainConfig_1B(TrainConfig):
    """
    1B 版本覆盖参数（需手动切换 TRAIN_CONFIG 指向本类或复制为主类）。
    """
    model_version = "RankMixer_Shen0118"
    train_params = copy.deepcopy(TrainConfig.train_params)
    train_params["rankmixer"].update({
        "d_model": 1536,
        "num_tokens": 32,
        "num_heads": 32,
        "use_moe": True,
        "moe_num_experts": 8,
        "moe_sparsity_ratio": 0.125,
        "moe_routing_type": "relu_dtsi",
        "moe_l1_lambda": 0.01,
    })

