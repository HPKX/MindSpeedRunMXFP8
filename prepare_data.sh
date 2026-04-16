#!/bin/bash
# ============================================================================
# DeepSeek V3 MXFP8 训练数据准备脚本
#
# 前置条件：
#   1. 已执行 setup_mxfp8_env.sh 完成环境部署
#   2. 已 source env_mxfp8.sh
#
# 用法：
#   bash prepare_data.sh [MODE] [DATA_DIR]
#
# 参数：
#   MODE     - smoke (默认) | wikitext | custom
#              smoke:    内嵌 ~20 条样本，数秒完成，仅用于 smoke test
#              wikitext: 下载 wikitext-2-raw-v1（~5MB），用于轻量训练验证
#              custom:   使用你自己的数据。需先把 jsonl 文件放到
#                        $DATA_DIR/input.jsonl，每行 {"text": "..."}
#   DATA_DIR - 数据输出目录（默认 <脚本所在目录>/dataset/deepseek3）
#
# 行为：
#   以本脚本所在目录为工作目录，自动 source 同目录 env_mxfp8.sh。
# ============================================================================

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
log_info()  { echo -e "${GREEN}[DATA]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[DATA]${NC} $*"; }
log_error() { echo -e "${RED}[DATA]${NC} $*"; }
die()       { log_error "$*"; exit 1; }

MODE="${1:-smoke}"

# 工作目录 = 脚本所在目录
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || die "无法定位脚本所在目录"
[ -n "$WORK_DIR" ] || die "工作目录路径为空"

# 自动加载 env_mxfp8.sh
if [ -f "$WORK_DIR/env_mxfp8.sh" ]; then
    # shellcheck disable=SC1090,SC1091
    source "$WORK_DIR/env_mxfp8.sh"
fi

DATA_DIR="${2:-$WORK_DIR/dataset/deepseek3}"
LLM_DIR="$WORK_DIR/MindSpeed-LLM"

if [ ! -d "$LLM_DIR" ] || [ ! -f "$LLM_DIR/preprocess_data.py" ]; then
    log_error "$LLM_DIR/preprocess_data.py 不存在"
    log_error "请先在工作目录下执行: ./setup_mxfp8_env.sh"
    exit 1
fi

mkdir -p "$DATA_DIR"
log_info "数据目录: $DATA_DIR"
log_info "模式:     $MODE"

# ========================= Step 1: 下载 Tokenizer ===========================
log_info "========== Step 1: 下载 DeepSeek-V3 Tokenizer =========="

TOKENIZER_FILES=(
    "tokenizer.json"
    "tokenizer_config.json"
    "config.json"
)

# 判断是否已下载完整
ALL_EXIST=true
for f in "${TOKENIZER_FILES[@]}"; do
    if [ ! -f "$DATA_DIR/$f" ]; then
        ALL_EXIST=false
        break
    fi
done

if [ "$ALL_EXIST" = true ]; then
    log_info "Tokenizer 文件已存在，跳过下载"
else
    # 优先使用 huggingface-cli
    if command -v huggingface-cli >/dev/null 2>&1; then
        log_info "使用 huggingface-cli 下载..."
        huggingface-cli download deepseek-ai/DeepSeek-V3 \
            --include "tokenizer*" "config.json" \
            --local-dir "$DATA_DIR" \
            --local-dir-use-symlinks False
    else
        log_warn "huggingface-cli 不可用，尝试 pip install huggingface_hub"
        # 保留 stderr 便于诊断；|| true 允许失败后走 else 分支给出手动下载指引
        pip3 install huggingface_hub || true
        if command -v huggingface-cli >/dev/null 2>&1; then
            huggingface-cli download deepseek-ai/DeepSeek-V3 \
                --include "tokenizer*" "config.json" \
                --local-dir "$DATA_DIR" \
                --local-dir-use-symlinks False
        else
            log_error "无法安装 huggingface-cli，请手动下载："
            log_error "  https://huggingface.co/deepseek-ai/DeepSeek-V3/tree/main"
            log_error "  把 tokenizer.json / tokenizer_config.json / config.json 放到 $DATA_DIR"
            exit 1
        fi
    fi
fi

log_info "Tokenizer 就绪: $DATA_DIR"

# ========================= Step 2: 准备原始 JSONL ===========================
log_info "========== Step 2: 准备原始 JSONL =========="

RAW_JSONL="$DATA_DIR/enwiki_raw.jsonl"

case "$MODE" in
    smoke)
        log_info "生成内嵌 smoke 样本..."
        python3 - << 'PYEOF' > "$RAW_JSONL"
import json

# 20 条短文本样本，用于 smoke test
samples = [
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "In machine learning, neural networks approximate complex functions through layers of transformations.",
    "Transformer architectures revolutionized natural language processing with self-attention mechanisms.",
    "Mixed precision training uses lower precision formats like FP16 or FP8 to accelerate computation.",
    "Distributed training partitions models and data across multiple accelerators for scalability.",
    "Microscaling FP8 provides block-wise scaling factors to maintain numerical stability at 8-bit precision.",
    "The backward pass computes gradients through reverse-mode automatic differentiation.",
    "Mixture of Experts models route inputs to specialized sub-networks to increase capacity efficiently.",
    "Multi-head latent attention reduces the memory footprint of key-value caches during inference.",
    "Pipeline parallelism splits model layers across devices to train very large models.",
    "Tensor parallelism splits individual layer computations across devices along the hidden dimension.",
    "Expert parallelism distributes MoE experts across devices to scale the total parameter count.",
    "Flash attention fuses attention computation to reduce memory bandwidth usage significantly.",
    "Rotary position embeddings encode relative positions through rotation matrices in the attention computation.",
    "RMSNorm normalizes activations using the root mean square without subtracting the mean.",
    "Gradient accumulation simulates larger batch sizes on devices with limited memory capacity.",
    "Weight decay regularizes parameters by shrinking them toward zero during optimization steps.",
    "The Adam optimizer combines momentum and adaptive learning rates using first and second moments of gradients.",
    "Cosine learning rate schedules smoothly decay the learning rate over the course of training.",
    "Gradient clipping prevents exploding gradients by rescaling them when their norm exceeds a threshold.",
]

# 每条重复 5 次以增加 token 数量（保证 preprocess 有足够数据）
for _ in range(5):
    for s in samples:
        print(json.dumps({"text": s}))
PYEOF
        # 验证写入成功（磁盘满 / IO 错误时 heredoc 可能生成空文件）
        [ -s "$RAW_JSONL" ] || die "smoke 样本生成失败或为空文件: $RAW_JSONL"
        WC=$(wc -l < "$RAW_JSONL")
        log_info "smoke 样本已生成: $RAW_JSONL ($WC 行)"
        ;;

    wikitext)
        log_info "下载 wikitext-2-raw-v1..."
        # 预检 datasets 库可用性（由 MindSpeed-LLM/requirements.txt 带入）
        python3 -c "import datasets" 2>/dev/null \
            || die "datasets 库不可用，请重跑 setup_mxfp8_env.sh 或手动 pip3 install datasets"
        if [ -f "$RAW_JSONL" ]; then
            log_info "$RAW_JSONL 已存在，跳过下载"
        else
            # 使用带单引号的 heredoc（'PYEOF'）关闭 shell 变量展开，避免 $RAW_JSONL
            # 被 shell 先行展开后注入 Python 源码。路径通过环境变量传递给 Python。
            RAW_JSONL="$RAW_JSONL" python3 - << 'PYEOF'
import os
from datasets import load_dataset
import json

output = os.environ["RAW_JSONL"]
print(f"Loading wikitext-2-raw-v1... -> {output}")
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")

with open(output, "w") as f:
    count = 0
    for item in ds:
        text = item["text"].strip()
        if len(text) < 50:  # 过滤空行和短行
            continue
        f.write(json.dumps({"text": text}) + "\n")
        count += 1
print(f"Saved {count} samples to {output}")
PYEOF
            [ -s "$RAW_JSONL" ] || die "wikitext 下载失败或为空文件: $RAW_JSONL"
            log_info "wikitext 已保存: $RAW_JSONL"
        fi
        ;;

    custom)
        CUSTOM_INPUT="$DATA_DIR/input.jsonl"
        if [ ! -f "$CUSTOM_INPUT" ]; then
            log_error "custom 模式下需要先准备 $CUSTOM_INPUT"
            log_error "要求 jsonl 格式，每行 {\"text\": \"...\"}"
            exit 1
        fi
        RAW_JSONL="$CUSTOM_INPUT"
        log_info "使用自定义数据: $RAW_JSONL"
        ;;

    *)
        log_error "未知 MODE: $MODE (支持 smoke / wikitext / custom)"
        exit 1
        ;;
esac

# ========================= Step 3: 预处理为二进制 ===========================
log_info "========== Step 3: 预处理为 .bin + .idx =========="

OUTPUT_PREFIX="$DATA_DIR/enwiki_text_document"

# 已存在就跳过（幂等）
if [ -f "${OUTPUT_PREFIX}.bin" ] && [ -f "${OUTPUT_PREFIX}.idx" ]; then
    log_info ".bin 和 .idx 已存在，跳过预处理"
    log_warn "  如需重新处理，请删除: ${OUTPUT_PREFIX}.{bin,idx}"
else
    log_info "开始预处理（调用 preprocess_data.py）..."
    # 子 shell 隔离 cd —— 无论 python3 成功与否，CWD 都会自动还原
    (
        cd "$LLM_DIR"
        python3 preprocess_data.py \
            --input "$RAW_JSONL" \
            --output-prefix "$OUTPUT_PREFIX" \
            --tokenizer-type PretrainedFromHF \
            --tokenizer-name-or-path "$DATA_DIR" \
            --workers 4 \
            --json-keys text \
            --append-eod
    ) || die "preprocess_data.py 执行失败"
    log_info "预处理完成"
fi

# ========================= Step 4: 验证产物 =================================
log_info "========== Step 4: 验证数据产物 =========="

MISSING=()
for ext in bin idx; do
    f="${OUTPUT_PREFIX}.${ext}"
    if [ -f "$f" ]; then
        SIZE=$(du -h "$f" | cut -f1)
        log_info "  ✓ ${f}  ($SIZE)"
    else
        MISSING+=("$f")
    fi
done

for f in tokenizer.json tokenizer_config.json config.json; do
    if [ -f "$DATA_DIR/$f" ]; then
        log_info "  ✓ $DATA_DIR/$f"
    else
        MISSING+=("$DATA_DIR/$f")
    fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
    log_error "缺失文件: ${MISSING[*]}"
    exit 1
fi

# ========================= 完成 =============================================
echo ""
log_info "=============================================="
log_info "  数据准备完成"
log_info "=============================================="
echo ""
echo "  数据路径："
echo "    DATA_PATH       = ${OUTPUT_PREFIX}"
echo "    TOKENIZER_PATH  = ${DATA_DIR}"
echo ""

# 检查训练脚本路径是否已被 setup_mxfp8_env.sh 改写
TRAIN_SCRIPT="$WORK_DIR/MindSpeedRun/scripts/deepseek3_swap/8k_fp8_sbh_8p.sh"
if [ -f "$TRAIN_SCRIPT" ]; then
    if grep -q "^DATA_PATH=.*${DATA_DIR}" "$TRAIN_SCRIPT" 2>/dev/null; then
        log_info "✓ 训练脚本路径已对齐: $TRAIN_SCRIPT"
    else
        log_warn "训练脚本路径与当前数据目录不一致"
        log_warn "请执行以下命令手动对齐："
        echo ""
        echo "    sed -i 's|^DATA_PATH=.*|DATA_PATH=\"${OUTPUT_PREFIX}\"|' \"$TRAIN_SCRIPT\""
        echo "    sed -i 's|^TOKENIZER_PATH=.*|TOKENIZER_PATH=\"${DATA_DIR}\"|' \"$TRAIN_SCRIPT\""
        echo ""
    fi
fi

echo "  下一步启动训练："
echo "    cd $WORK_DIR"
echo "    ./run_training.sh smoke"
echo ""
