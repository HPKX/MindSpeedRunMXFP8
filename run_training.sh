#!/bin/bash
# ============================================================================
# DeepSeek V3 MXFP8 一键启动训练脚本
#
# 用法：
#   ./run_training.sh                      # 单机 8 卡，自动检测 IP
#   ./run_training.sh smoke                # 2 层 2 步，快速验证
#   ./run_training.sh normal "IP1 IP2"     # 多机启动（自动推导前缀）
#   ./run_training.sh smoke "" 192         # 单机 smoke，显式指定 inet 前缀
#
# 参数：
#   $1  MODE       - normal (默认) | smoke
#   $2  IP_LIST    - 多机 IP（空格分隔）；留空自动取本机 IP
#   $3  INET_PREFIX - inet 前缀数字；留空自动从本机 IP 提取
#
# 行为：
#   以本脚本所在目录为工作目录，自动 source 同目录 env_mxfp8.sh。
# ============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
log_info()  { echo -e "${GREEN}[RUN]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[RUN]${NC} $*"; }
log_error() { echo -e "${RED}[RUN]${NC} $*"; }
die()       { log_error "$*"; exit 1; }

MODE="${1:-normal}"
IP_LIST="${2:-}"
INET_PREFIX="${3:-}"

# 工作目录 = 脚本所在目录
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || die "无法定位脚本所在目录"
[ -n "$WORK_DIR" ] || die "工作目录路径为空"
LLM_DIR="$WORK_DIR/MindSpeed-LLM"
DATA_DIR="$WORK_DIR/dataset/deepseek3"
TRAIN_SCRIPT="$WORK_DIR/MindSpeedRun/scripts/deepseek3_swap/8k_fp8_sbh_8p.sh"

log_info "========== 环境检查 =========="
log_info "工作目录: $WORK_DIR"

# 自动加载 env_mxfp8.sh
if [ -f "$WORK_DIR/env_mxfp8.sh" ]; then
    # shellcheck disable=SC1090,SC1091
    source "$WORK_DIR/env_mxfp8.sh"
else
    log_error "未找到 $WORK_DIR/env_mxfp8.sh"
    log_error "请先在工作目录下执行: ./setup_mxfp8_env.sh"
    exit 1
fi

# 检查关键目录
for p in "$LLM_DIR" "$DATA_DIR"; do
    if [ ! -d "$p" ]; then
        log_error "目录不存在: $p"
        log_error "请先完成 setup_mxfp8_env.sh + prepare_data.sh"
        exit 1
    fi
done
if [ ! -f "$TRAIN_SCRIPT" ]; then
    log_error "训练脚本不存在: $TRAIN_SCRIPT"
    exit 1
fi

# ========================= 数据检查 =========================================
log_info "========== 数据检查 =========="

REQUIRED=(
    "$DATA_DIR/tokenizer.json"
    "$DATA_DIR/tokenizer_config.json"
    "$DATA_DIR/enwiki_text_document.bin"
    "$DATA_DIR/enwiki_text_document.idx"
)

MISSING=()
for f in "${REQUIRED[@]}"; do
    if [ ! -f "$f" ]; then
        MISSING+=("$f")
    fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
    log_error "数据未准备好，缺失："
    for f in "${MISSING[@]}"; do
        log_error "  - $f"
    done
    log_error ""
    log_error "请先执行: ./prepare_data.sh smoke"
    exit 1
fi
log_info "数据就绪: $DATA_DIR"

# ========================= 自动探测 IP / INET_PREFIX ========================
log_info "========== 网络配置 =========="

# 本机所有非环回 IP（优先 hostname -I，fallback 到 ifconfig 以兼容 macOS / 老系统）
LOCAL_IPS=$(hostname -I 2>/dev/null || true)
if [ -z "$LOCAL_IPS" ]; then
    LOCAL_IPS=$(ifconfig 2>/dev/null | grep -E "inet [0-9]" | grep -v "127.0.0.1" | awk '{print $2}' || true)
fi

# 自动取本机 IP（未指定 IP_LIST 时）
if [ -z "$IP_LIST" ]; then
    IP_LIST=$(echo "$LOCAL_IPS" | awk 'NF{print $1; exit}')
    [ -n "$IP_LIST" ] || die "无法自动获取本机 IP，请手动指定: ./run_training.sh $MODE \"<IP>\" [<INET_PREFIX>]"
    log_info "自动获取本机 IP: $IP_LIST"
fi

# 推导 INET_PREFIX（训练脚本用它 grep "inet $INET_PREFIX"，必须精确到整个 IP 避免
# 多网卡机器上误匹配同网段但错网卡的 IP。只要本机某个 IP 在 IP_LIST 里，就用那个
# 完整 IP 作为 prefix；否则回退到 IP_LIST 的第一个 IP。）
if [ -z "$INET_PREFIX" ]; then
    MY_IP=""
    for candidate in $IP_LIST; do
        for local_ip in $LOCAL_IPS; do
            if [ "$candidate" = "$local_ip" ]; then
                MY_IP="$candidate"
                break 2
            fi
        done
    done
    if [ -z "$MY_IP" ]; then
        MY_IP=$(echo "$IP_LIST" | awk '{print $1}')
        log_warn "本机 IP 不在 IP_LIST 中，fallback 使用 $MY_IP"
        log_warn "  本机 IP: $(echo "$LOCAL_IPS" | tr '\n' ' ')"
        log_warn "  IP_LIST: $IP_LIST"
    fi
    # 用完整 IP 作为 INET_PREFIX（训练脚本里 grep "inet $INET_PREFIX" 会精确匹配到本行）
    INET_PREFIX="$MY_IP"
    log_info "推导 INET_PREFIX = $INET_PREFIX （完整 IP，精确匹配）"
fi

log_info "  IP_LIST:     $IP_LIST"
log_info "  INET_PREFIX: $INET_PREFIX"
log_info "  节点数:      $(echo "$IP_LIST" | wc -w)"

# ========================= Smoke 模式参数覆盖 ===============================
if [ "$MODE" = "smoke" ]; then
    log_info "========== Smoke 模式 =========="
    log_info "  将用更小规模配置快速验证（2 iter，seq_len=1024）"

    # 原训练脚本位于 MindSpeedRun 目录（git 仓库），不能直接 sed 修改。
    # 复制一份到 logs/ 下修改。mktemp 保证绝对唯一（即使 PID 复用、同一秒内连续调用）。
    # 注：template 末尾必须是 XXXXXX（BSD/macOS 硬性要求），不带 .sh 后缀—
    #     bash 执行脚本不依赖扩展名。
    mkdir -p "$WORK_DIR/logs"
    SMOKE_SCRIPT="$(mktemp "$WORK_DIR/logs/_smoke_XXXXXX")" \
        || die "mktemp 创建 smoke 临时脚本失败"
    # 退出时（正常结束 / Ctrl+C / 异常）清理临时脚本，避免 logs/ 堆积
    trap 'rm -f "$SMOKE_SCRIPT"' EXIT
    cp "$TRAIN_SCRIPT" "$SMOKE_SCRIPT"
    # 单次 sed 多个 -e，只产生一份 .tmp（macOS BSD sed 要求 -i 带 suffix；Linux GNU sed 兼容）
    sed -i.tmp \
        -e 's|^SEQ_LEN=.*|SEQ_LEN=1024|' \
        -e 's|^GBS=.*|GBS=8|' \
        -e 's|--train-iters 10|--train-iters 2|' \
        -e 's|--seq-length \${SEQ_LEN}|--seq-length 1024|' \
        "$SMOKE_SCRIPT"
    rm -f "${SMOKE_SCRIPT}.tmp"
    chmod +x "$SMOKE_SCRIPT"
    TRAIN_SCRIPT="$SMOKE_SCRIPT"
    log_info "  smoke 脚本: $TRAIN_SCRIPT （退出时自动删除）"
fi

# ========================= 启动训练 =========================================
log_info "========== 启动训练 =========="
log_info "训练脚本: $TRAIN_SCRIPT"
log_info "执行目录: $LLM_DIR"
log_info ""
log_info "执行命令:"
log_info "  cd $LLM_DIR"
log_info "  bash $TRAIN_SCRIPT \"$IP_LIST\" $INET_PREFIX"
log_info ""

cd "$LLM_DIR" || die "无法进入 MindSpeed-LLM 目录: $LLM_DIR"
mkdir -p logs

# 实际启动训练
# 注意：不能用 exec —— exec 替换当前进程后，EXIT trap 不会触发，
#       smoke 模式下的临时脚本会被泄露。必须用 bash + wait 语义，
#       训练结束后脚本正常退出，trap 清理 SMOKE_SCRIPT。
bash "$TRAIN_SCRIPT" "$IP_LIST" "$INET_PREFIX"
