#!/bin/bash
# ============================================================================
# DeepSeek V3 MXFP8 训练环境一键配置脚本（msadapter + MindSpore 路径）
#
# 前置条件：
#   1. CANN 已安装（Toolkit + Kernel + NNAL）
#   2. NPU 驱动固件已安装
#   3. Python 3.10.x 可用
#   4. pip3 可用
#
# 用法：
#   bash /path/to/setup_mxfp8_env.sh
#
# 行为：
#   脚本所在目录即为工作目录，所有仓库克隆到该目录下。
#   把 3 个脚本（setup_mxfp8_env.sh / prepare_data.sh / run_training.sh）
#   放到目标目录后依次执行即可，不需要任何参数。
# ============================================================================

set -e

# ----------------------------- 配置区 --------------------------------------
MINDSPORE_VERSION="2.9.0"
MEGATRON_BRANCH="core_v0.12.1"
MINDSPEED_BRANCH="master"
MINDSPEED_LLM_BRANCH="master"
MSADAPTER_REPO="https://gitcode.com/hz893/msadapter.git"
MSADAPTER_BRANCH="mxfp8_mock"
MINDSPEED_REPO="https://gitcode.com/Ascend/MindSpeed.git"
MEGATRON_REPO="https://github.com/NVIDIA/Megatron-LM.git"
MINDSPEED_LLM_REPO="https://gitcode.com/ascend/MindSpeed-LLM.git"
MINDSPEEDRUN_REPO="https://gitcode.com/RyanWang1022/MindSpeedRun.git"
MINDSPEEDRUN_BRANCH="llm0121_gitcode"
# ---------------------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# 把命令的退出码捕获用于报错。
# 不能沿用"命令；if [ $? -ne 0 ]"模式 —— 在 set -e 下，失败的命令会立刻终止脚本，
# 后续的 check 永远没机会运行。正确做法是把命令和错误处理连接在同一逻辑表达式里：
#     cmd || die "msg"
# die() 只负责打印错误并退出，不依赖 $?。
die() {
    log_error "$*"
    exit 1
}

# 卸载真实 torch / torch_npu / torchvision / torchaudio
# （它们被 transformers / accelerate / peft / gpytorch 等依赖传递性拉入，
#   而 msadapter 通过 PYTHONPATH 代理 import torch → MindSpore，
#   两者共存会导致 import 顺序混乱，必须彻底清掉真实包）
cleanup_torch() {
    local removed=0
    for pkg in torch torch-npu torch_npu torchvision torchaudio; do
        if pip3 show "$pkg" >/dev/null 2>&1; then
            log_warn "  [torch-cleanup] 卸载 $pkg"
            pip3 uninstall --yes "$pkg" >/dev/null 2>&1 || true
            removed=$((removed + 1))
        fi
    done
    if [ $removed -gt 0 ]; then
        log_info "  [torch-cleanup] 共清理 $removed 个 torch 相关包"
    else
        log_info "  [torch-cleanup] 无 torch 相关包，跳过"
    fi
}

# ========================= Step 0: 前置检查 + 生成环境脚本 ==================
log_info "========== Step 0: 前置检查 =========="

# Python 版本
PYTHON_VER=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo "$PYTHON_VER" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VER" | cut -d. -f2)
if [ "$PYTHON_MAJOR" -ne 3 ] || [ "$PYTHON_MINOR" -ne 10 ]; then
    log_warn "Python 版本为 $PYTHON_VER，推荐 3.10.x"
else
    log_info "Python 版本: $PYTHON_VER"
fi

# CANN 仅做检查，不主动 source（机器自带，用户 shell 已加载）
if [ -z "$ASCEND_HOME_PATH" ]; then
    log_warn "ASCEND_HOME_PATH 未设置"
    log_warn "如后续 pip 安装或训练报错，请先 source 你环境中的 CANN set_env.sh"
else
    log_info "CANN 环境: $ASCEND_HOME_PATH"
fi

# 工作目录 = 脚本所在目录
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || die "无法定位脚本所在目录"
[ -n "$WORK_DIR" ] || die "工作目录路径为空"
cd "$WORK_DIR" || die "无法进入工作目录: $WORK_DIR"
log_info "工作目录: $WORK_DIR"

# ★ 提前生成 env_mxfp8.sh（即使后续失败也能 source 做手动恢复）
ENV_SCRIPT="$WORK_DIR/env_mxfp8.sh"
cat > "$ENV_SCRIPT" << 'EOF'
#!/bin/bash
# DeepSeek V3 MXFP8 训练环境变量
# 用法: source ./env_mxfp8.sh
#
# 自动以本文件所在目录为工作目录，无需配置。

_THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MXFP8_WORK_DIR="$_THIS_DIR"
export MXFP8_LLM_DIR="$_THIS_DIR/MindSpeed-LLM"
export MXFP8_DATA_DIR="$_THIS_DIR/dataset/deepseek3"

# msadapter（代理 import torch → MindSpore）
export PYTHONPATH="$_THIS_DIR/msadapter:$_THIS_DIR/msadapter/msa_thirdparty:$PYTHONPATH"

if [ -z "$ASCEND_HOME_PATH" ]; then
    echo "[MXFP8 ENV] WARN: ASCEND_HOME_PATH 未设置，请确认 CANN 已 source"
fi

echo "[MXFP8 ENV] 环境变量已加载 ($_THIS_DIR)"
unset _THIS_DIR
EOF
chmod +x "$ENV_SCRIPT"
log_info "env_mxfp8.sh 已生成: $ENV_SCRIPT"

# git 配置（大仓库拉取优化）
git config --global http.postBuffer 524288000
git config --global http.lowSpeedLimit 0
git config --global http.lowSpeedTime 999999

# ========================= Step 1: 安装 MindSpore ===========================
log_info "========== Step 1: 安装 MindSpore =========="

if python3 -c "import mindspore; print(mindspore.__version__)" 2>/dev/null; then
    MS_VER=$(python3 -c "import mindspore; print(mindspore.__version__)")
    log_info "MindSpore 已安装: $MS_VER"
else
    log_info "安装 MindSpore $MINDSPORE_VERSION ..."
    pip3 install mindspore==${MINDSPORE_VERSION} || die "MindSpore 安装失败"
    log_info "MindSpore $MINDSPORE_VERSION 安装成功"
fi

# ========================= Step 2: 克隆 msadapter ===========================
log_info "========== Step 2: 克隆 msadapter =========="

if [ -d "msadapter/.git" ]; then
    log_info "msadapter 已存在，跳过克隆"
else
    git clone "$MSADAPTER_REPO" -b "$MSADAPTER_BRANCH" || die "msadapter 克隆失败"
    log_info "msadapter 克隆成功"
fi

# 验证 mxfp8_mock.py 是否存在
if [ ! -f "msadapter/msa_thirdparty/torch_npu/mxfp8_mock.py" ]; then
    log_error "msadapter/msa_thirdparty/torch_npu/mxfp8_mock.py 不存在"
    log_error "请确认 msadapter 分支包含 MXFP8 mock 代码"
    exit 1
fi
log_info "mxfp8_mock.py 已确认存在"

# ========================= Step 3: 克隆 MindSpeed ===========================
log_info "========== Step 3: 克隆并安装 MindSpeed =========="

if [ -d "MindSpeed/.git" ]; then
    log_info "MindSpeed 已存在，跳过克隆"
else
    git clone --depth 1 "$MINDSPEED_REPO" -b "$MINDSPEED_BRANCH" || die "MindSpeed 克隆失败"
    log_info "MindSpeed 克隆成功"
fi

(
    cd MindSpeed
    pip3 install -r requirements.txt || die "MindSpeed 依赖安装失败"
    pip3 install -e . || die "MindSpeed 安装失败"
)
log_info "MindSpeed 安装成功"

# ========================= Step 4: 克隆 Megatron-LM =========================
log_info "========== Step 4: 克隆 Megatron-LM =========="

if [ -d "Megatron-LM/.git" ]; then
    log_info "Megatron-LM 已存在，跳过克隆"
else
    git clone --depth 1 "$MEGATRON_REPO" -b "$MEGATRON_BRANCH" \
        || die "Megatron-LM 克隆失败（如 github 不可达，可用 gitee 镜像）"
    log_info "Megatron-LM 克隆成功"
fi

# ========================= Step 5: 克隆 MindSpeed-LLM =======================
log_info "========== Step 5: 克隆 MindSpeed-LLM =========="

if [ -d "MindSpeed-LLM/.git" ]; then
    log_info "MindSpeed-LLM 已存在，跳过克隆"
else
    git clone "$MINDSPEED_LLM_REPO" -b "$MINDSPEED_LLM_BRANCH" || die "MindSpeed-LLM 克隆失败"
    log_info "MindSpeed-LLM 克隆成功"
fi

# 复制 megatron 到 MindSpeed-LLM
if [ -d "MindSpeed-LLM/megatron" ]; then
    log_info "MindSpeed-LLM/megatron 已存在，跳过复制"
else
    cp -r Megatron-LM/megatron MindSpeed-LLM/ || die "megatron 目录复制失败"
    log_info "megatron 目录复制成功"
fi

# 安装 MindSpeed-LLM 依赖（fail-fast：datasets/transformers/ray 等是后续必需依赖，
# 不能像之前那样 warn 后继续，否则 prepare_data.sh wikitext 模式会失败；训练也可能因
# 缺关键包而在中途崩。如果确实有无法安装的单个包，请手动从 requirements.txt 挑出后重跑。）
(
    cd MindSpeed-LLM
    pip3 install -r requirements.txt || die "MindSpeed-LLM 依赖安装失败，请检查上方 pip 输出。若某个非必需包无法安装，可手动修改 MindSpeed-LLM/requirements.txt 后重跑。"
    mkdir -p logs
)

# ========================= Step 6: 克隆 MindSpeedRun（训练脚本 + patch）======
log_info "========== Step 6: 克隆 MindSpeedRun =========="

if [ -d "MindSpeedRun/.git" ]; then
    log_info "MindSpeedRun 已存在，跳过克隆"
else
    git clone "$MINDSPEEDRUN_REPO" -b "$MINDSPEEDRUN_BRANCH" || die "MindSpeedRun 克隆失败"
    log_info "MindSpeedRun 克隆成功"
fi

# 自动改写训练脚本中硬编码的数据路径，指向工作目录
DATA_BASE="$WORK_DIR/dataset/deepseek3"
TRAIN_SCRIPT="$WORK_DIR/MindSpeedRun/scripts/deepseek3_swap/8k_fp8_sbh_8p.sh"
if [ -f "$TRAIN_SCRIPT" ]; then
    if grep -q "^DATA_PATH=\"$DATA_BASE" "$TRAIN_SCRIPT" 2>/dev/null; then
        log_info "训练脚本路径已对齐工作目录，跳过改写"
    else
        # 单次 sed 多个 -e 表达式，只产生一份 .bak（避免第二次 sed 覆盖掉第一次的备份）
        sed -i.bak \
            -e "s|^DATA_PATH=.*|DATA_PATH=\"$DATA_BASE/enwiki_text_document\"|" \
            -e "s|^TOKENIZER_PATH=.*|TOKENIZER_PATH=\"$DATA_BASE\"|" \
            "$TRAIN_SCRIPT"
        log_info "训练脚本数据路径已改写 → $DATA_BASE"
        log_info "  （原始文件备份在 ${TRAIN_SCRIPT}.bak）"
    fi
fi

# ========================= Step 7: 应用 Patch ===============================
log_info "========== Step 7: 应用 Patch =========="

# 三态判定 patch 状态（过去只判断两态会把 conflict 错当作 already-applied 吞掉）：
#   1. 能干净正向应用 → 应用
#   2. 能干净反向应用 → 已应用，跳过
#   3. 两者都失败   → 真冲突（环境不一致），die
apply_patch_safely() {
    local patch_path="$1"
    local patch_name
    patch_name="$(basename "$patch_path")"
    if git apply --check "$patch_path" 2>/dev/null; then
        git apply "$patch_path" || die "$patch_name 应用失败"
        log_info "$patch_name 应用成功"
    elif git apply --reverse --check "$patch_path" 2>/dev/null; then
        log_info "$patch_name 已应用，跳过"
    else
        die "$patch_name 冲突（环境与 patch 不一致，可能是仓库版本不对或已被其他 patch 修改过）。请检查 MindSpeed-LLM 目录状态或重新 clone。"
    fi
}

# 子 shell 隔离 cd：即使中间某个 git apply 在 set -e 下触发退出，
# 主 shell 的 CWD 也会自动还原为 $WORK_DIR（对 source 执行尤其重要）
(
    cd MindSpeed-LLM

    # MindSpeed-LLM patch（HCCL 通信模式）
    if [ -f "../MindSpeedRun/patch/MindSpeed-LLM.patch" ]; then
        apply_patch_safely "../MindSpeedRun/patch/MindSpeed-LLM.patch"
    fi

    # Megatron 源码 patch（逐个应用）
    for patch_file in pretrain_gpt.patch base.patch moe_utils.patch training.patch gpt_dataset.patch; do
        patch_path="../MindSpeedRun/run/$patch_file"
        if [ -f "$patch_path" ]; then
            apply_patch_safely "$patch_path"
        fi
    done
)

# ========================= Step 8: 环境验证 + 最终清理 ======================
log_info "========== Step 8: 最终清理 torch（让 msadapter 代理生效）=========="

# 前面所有 pip install 都会通过 transformers / accelerate / peft / gpytorch
# 等传递依赖拉入真实 torch。由于 pip 基于 dist-info 判断依赖是否满足，无论
# msadapter 何时加入 PYTHONPATH 都无法阻止 pip 安装真实 torch。因此统一在
# 所有 pip 完成后执行唯一一次清理即可。
cleanup_torch

log_info "========== Step 8: 环境验证 =========="

# 临时设置 PYTHONPATH 用于验证
export PYTHONPATH="$WORK_DIR/msadapter:$WORK_DIR/msadapter/msa_thirdparty:$PYTHONPATH"

# 验证 MindSpore（stderr 保留，失败时用户能看到真实 traceback）
python3 -c "import mindspore; print(f'MindSpore: {mindspore.__version__}')" \
    || die "MindSpore import 失败"
log_info "MindSpore 验证通过"

# 验证 msadapter 代理：torch.__file__ 必须指向 msadapter 路径，不能是 site-packages
# 注意："只要不含 msadapter 就报错"，覆盖 torch.__file__ 为空、torch 是命名空间包等边缘情况
python3 -c "
import sys
import msadapter
import torch
torch_path = getattr(torch, '__file__', '') or ''
if 'msadapter' not in torch_path:
    print(f'[ERROR] torch 未通过 msadapter 代理: {torch_path!r}', file=sys.stderr)
    print('[ERROR] 请重新运行 setup_mxfp8_env.sh 或手动: pip uninstall torch torch_npu torchvision', file=sys.stderr)
    sys.exit(1)
print(f'torch (via msadapter): {torch.__version__}')
print(f'torch path: {torch_path}')
" || die "msadapter 验证失败（torch 未正确代理）"
log_info "msadapter 代理 torch 验证通过"

# 验证 MXFP8 mock
python3 -c "
import msadapter
import torch_npu
checks = [
    hasattr(torch_npu, 'float8_e8m0fnu'),
    hasattr(torch_npu, 'npu_dynamic_mx_quant'),
    hasattr(torch_npu, 'npu_dynamic_mx_quant_with_dual_axis'),
    hasattr(torch_npu, 'npu_quant_matmul'),
    hasattr(torch_npu, 'npu_add_quant_matmul_'),
    hasattr(torch_npu, 'npu_grouped_matmul'),
    hasattr(torch_npu, 'npu_add_quant_gmm_'),
    hasattr(torch_npu, 'npu_all_gather_quant_mm'),
    hasattr(torch_npu, 'npu_quant_mm_reduce_scatter'),
    hasattr(torch_npu, 'npu_grouped_dynamic_mx_quant'),
    hasattr(torch_npu, 'hifloat8'),
]
passed = sum(checks)
total = len(checks)
print(f'MXFP8 mock 算子: {passed}/{total} 通过')
assert passed == total, f'{total - passed} 个算子缺失'
" || die "MXFP8 mock 验证失败"
log_info "MXFP8 mock 全部 11 项验证通过"

# 验证 MindSpeed
python3 -c "import mindspeed; print(f'MindSpeed: {mindspeed.__version__}')" \
    || die "MindSpeed import 失败"
log_info "MindSpeed 验证通过"

# 验证 pretrain_gpt.patch 已应用
# 不用 grep 固定字符串（MindSpeed-LLM 版本升级后 import 行可能变化，导致误报）
# 改用 git apply --reverse --check：能干净反向应用才说明 patch 已生效
PATCH_FILE="$WORK_DIR/MindSpeedRun/run/pretrain_gpt.patch"
if [ -f "$PATCH_FILE" ]; then
    if (cd MindSpeed-LLM && git apply --reverse --check "$PATCH_FILE") 2>/dev/null; then
        log_info "pretrain_gpt.patch 已应用验证通过"
    else
        log_warn "pretrain_gpt.patch 未应用或状态异常，请重新运行 Step 7"
    fi
fi

# ========================= 完成 =============================================
echo ""
log_info "=============================================="
log_info "  MXFP8 训练环境配置完成"
log_info "=============================================="
echo ""
echo "  工作目录 ($WORK_DIR) 结构:"
echo "    ├── setup_mxfp8_env.sh   # 本脚本（一次性部署）"
echo "    ├── prepare_data.sh      # 数据准备"
echo "    ├── run_training.sh      # 一键启动训练"
echo "    ├── env_mxfp8.sh         # 环境变量（每次 source）"
echo "    ├── msadapter/           # import torch → MindSpore"
echo "    ├── MindSpeed/           # MXFP8 monkey-patch 机制"
echo "    ├── Megatron-LM/         # core_v0.12.1"
echo "    ├── MindSpeed-LLM/       # 训练主目录（含 megatron/ + patch）"
echo "    ├── MindSpeedRun/        # 训练脚本 + patch（路径已自动对齐）"
echo "    └── dataset/deepseek3/   # 数据和 tokenizer（prepare_data.sh 生成）"
echo ""
echo "  下一步（在工作目录下依次执行）:"
echo "    cd $WORK_DIR"
echo "    ./prepare_data.sh smoke      # 准备数据"
echo "    ./run_training.sh smoke      # 启动训练（自动 source env）"
echo ""
