# Llama_Pure_Compute — Hardware NCU Profiling Suite
# Force clean UTF-8 environment encoding for Python sub-processes spawned by NCU
set -euo pipefail

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export NCU_PYTHONIOENCODING=utf-8
export NSYS_PYTHONIOENCODING=utf-8
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROFILES_OUT_DIR="${PROJECT_ROOT}/profiles/ncu"

mkdir -p "${PROFILES_OUT_DIR}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# Verify Nsight Compute (ncu) is present
if ! command -v ncu &> /dev/null; then
    echo "[ERROR] 'ncu' (NVIDIA Nsight Compute CLI) was not found in PATH."
    echo "Please ensure CUDA Toolkit and Nsight Compute binaries are sourced."
    exit 1
fi

echo "==========================================================="
echo " Starting Llama_Pure_Compute Hardware Profiling (Nsight Compute)"
echo " Output Directory: ${PROFILES_OUT_DIR}"
echo "============================================================"

NCU_COMMON_FLAGS=(
    -f
    --target-processes all
    --launch-skip 25
    --launch-count 3
    --clock-control base
)

# ==============================================================================
# 1. Profile Fused RMSNorm + SwiGLU CUDA Kernel
# ==============================================================================
RMSSWIGLU_REP="${PROFILES_OUT_DIR}/profile_rmsswiglu_ga102.ncu-rep"

echo -e "\n[1/2] Profiling Fused RMSNorm + SwiGLU Kernel..."
echo "Filter Kernel: regex:rmsswiglu|fused|rmsnorm"

ncu "${NCU_COMMON_FLAGS[@]}" \
    -o "${RMSSWIGLU_REP%.ncu-rep}" \
    --kernel-name "regex:rmsswiglu|fused|rmsnorm" \
    --set full \
    python3 "${SCRIPT_DIR}/benchmark_rmsswiglu.py" --hidden-dim 4096 --inter-dim 11008

echo "[SUCCESS] Saved RMSSwiGLU report to: ${RMSSWIGLU_REP}"

# ==============================================================================
# 2. Profile Triton FlashAttention-v2 Kernel
# ==============================================================================
ATTENTION_REP="${PROFILES_OUT_DIR}/profile_flash_attn_ga102.ncu-rep"

echo -e "\n[2/2] Profiling Triton FlashAttention-v2 Kernel..."
echo "Filter Kernel: regex:flash_attn|triton_fa|attn"

ncu "${NCU_COMMON_FLAGS[@]}" \
    -o "${ATTENTION_REP%.ncu-rep}" \
    --kernel-name "regex:flash_attn|triton_fa|attn" \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section ComputeWorkloadAnalysis \
    --section WarpStateStats \
    python3 "${SCRIPT_DIR}/benchmark_attention.py"

echo "[SUCCESS] Saved FlashAttention report to: ${ATTENTION_REP}"

echo -e "\n==========================================================="
echo " Profiling Complete!"
echo " Open reports in NVIDIA Nsight Compute GUI:"
echo "   ncu-ui ${RMSSWIGLU_REP}"
echo "   ncu-ui ${ATTENTION_REP}"
echo "=================================================================="