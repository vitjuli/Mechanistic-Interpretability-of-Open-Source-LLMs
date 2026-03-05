#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=rebuild_venv
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/rebuild_venv-%j.out
#SBATCH --error=logs/rebuild_venv-%j.err

# =============================================================================
# Rebuild venv packages ON an AMD EPYC compute node (ampere partition).
#
# WHY: The login node is Intel Xeon (Cascade Lake, has AVX-512).
#      Default modules (rhel8/cclake-env, intel-oneapi-compilers) inject
#      Intel math libraries (libsvml, libimf) compiled for Cascade Lake.
#      Packages installed under those modules may link against AVX-512 code,
#      which causes SIGILL (exit 132) on AMD EPYC 7763 (AVX2 only).
#
# FIX: Purge Intel modules, then reinstall the affected compiled packages
#      directly on an AMD compute node so wheels are built/linked for AVX2.
# =============================================================================

set -euo pipefail

echo "=== Rebuild venv for ampere (AMD EPYC) ==="
echo "HOST=$(hostname)"
echo "JOBID=$SLURM_JOB_ID"
date
echo

# ---- Step 1: Purge Intel/Cascade Lake modules ----
echo "--- Modules BEFORE purge ---"
module list 2>&1
echo

module purge
echo "--- Modules AFTER purge ---"
module list 2>&1
echo

# Load only what we need for CUDA + compilation
# (adjust module names if these don't exist on your CSD3 setup)
module load rhel8/default-amp 2>/dev/null || module load rhel8/default-gpu 2>/dev/null || true
module load cuda/12.1 2>/dev/null || module load cuda/12.2 2>/dev/null || true

echo "--- Modules LOADED ---"
module list 2>&1
echo

# ---- Step 2: Activate venv ----
cd /rds/user/iv294/hpc-work/thesis/project
source venv/bin/activate

echo "Python: $(which python)"
python --version
echo

# CPU info on the compute node
echo "--- CPU on this node ---"
lscpu | grep -E "Model name|Flags" | head -2
echo

# ---- Step 3: Reinstall the packages that had SIGILL ----
# These all crashed with exit 132 in the diagnostic:
#   pandas, torch, scipy, sklearn, tokenizers
# Force reinstall from wheels (no cached .so files)

echo "--- Reinstalling affected packages (no cache) ---"
pip install --no-cache-dir --force-reinstall \
    torch==2.4.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

pip install --no-cache-dir --force-reinstall \
    pandas==2.1.0 \
    scipy \
    scikit-learn \
    tokenizers==0.19.1

# Also reinstall triton if present
pip install --no-cache-dir --force-reinstall triton 2>/dev/null || true

echo
echo "--- Pinning transformers to avoid 5.x ---"
pip install transformers==4.44.2 huggingface_hub==0.25.0

echo
echo "--- Verifying imports ---"

test_import() {
    echo -n "  $1 ... "
    output=$(python -c "$2" 2>&1)
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "OK  ($output)"
    elif [ $rc -eq 132 ]; then
        echo "*** SIGILL (exit 132) *** STILL BROKEN"
    else
        echo "FAILED (exit $rc): $output"
    fi
}

test_import "numpy"          "import numpy; print(numpy.__version__)"
test_import "pandas"         "import pandas; print(pandas.__version__)"
test_import "torch"          "import torch; print(torch.__version__)"
test_import "torch.cuda"     "import torch; print(torch.cuda.is_available())"
test_import "tokenizers"     "import tokenizers; print(tokenizers.__version__)"
test_import "transformers"   "import transformers; print(transformers.__version__)"
test_import "scipy"          "import scipy; print(scipy.__version__)"
test_import "sklearn"        "import sklearn; print(sklearn.__version__)"
test_import "torch CPU matmul" "import torch; print((torch.randn(64,64) @ torch.randn(64,64)).sum().item())"
test_import "torch CUDA"     "import torch; a=torch.randn(64,64,device='cuda'); print((a@a).sum().item())"

echo
echo "=== Rebuild complete ==="
date
