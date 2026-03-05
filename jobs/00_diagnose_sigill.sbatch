#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=sigill_dx
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/sigill_dx-%j.out
#SBATCH --error=logs/sigill_dx-%j.err

echo "HOST=$(hostname)"
echo "JOBID=$SLURM_JOB_ID"
date
echo

# Purge Intel/Cascade Lake modules (AVX-512 -> SIGILL on AMD EPYC)
module purge
module load rhel8/default-amp 2>/dev/null || module load rhel8/default-gpu 2>/dev/null || true

cd /rds/user/iv294/hpc-work/thesis/project
source venv/bin/activate

export ATEN_CPU_CAPABILITY=avx2
export MKL_ENABLE_INSTRUCTIONS=AVX2
export TOKENIZERS_PARALLELISM=false

# Each import runs in its own subprocess.
# SIGILL = exit code 132. Only that subprocess dies.
test_import() {
    echo -n "  $1 ... "
    output=$(python -c "$2" 2>&1)
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "OK  ($output)"
    elif [ $rc -eq 132 ]; then
        echo "*** SIGILL (exit 132) *** <-- CULPRIT"
    else
        echo "FAILED (exit $rc): $output"
    fi
}

echo "=== Phase 1: Isolate which import crashes ==="
test_import "bare python"    "print('ok')"
test_import "numpy"          "import numpy; print(numpy.__version__)"
test_import "pandas"         "import pandas; print(pandas.__version__)"
test_import "yaml"           "import yaml; print(yaml.__version__)"
test_import "torch"          "import torch; print(torch.__version__)"
test_import "torch.cuda"     "import torch; print(torch.cuda.is_available())"
test_import "safetensors"    "import safetensors; print(safetensors.__version__)"
test_import "tokenizers"     "import tokenizers; print(tokenizers.__version__)"
test_import "transformers"   "import transformers; print(transformers.__version__)"
test_import "accelerate"     "import accelerate; print(accelerate.__version__)"
test_import "huggingface_hub" "import huggingface_hub; print(huggingface_hub.__version__)"
test_import "regex"          "import regex; print(regex.__version__)"
test_import "scipy"          "import scipy; print(scipy.__version__)"
test_import "sklearn"        "import sklearn; print(sklearn.__version__)"
test_import "networkx"       "import networkx; print(networkx.__version__)"

echo
echo "=== Phase 2: Compute tests ==="
test_import "numpy matmul"     "import numpy as np; print((np.random.randn(64,64) @ np.random.randn(64,64)).sum())"
test_import "torch CPU matmul" "import torch; print((torch.randn(64,64) @ torch.randn(64,64)).sum().item())"
test_import "torch CUDA"       "import torch; a=torch.randn(64,64,device='cuda'); print((a@a).sum().item())"
test_import "torch bf16 CUDA"  "import torch; a=torch.randn(64,64,device='cuda',dtype=torch.bfloat16); print('ok')"

echo
echo "=== Phase 3: Model loading chain ==="
test_import "AutoTokenizer"    "from transformers import AutoTokenizer; print('ok')"
test_import "AutoModelForCausalLM" "from transformers import AutoModelForCausalLM; print('ok')"
test_import "src.model_utils"  "import sys; sys.path.insert(0,'.'); from src.model_utils import ModelWrapper; print('ok')"
test_import "src.transcoder"   "import sys; sys.path.insert(0,'.'); from src.transcoder import load_transcoder_set; print('ok')"

echo
echo "=== Phase 4: Versions ==="
pip list --format=freeze 2>/dev/null | grep -iE "torch|numpy|token|triton|transformers|safetensors|accelerate|regex|scipy|sklearn"

echo
echo "=== Done ==="
date
