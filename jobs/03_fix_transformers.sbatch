#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=fix_trf
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/fix_trf-%j.out
#SBATCH --error=logs/fix_trf-%j.err

# Final fix: install transformers>=4.51 (required for Qwen3-4B)
# with matching tokenizers and huggingface_hub.
# This restores compatibility to what you had before the breakage.

set -euo pipefail

echo "HOST=$(hostname)"
date

module purge
module load rhel8/default-amp 2>/dev/null || true

cd /rds/user/iv294/hpc-work/thesis/project
source venv/bin/activate

echo "=== Before ==="
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "import tokenizers; print('tokenizers', tokenizers.__version__)"
python -c "import huggingface_hub; print('huggingface_hub', huggingface_hub.__version__)"

echo
echo "=== Installing correct versions ==="
pip install --no-cache-dir \
    "transformers>=4.51.0,<5.0.0" \
    "tokenizers>=0.21,<0.22" \
    "huggingface_hub>=0.30.0,<1.0.0"

echo
echo "=== After ==="
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "import tokenizers; print('tokenizers', tokenizers.__version__)"
python -c "import huggingface_hub; print('huggingface_hub', huggingface_hub.__version__)"

echo
echo "=== Full smoke test ==="
python -c "import numpy; print('numpy', numpy.__version__)"
python -c "import pandas; print('pandas', pandas.__version__)"
python -c "import torch; print('torch', torch.__version__)"
python -c "import scipy; print('scipy', scipy.__version__)"
python -c "import sklearn; print('sklearn', sklearn.__version__)"
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B')
print('Qwen3 tokenizer loaded OK, vocab size:', tok.vocab_size)
"
python -c "
import torch
a = torch.randn(64, 64, device='cuda')
print('CUDA matmul OK:', (a @ a).sum().item())
"

echo
echo "=== All OK ==="
date
