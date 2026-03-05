#!/bin/bash
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --job-name=fix_numpy
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/fix_numpy-%j.out
#SBATCH --error=logs/fix_numpy-%j.err

set -euo pipefail

echo "HOST=$(hostname)"
date

module purge
module load rhel8/default-amp 2>/dev/null || true

cd /rds/user/iv294/hpc-work/thesis/project
source venv/bin/activate

echo "Before: numpy=$(python -c 'import numpy; print(numpy.__version__)')"

# Pin numpy back to 1.26.4 — pandas 2.1.0 was compiled against numpy 1.x
pip install --no-cache-dir numpy==1.26.4

echo "After:  numpy=$(python -c 'import numpy; print(numpy.__version__)')"

echo
echo "--- Quick smoke test ---"
python -c "import pandas; print('pandas', pandas.__version__, 'OK')"
python -c "import torch; print('torch', torch.__version__, 'OK')"
python -c "import scipy; print('scipy', scipy.__version__, 'OK')"
python -c "import sklearn; print('sklearn', sklearn.__version__, 'OK')"
python -c "import tokenizers; print('tokenizers', tokenizers.__version__, 'OK')"
python -c "import transformers; print('transformers', transformers.__version__, 'OK')"
python -c "import torch; a=torch.randn(64,64,device='cuda'); print('CUDA matmul OK:', (a@a).sum().item())"

echo
echo "=== All OK ==="
date
