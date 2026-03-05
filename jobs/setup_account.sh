#!/bin/bash
# Run this script to set your SLURM account in all job files
#
# Usage: ./setup_account.sh YOUR-CPU-ACCOUNT
# Example: ./setup_account.sh COMPUTERLAB-SL2-CPU
#
# All jobs now run on CPU for initial pipeline testing

if [ $# -ne 1 ]; then
    echo "Usage: $0 <CPU_ACCOUNT>"
    echo "Example: $0 COMPUTERLAB-SL2-CPU"
    echo ""
    echo "To find your accounts, run: sacctmgr show assoc user=\$USER format=account%30"
    exit 1
fi

CPU_ACCOUNT=$1

echo "Setting CPU account to: $CPU_ACCOUNT"
echo ""

# All jobs use CPU
sed -i "s/CHANGEME-SL2-CPU/$CPU_ACCOUNT/g" jobs/01_generate_prompts.sh
sed -i "s/CHANGEME-SL2-CPU/$CPU_ACCOUNT/g" jobs/02_run_baseline.sh
sed -i "s/CHANGEME-SL2-CPU/$CPU_ACCOUNT/g" jobs/03_capture_activations.sh
sed -i "s/CHANGEME-SL2-CPU/$CPU_ACCOUNT/g" jobs/04_extract_features.sh
sed -i "s/CHANGEME-SL2-CPU/$CPU_ACCOUNT/g" jobs/06_attribution_graph.sh
sed -i "s/CHANGEME-SL2-CPU/$CPU_ACCOUNT/g" jobs/07_interventions.sh
sed -i "s/CHANGEME-SL2-CPU/$CPU_ACCOUNT/g" jobs/08_figures.sh

echo "Done! Account name updated in all job scripts."
echo ""
echo "Verify with: grep -h '#SBATCH -A' jobs/*.sh"
