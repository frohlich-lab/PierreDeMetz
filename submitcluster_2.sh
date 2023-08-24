#!/bin/sh
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t 3-00:00
#SBATCH -p cpu
#SBATCH --mem=32GB
#SBATCH -o snakelog.out
#SBATCH -e snakelog.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=demetzp@crick.ac.uk

ml Python/3.10.8-GCCcore-12.2.0-bare
ml Anaconda3/2023.03
source /camp/apps/eb/software/Anaconda/conda.env.sh

env_name=$(head -1 environement.yml | cut -d' ' -f2)

if conda info --envs | grep -q "$env_name"; then
    echo "Environment $env_name exists, updating..."
    conda env update -f environement.yml
else
    echo "Environment $env_name does not exist, creating..."
    conda env create -f environement.yml
fi

conda activate $env_name

export WANDB_API_KEY=########
cd inst/python/
snakemake --local-cores 1 -j 10000 \
     --slurm  --resources disk_mb=6000   --default-resources slurm_account=u_froehlichf slurm_partition=cpu










