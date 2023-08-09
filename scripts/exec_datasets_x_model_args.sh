#!/bin/bash
#SBATCH -p bosch_gpu-rtx2080
#SBATCH --job-name dinov2_meta_album_vitg14_lora
#SBATCH -o experiments/metaalbum/vitg14_lora/logs/%A-%a.%x.out
#SBATCH -e experiments/metaalbum/vitg14_lora/logs/%A-%a.%x.err
#SBATCH --gres=gpu:1
#SBATCH -t 20:00:00
#SBATCH --array 1-30%20

#source /home/ferreira/.miniconda/bin/activate dinov2
#export PYTHONPATH="${PYTHONPATH}:${HOME}/AutoFinetune"
source ~/.profile
conda activate dinov2

ARGS_FILE=experiments/vitg14_linear_meta_album_lora.args
TASK_SPECIFIC_ARGS=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $ARGS_FILE)

echo $TASK_SPECIFIC_ARGS
python dinov2/eval/linear.py $TASK_SPECIFIC_ARGS

