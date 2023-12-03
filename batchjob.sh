#!/bin/sh

#  assert correct run dir
run_dir="DQN-Multiagent-ATARI-Boxing"
if ! [ "$(basename $PWD)" = $run_dir ];
then
    echo -e "\033[0;31mScript must be submitted from the directory: $run_dir\033[0m"
    exit 1
fi

# create dir for logs
mkdir -p "logs/"

### General options
### –- specify queue --
# BSUB -q gpuv100
### -- set the job Name --
#BSUB -J rl_1vRand
### -- ask for number of cores (default: 1) -- 
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now 
#BSUB -W 2:00
### -- request 5GB of system-memory --
#BSUB -R "rusage[mem=8GB]"
### -- set the email address --
##BSUB -u s194324@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion-- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o logs/rl%J.out 
#BSUB -e logs/rl%J.err 
### -- end of LSF options --

# activate env
source rl_env/bin/activate

# load additional modules
module load cuda/12.1

# run scripts
python src/main_rainbow.py