#!/bin/bash

echo "\e[33mINFO: create_env.sh will create a virtual environment to use on the DTUs HPC cluster"

# set env name
env_name="rl_env"

### Make python environment
module load python3/3.8.17 # only use if on HPC
python3 -m venv $env_name

### Activate environment 
source $env_name/bin/activate

python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio
python -m pip install -r requirements.txt


if [ $( basename $PWD ) = "DQN_Multiagent-ATARI-Boxing" ]
then 
    echo "Virtual environment created at $PWD"
else
    echo "\e[33mWARN: Virtual environment was not created with DQN_Multiagent-ATARI-Boxing as basename, instead it was created at $PWD\e[0m"
fi

deactivate

# tell use to manually install some packages as they have problems being installed through requirements.txt
echo "\e[33mPlease activate the environment and use 'python -m pip install six setuptools appdirs'\e[0m"


