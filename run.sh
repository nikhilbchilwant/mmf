#!/bin/bash
#----------------------------------
# Specifying grid engine options
#----------------------------------
#$ -S /bin/bash
# the working directory where the commands below will
# be executed: (make sure to specify)
#$ -wd /data/users/nchilwant/mmf  
#
# logging files will go here: (make sure to specify)
#$ -e /data/users/nchilwant/log/ -o /data/users/nchilwant/log/
#
# Specify the node on which to run
#$ -l hostname=cl13lx
#----------------------------------
#  Running some bash commands
#----------------------------------
export PATH="/nethome/nchilwant/miniconda3/bin:$PATH"
source activate mmf
nvidia-smi
#----------------------------------
# Running your code (here we run some python script as an example)
#----------------------------------
pwd
#CUDA_VISIBLE_DEVICES=0,1 mmf_run config=projects/hateful_memes/configs/visual_bert/defaults.yaml model=visual_bert dataset=hateful_memes run_type=train_val_test training.max_epochs=10 evaluation.predict=true
#CUDA_VISIBLE_DEVICES=0,1 mmf_predict config=projects/hateful_memes/configs/visual_bert/defaults.yaml model=visual_bert dataset=hateful_memes run_type=train_val
CUDA_VISIBLE_DEVICES=0,1,2,3 mmf_run config=projects/hateful_memes/configs/mmbt/defaults.yaml \
    model=mmbt \
    dataset=hateful_memes \
    run_type=train_val \
    training.batch_size=32
CUDA_VISIBLE_DEVICES=0,1,2,3 mmf_run config=projects/hateful_memes/configs/mmbt/defaults.yaml \
    model=mmbt \
    dataset=hateful_memes \
    run_type=val \
    checkpoint.resume_file=./save/mmbt_final.pth \
    checkpoint.resume_pretrained=False \
    training.batch_size=32
echo "Finished execution."