#!/bin/bash
#----------------------------------
# Specifying grid engine options
#----------------------------------
#$ -S /bin/bash
# the working directory where the commands below will
# be executed: (make sure to specify)
#$ -wd /data/users/nchilwant/projects/mmf
#
# logging files will go here: (make sure to specify)
#$ -e /data/users/nchilwant/log/ -o /data/users/nchilwant/log/
#
# Specify the node on which to run
#$ -l hostname=cl10lx
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
export MMF_DATA_DIR="/data/users/nchilwant/dataset/HM/mmf/data"
export MMF_CACHE_DIR="/data/users/nchilwant/dataset/HM/cache"
export CUDA_VISIBLE_DEVICES=1

#VisualBERT
#echo "starting the train phase"
#export MMF_SAVE_DIR="/data/users/nchilwant/visual_bert_coco"
#CUDA_VISIBLE_DEVICES=1,2 mmf_run config=projects/hateful_memes/configs/visual_bert/from_coco.yaml \
#    run_type=train_val \
#    dataset=hateful_memes \
#    model=visual_bert \
##    training.batch_size_per_device=16 \
##    training.evaluation_interval=100 \
#    training.checkpoint_interval=100000

#echo "starting the validation phase"
#CUDA_VISIBLE_DEVICES=0,1,2,3 mmf_predict config=projects/hateful_memes/configs/visual_bert/from_coco.yaml \
#    model=visual_bert \
#    dataset=hateful_memes \
#    run_type=val \
#    checkpoint.resume_file=./save/visual_bert_final.pth \
#    checkpoint.resume_pretrained=False
##    training.batch_size_per_device=32

#echo "starting the test phase"
#CUDA_VISIBLE_DEVICES=1 mmf_predict config=projects/hateful_memes/configs/visual_bert/from_coco.yaml \
#    model=visual_bert \
#    dataset=hateful_memes \
#    run_type=test \
#    checkpoint.resume_file=./save/visual_bert_final.pth \
#    checkpoint.resume_pretrained=False \
##    training.batch_size_per_device=32 \
#    dataset_config.hateful_memes.annotations.val[0]=hateful_memes/defaults/annotations/dev_seen.jsonl \
#    dataset_config.hateful_memes.annotations.test[0]=hateful_memes/defaults/annotations/test_seen.jsonl

#MMBT
#export MMF_SAVE_DIR="/data/users/nchilwant/mmbt"
#mmf_run config=projects/hateful_memes/configs/mmbt/defaults.yaml \
#    model=mmbt \
#    dataset=hateful_memes \
#    run_type=train_val \
#    training.checkpoint_interval=100000 \
#    training.batch_size_per_device=16
#
#mmf_run config=projects/hateful_memes/configs/mmbt/defaults.yaml \
#    model=mmbt \
#    dataset=hateful_memes \
#    run_type=val \
#    checkpoint.resume_file=./save/mmbt_final.pth \
#    checkpoint.resume_pretrained=False
#
#mmf_predict config=projects/hateful_memes/configs/mmbt/defaults.yaml \
#    model=mmbt \
#    dataset=hateful_memes \
#    run_type=test \
#    checkpoint.resume_file=./save/mmbt_final.pth \
#    checkpoint.resume_pretrained=False

#ConcatBERT
#echo "training concat_BERT"
#export MMF_SAVE_DIR="./concat_BERT"
#mmf_run config="projects/hateful_memes/configs/concat_bert/defaults.yaml" \
#    model=concat_bert \
#    dataset=hateful_memes \
#    run_type=train_val \
#    training.max_epochs=20 \
#    training.batch_size_per_device=16 \
#    training.evaluation_interval=100

#echo "training mutan_BERT"
export MMF_SAVE_DIR="./mutan_BERT"
mmf_run config="projects/hateful_memes/configs/concat_bert/mutan_bert.yaml" \
    model=mutan_bert \
    dataset=hateful_memes \
    run_type=train_val \
    training.max_epochs=10 \
    training.batch_size_per_device=16 \
#    training.evaluation_interval=1000

#mmf_predict config="projects/hateful_memes/configs/concat_bert/defaults.yaml" \
#    model=<REPLACE_WITH_MODEL_KEY> dataset=hateful_memes \
#    run_type=test checkpoint.resume_file=<path_to_best_trained_model> checkpoint.resume_pretrained=False

echo "Finished execution."