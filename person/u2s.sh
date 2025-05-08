#!/bin/bash
# shanghai

source /home/ma-user/anaconda3/etc/profile.d/conda.sh


# ywh added
env_name=u2s

if conda info --envs | grep -q ${env_name} ; then
    echo "${env_name} 环境已存在，正在激活..."
    conda activate ${env_name}
else
    echo "${env_name} 环境不存在，正在创建..."
    conda create -n ${env_name} python=3.9 -y
    conda activate ${env_name}
    cd /home/ma-user/work/yangwenhan/u2s/
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0  tensorboard
    pip install protobuf pillow Unidecode  scipy phonemizer numpy matplotlib librosa Cython
    # Cython-version Monotonoic Alignment Search torch torchvision tensorboard
    cd monotonic_align
    python setup.py build_ext --inplace
    pip install whisper_normalizer lhotse editdistance transformers psutil
fi
# pip install protobuf pillow Unidecode torch torchvision tensorboard scipy phonemizer numpy matplotlib librosa Cython
# # Cython-version Monotonoic Alignment Search
# cd monotonic_align  pip install protobuf pillow Unidecode torch torchvision tensorboard scipy phonemizer numpy matplotlib librosa Cython
# python setup.py build_ext --inplace
# pip install whisper_normalizer lhotse editdistance transformers psutil
#

CUTS="/home/ma-user/work/yangwenhan/zipformer/data/PhoneLS960offlineCtc_Ckpt200_1000fsqUnits/stu_units_UnitPhonefflineCtc_150epo"
# CUTS=/home/ma-user/work/yangwenhan/zipformer/data/LS100_offlineFTphone_ctc60epoch_fsqUnits/stu_units_Epoch220
cd /home/ma-user/work/yangwenhan/u2s/
# python my_train_ms_single_gpu.py
python /home/ma-user/work/yangwenhan/u2s/pipe_run.py \
    --cuts_path ${CUTS} \
    --train 1 --synthesis 1
# conda create -n whisper python=3.8 -y
# conda activate whisper
# pip install torch torchvision openai-whisper lhotse editdistance transformers psutil

# python /home/ma-user/work/yangwenhan/u2s/pipe_run.py 
#     --cuts_path ${CUTS} \
#     --train 0





# conda activate speechbrain2
# pip install --upgrade pip

# pip install torch==1.13.1  soundfile torchvision==0.14.1 torchaudio==0.13.1 speechbrain huggingface_hub==0.8.0 transformers==4.21.0 numpy tqdm sentencepiece scipy joblib
# pip install  hyperpyyaml black packaging pandas pygtrie
# pip install --editable .

# # ywh added above
# if [[ -z "${MA_VJ_NAME}" ]]; then   # In DevContainer
#     NPUS_PER_NODE=8 # To be customized according to your DevContainer
#     MASTER_ADDR=localhost
#     NNODES=1
#     NODE_RANK=0
# else    # Training job of ModelArts
#     NPUS_PER_NODE=$MA_NUM_GPUS
#     MASTER_ADDR="${MA_VJ_NAME}-${MA_TASK_NAME}-0.${MA_VJ_NAME}"
#     NNODES=$MA_NUM_HOSTS
#     if [ ${NNODES} -gt 1 ]; then
#         NODE_RANK=$VC_TASK_INDEX
#     else
#         NODE_RANK=0
#     fi
# fi
# MASTER_PORT=6000
# WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))
#
#
#
# base_path=$(cd $(dirname $0) && pwd)
# echo $base_path
#
# cd $base_path
#
# ls
# DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --master_addr $MASTER_ADDR --master_port $MASTER_PORT --nnodes $NNODES  --node_rank $NODE_RANK"
#
# # DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --standalone"
#
# echo $DISTRIBUTED_ARGS
#
# # torchrun $DISTRIBUTED_ARGS $base_path/train.py $base_path/$1 --find_unused_parameters
# python -m torch.distributed.run $DISTRIBUTED_ARGS $base_path/train.py $base_path/$1 --find_unused_parameters

# sleep 100000
# https://github.com/speechbrain/speechbrain/issues/2588
