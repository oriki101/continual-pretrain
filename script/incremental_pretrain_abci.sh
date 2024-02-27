#!/bin/bash
#$ -l rt_F=2
#$ -l h_rt=0:05:00
#$ -l USE_BEEOND=1
#$ -j y
#$ -o logs/
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.10
module load cuda/11.7/11.7.1
module load cudnn/8.5/8.5.0
module load nccl/2.14/2.14.3-1
module load hpcx/2.12


source ~/llm-env/bin/activate

GPUS_PER_NODE=4
NNODES=2
NUM_GPUS=$((${GPUS_PER_NODE} * ${NNODES}))
echo ${NUM_GPUS}
echo ${GPUS_PER_NODE}


# マルチノード用のアドレスの設定
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=50000
echo ${MASTER_ADDR}
echo ${MASTER_PORT}

# hostfile作成
HOSTFILE_NAME=./hostfile/hostfile_${JOB_ID}
while read -r line
do
  echo "${line} slots=${GPUS_PER_NODE}"
done < "$SGE_JOB_HOSTLIST" > "$HOSTFILE_NAME"

# huggingfaceのモデル保存先を指定(容量対策)
export HF_HOME=/scratch/$(whoami)/.cache/huggingface/


mpirun -np ${NUM_GPUS} \
	-npernode ${GPUS_PER_NODE} \
	-hostfile $HOSTFILE_NAME \
	-x MASTER_ADDR=$MASTER_ADDR \
	-x MASTER_PORT=$MASTER_PORT \
	-bind-to none -map-by slot \
	-x NCCL_DEBUG=INFO  -x PATH \
	-mca pml ob1 -mca btl ^openib \
	-mca coll ^hcoll \
	--mca btl_tcp_if_include eno1 \
	python train_deepspeed.py

