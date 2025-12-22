# 开源参考
**MindSpeed**: https://gitcode.com/Ascend/MindSpeed

**MindSpeed-LLM**: https://gitcode.com/Ascend/MindSpeed-LLM

**MindSpped-LLM -> Qwen3**: https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/tests/0day/qwen3

**MindSpeed/Qwen3-1.7B-Base**: https://modelers.cn/models/MindSpeed/Qwen3-1.7B-Base

**Alpaca dataset**: https://modelers.cn/datasets/AI_Connect/alpaca/tree/main/data

# 镜像及镜像启动为容器
这里 base 镜像使用 MindIE 2.2.RC1 官方镜像，参考链接如下，

MindIE 昇腾镜像仓库: https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f

## 镜像
```shell
~# docker images
REPOSITORY                                          TAG
swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie   2.2.RC1-800I-A2-py311-openeuler24.03-lts
```

## 镜像启动为容器
通过 bash run.sh 脚本运行方式将镜像启动为容器，以下为示例（ <...> 为根据实测环境自定义修改）
```shell
~# vim run.sh

#!/bin/bash
docker run -itd -u root --net=host --shm-size=500g --privileged --name <Qwen3-1point7b-Pretrain>  \
--device=/dev/davinci_manager \
--device=/dev/hisi_hdc \
--device=/dev/devmm_svm \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/sbin:/usr/local/sbin \
-v </HOST_data>:</CONTAINER_data> \
<IMAGE_name>:<IMAGE_tag> \
/bin/bash
```

# MindSpeed-LLM 代码仓及相关依赖安装
## 配套关系
参考 https://gitcode.com/Ascend/MindSpeed-LLM “版本说明”章节，后续版本更新以该章节（商用）为准，

需注意 MindSpeed LLM, Megatron, torch& torch_npu 之间的配套关系，这里以 MindSpeed LLM版本为例，仅列举商用版，

| MindSpeed LLM | MindSpped          | Megatron     | PyTorch      | torch_npu | CANN    | Python    |
| ------------- | ------------------ | ------------ | ------------ | --------- | ------- | --------- |
| 2.2.0         | 2.2.0_core_r0.12.1 | core_v0.12.1 | 2.7.1        | 7.2.0     | 8.3.RC1 | 3.10      |
| 2.1.0         | 2.1.0_core_r0.8.0  | core_r0.8.0  | 2.1.0, 2.6.0 | 7.1.0     | 8.2.RC1 | 3.8, 3.10 |

注，torch_npu 版本，以 Ascend Extension for PyTorch 相关版本配套说明为准，

注，MindIE 2.2.RC1 对应 Python 为 3.11，实测 MindSpeed LLM 为 2.2.0 可以满足（Python 版本可使用比配套版本高的），

## torch& torch_npu& apex& torchvision
这里以 MindSpped LLM 为2.2.0 版本及配套关系，作为示例，

- torch
```shell
下载软件包
~# wget https://download.pytorch.org/whl/cpu/torch-2.7.1%2Bcpu-cp311-cp311-manylinux_2_28_aarch64.whl

安装命令
~# pip3 install torch-2.7.1+cpu-cp311-cp311-manylinux_2_28_aarch64.whl
```

- torch_npu
```shell
下载插件包
~# wget https://gitcode.com/Ascend/pytorch/releases/download/v7.2.0-pytorch2.7.1/torch_npu-2.7.1-cp311-cp311-manylinux_2_28_aarch64.whl

安装命令
~# pip3 install torch_npu-2.7.1-cp311-cp311-manylinux_2_28_aarch64.whl
```

- apex
```shell
~# git clone -b master https://gitee.com/ascend/apex.git
~# cd apex/
apex# bash scripts/build.sh --python=3.11
```
这里 python3.11 可编译生成，过程中会自动拉取 apex 官方源码，请保证网络畅通，生成的二进制包在 apex/dist 目录下，
```shell
apex/apex/dist# ll apex-0.1+ascend-cp311-cp311-linux_aarch64.whl

apex/apex/dist# pip3 install --upgrade apex-0.1+ascend-cp311-cp311-linux_aarch64.whl
Processing ./apex-0.1+ascend-cp311-cp311-linux_aarch64.whl
Installing collected packages: apex
Successfully installed apex-0.1+ascend
```

- torchvision
torchvision 与 torch& torch_npu 版本配套保持对应关系，
```shell
pip3 install torchvision==0.22.1
```

## 使能环境变量
这里以 MindSpeed LLM 为 2.2.0 版本及配套关系，作为示例，
```shell
~# source /usr/local/Ascend/ascend-toolkit/set_env.sh
~# source /usr/local/Ascend/nnal/atb/set_env.sh
```

## 安装MindSpeed加速库
```shell
~# git clone https://gitcode.com/ascend/MindSpeed.git
~# cd MindSpeed
MindSpeed# git checkout 2.2.0_core_r0.12.1  # default `master`, means checkout commit from MindSpeed master
MindSpeed# pip3 install -r requirements.txt 
MindSpeed# pip3 install -e .
MindSpeed# cd ..
```

## 准备MindSpeed-LLM及Megatron-LM源码
```shell
~# git clone https://gitcode.com/ascend/MindSpeed-LLM.git 
~# git clone https://github.com/NVIDIA/Megatron-LM.git  # megatron 从 github 下载，请确保网络能访问
~# cd Megatron-LM
Megatron-LM# git checkout core_v0.12.1
Megatron-LM# cp -r megatron ../MindSpeed-LLM/
Megatron-LM# cd ../MindSpeed-LLM
MindSpeed-LLM# git checkout 2.2.0  # default `master`, means checkout commit from MindSpeed-LLM master

MindSpeed-LLM# pip3 install -r requirements.txt  # 安装其余依赖库
```

## transformers
Qwen3 系列模型要求 transformers 版本为4.51.0，

参考链接: https://gitcode.com/Ascend/MindSpeed-LLM/tree/master/tests/0day/qwen3

```shell
~# pip3 install transformers == 4.51.0
```

也有说，Qwen3 由于首发最新版本支持，要求transformers版本为4.51.3，

参考链接: https://modelers.cn/models/MindSpeed/Qwen3-1.7B-Base

```shell
~# pip install transformers == 4.51.3
```

# 镜像打包
可选 notebook 安装，如果需要将镜像用于在线开发，可选装 notebook 打包到镜像中，

上诉过程，可以将容器 commit 打包为 .tar 文件，用于上传至 FusionOne AI 镜像管理中，命令参考，
```shell
~# docker commit 容器id或容器名 镜像名:镜像tag

~# docker save -o xxx.tar 镜像名:镜像tag
```

也可以通过 dockerfile 的方式 build 镜像，然后打包为 .tar 文件，再上传至FusionOne AI 镜像管理中，

# 模型和数据集准备
模型通过开源方式获取，可以先上传到 FusionOne AI 文件管理，再通过 FusionOne AI 模型管理进行关联，

数据通过开源方式获取，可以先上传到 FusionOne AI 文件管理，再通过 FusionOne AI 算法开发 -> 开发环境 对数据集进行处理，参考如下，

enwiki: https://huggingface.co/datasets/lsb/enwiki20230101/blob/main/data/train-00000-of-00042-d964455e17e96d5a.parquet

alpaca: https://modelers.cn/datasets/AI_Connect/alpaca/tree/main/data

# 算法准备

## 开发环境
FusionOne AI 算法开发 -> 开发环境，调试训练算法，这里通过 mindspeed 训练模型权重需要进行格式转换，以及数据集处理，

- 基础资源：上述过程需要一定的 CPU/内存 资源，这里，设置为 `24核/48GiB`，

- 使用加速卡：以 Ascend 910B 64G 为例，这里，可设置 `2` 卡，

- 运行命令：若镜像未打包 notebook，则需运行一个后台常驻命令保活容器启动状态，否则命令执行完毕容器即停止运行，这里，可使用 `sleep 36000`，

- 文件管理挂载：需关联挂载模型目录和数据处理工作目录，以便容器使用，示例如下，

| 文件管理路径      | 容器内路径 |
| --------------- | -------- |
| 平台目录/model   | /model   |
| 平台目录/fuyuxin | /fuyuxin |

注，这里未使用平台已集成好的 “数据集”（默认：数据集的容器内路径：/dataset） 和 “模型选择”（默认：模型的容器内路径：/modeldir），

- 确认：对状态为“运行中”开发环境“打开”进入容器，这里镜像未打包notebook，进入一个终端页面（若集成了notebook则启动notebook开发环境），

## 算法处理

### 算法拷贝
在容器中，对算法进行拷贝操作，拷贝的本质是因为保存算法时，会保存 /workspace 目录下的文件，

将 `/the/path/of/MindSpeed-LLM/` 整体拷贝到 `/workspace/` 目录下，
```shell
cp -r /the/path/of/MindSpeed-LLM/ /workspace/
```

### 数据集处理
编辑相应的算法脚本，
```shell
cd /workspace/MindSpeed-LLM/
ll examples/mcore/qwen3
```

对开源数据集进行处理，这里以 enwiki 为例，主要修改点为 `--input`, `--tokenizer-name-or-path`, `--output-prefix` 参数，
```shell
cp examples/mcore/qwen3/data_convert_qwen3_pretrain.sh examples/mcore/qwen3/data_convert_qwen3_pretrain.sh.org
vim examples/mcore/qwen3/data_convert_qwen3_pretrain.sh
```

修改示例如下，
- `--input` 修改为`/fuyuxin/dataset/train-00000-of-00042-d964455e17e96d5a.parquet`
- `--tokenizer-name-or-path` 修改为`/model/Qwen3-1point7B`
- `--output-prefix` 修改为 `/fuyuxin/dataset/enwiki`

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
#mkdir ./dataset

python ./preprocess_data.py \
    --input /fuyuxin/dataset/train-00000-of-00042-d964455e17e96d5a.parquet \
    --tokenizer-name-or-path /model/Qwen3-1point7B \
    --tokenizer-type PretrainedFromHF \
    --handler-name GeneralPretrainHandler \
    --output-prefix /fuyuxin/dataset/enwiki \
    --json-keys text \
    --workers 4 \
    --log-interval 1000
```

执行命令进行数据集处理，
```shell
bash examples/mcore/qwen3/data_convert_qwen3_pretrain.sh
```

处理完之后的文件显示如下，
```shell
ll /fuyuxin/dataset
drwxr-xr-x enwiki
-rw-r--r-- enwiki_text_document.bin
-rw-r--r-- enwiki_text_document.idx
-rwxr-x--- train-00000-of-00042-d964455e17e96d5a.parquet
```

### 模型权重处理
将 hf 格式的模型权重转换为 mcore 格式，

编辑模型权重文件格式转换脚本，
```shell
cp examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh.org
vim examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh
```

修改示例如下，这里是 `convert_ckpt_v2.py` 的脚本，
```shell
export CUDA_DEVICE_MAX_CONNECTIONS=1
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt_v2.py \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 2 \
    --load-dir /model/Qwen3-1point7B \
    --save-dir /fuyuxin/ckpt/ckpt-qwen3-1point7b \
    --model-type-hf qwen3
```

注，如果是 `convert_ckpt.py` 的脚本，
```shell
export CUDA_DEVICE_MAX_CONNECTIONS=1
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --load-model-type hf \
    --save-model-type mg \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 2 \
    --load-dir /model/Qwen3-1point7B \
    --save-dir /fuyuxin/ckpt/ckpt-qwen3-1point7b \
    --tokenizer-model /model/Qwen3-1point7B/tokenizer.json \
    --model-type-hf qwen3 \
    --params-dtype bf16 \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec
```

执行命令进行模型权重处理，
```shell
bash examples/mcore/qwen3/ckpt_convert_qwen3_hf2mcore.sh
```

处理完之后的文件显示如下，
```shell
ll /fuyuxin/ckpt/ckpt-qwen3-1point7b
drwxr-xr-x iter_0000001
-rw-r----- latest_checkpointed_iteration.txt
```

## 训练调试
编辑训练算法脚本，根据实际环境对脚本进行编辑，需注意并行策略跟模型权重转换并行策略保持一致，
```shell
cp examples/mcore/qwen3/pretrain_qwen3_1point7b_4K_ptd.sh examples/mcore/qwen3/pretrain_qwen3_1point7b_4K_ptd.sh.org
vim examples/mcore/qwen3/pretrain_qwen3_1point7b_4K_ptd.sh
```

修改示例如下，

根据实际环境，修改npu卡数以及并行策略，权重及权重处理目录、数据集处理目录等，
- NPUS_PER_NODE=2
- CKPT_LOAD_DIR="/fuyuxin/ckpt/ckpt-qwen3-1point7b"
- CKPT_SAVE_DIR="/workspace/model-out"
- DATA_PATH="/fuyuxin/dataset/enwiki_text_document"
- TOKENIZER_PATH="/model/Qwen3-1point7B"
- PP=2

注意修改日志保存路径，绝对路径形式且路径存在
- tee /fuyuxin/train_mcore_qwen3_1point7b.log

```shell
#!/bin/bash

export HCCL_CONNECT_TIMEOUT=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NPU_ASD_ENABLE=0

NPUS_PER_NODE=2
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0

WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
CKPT_LOAD_DIR="/fuyuxin/ckpt/ckpt-qwen3-1point7b"
CKPT_SAVE_DIR="/workspace/model-out"
DATA_PATH="/fuyuxin/dataset/enwiki_text_document"
TOKENIZER_PATH="/model/Qwen3-1point7B"

TP=1
PP=2
MBS=1
GBS=32
SEQ_LENGTH=4096
TRAIN_ITERS=2000
ROUTER_BALANCING_TYPE='softmax_topk'

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"


OPTIMIZE_ARGS="
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --no-masked-softmax-fusion \
    --use-distributed-optimizer
"

TRAIN_ARGS="
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --lr 1.25e-6 \
    --lr-decay-style cosine \
    --min-lr 1.25e-7 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --seed 42 \
    --bf16 \
    --train-iters ${TRAIN_ITERS} \
    --seq-length ${SEQ_LENGTH} \
    --no-shared-storage
"

MODEL_PARALLEL_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
"

GPT_ARGS="
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --kv-channels 128 \
    --qk-layernorm \
    --norm-topk-prob \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --num-layers 28 \
    --hidden-size 2048 \
    --ffn-hidden-size 6144 \
    --num-attention-heads 16 \
    --tokenizer-type PretrainedFromHF \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 151936 \
    --rotary-base 1000000 \
    --disable-bias-linear \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --attention-softmax-in-fp32 \
    --no-gradient-accumulation-fusion \
    --group-query-attention \
    --num-query-groups 8
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval ${TRAIN_ITERS} \
    --eval-interval ${TRAIN_ITERS} \
    --eval-iters 0 \
    --no-load-optim \
    --no-load-rng
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $OPTIMIZE_ARGS \
    $TRAIN_ARGS \
    $MODEL_PARALLEL_ARGS \
    --distributed-backend nccl \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    | tee /fuyuxin/train_mcore_qwen3_1point7b.log
```

执行命令进行训练代码调试，
```shell
bash examples/mcore/qwen3/pretrain_qwen3_1point7b_4K_ptd.sh
```
通过终端可以看到训练步数，

确认调试没有问题，则可以保存算法，在“开发环境”中对相应的任务进行“更多 -> 保存”处理，

# 大模型预训练
在“模型训练” -> “训练任务”中创建训练任务，

需要注意，
- 文件挂载方式跟开发环境中调试的方式必须保持一致，
- 如果通过关联算法的方式运行训练任务，算法默认挂载在 `/workspace/algorithm` 目录下，

新建训练任务，
- 镜像，与开发环境调试镜像保持一致，
- 运行命令，参考如下，
```shell
cd /workspace/algorithm/MindSpeed-LLM && bash examples/mcore/qwen3/pretrain_qwen3_1point7b_4K_ptd.sh
```
- 容器数据，这里设置为 `1`，
- 基础环境，这里 CPU/内存 设置为 `24核/48GiB`，
- 使用加速卡，以 Ascend 910B 64G 为例，这里，可设置 `2` 卡，

“训练配置”页签下，
- 算法名称，与“开发环境”中训练调试的算法保持一致，
- 文件管理挂载，与“开发环境”中训练调试的文件管理挂载方式保持一致，

新建任务完成之后，点击任务名称，然后运行日志，如果有训练步数打印，则说明训练任务已经开始，

无非就是一个模型目录，一个数据处理工作目录（数据集处理、模型转换），训练代码在制作镜像时就定好版本并做进去，后续方便软件时拷贝算法目录，
开发环境中调试，/workspace
训练任务中运行，/workspace/algorithm

