# 配置前准备
默认已经安装了所需驱动，配置Embedding模型和Rerank模型，准备以下安装包，

## bge-m3 模型
最终形式：bge-m3.zip

获取链接：https://modelscope.cn/models/BAAI/bge-m3/summary

```shell
git clone https://www.modelscope.cn/BAAI/bge-m3.git
```

## bge-reranker-v2-m3 模型
最终形式：bge-reranker-v2-m3.zip

获取链接：https://modelscope.cn/models/BAAI/bge-reranker-v2-m3/summary

```shell
git clone https://www.modelscope.cn/BAAI/bge-reranker-v2-m3.git
```

## TEI镜像
最终形式：text-embeddings-inference-*.tar

获取链接，这里以 Ampere 86 (A10，**A40**，...) 为例：ghcr.io/huggingface/text-embeddings-inference:86-1.7

镜像下载，
```shell
docker pull ghcr.io/huggingface/text-embeddings-inference:86-1.7
86-1.7: Pulling from huggingface/text-embeddings-inference
aece8493d397: Pull complete
9fe5ccccae45: Pull complete
8054e9d6e8d6: Downloading [===========================================>       ]  48.76MB/56.08MB
bdddd5cb92f6: Download complete
5324914b4472: Download complete
8054e9d6e8d6: Pull complete
bdddd5cb92f6: Pull complete
5324914b4472: Pull complete
b2f013a2512c: Pull complete
4df609bd0d92: Pull complete
Digest: sha256:57f80068bbdd3cf8c8899e8347b91ff30d45c3b608cefab5974bb9ac90ffe3b9
Status: Downloaded newer image for ghcr.io/huggingface/text-embeddings-inference:86-1.7
ghcr.io/huggingface/text-embeddings-inference:86-1.7
```

镜像打包，
```shell
docker save -o text-embeddings-inference-a40.tar ghcr.io/huggingface/text-embeddings-inference:86-1.7

ll -h
-rw-------  1 root root 1.1G Dec  2 07:05 text-embeddings-inference-a40.tar
```

# 上传镜像
首页“模型”页签 -> 镜像管理 -> “创建” -> 自定义输入“镜像名称”, “镜像用途”, “镜像版本号” -> 通过“导入方式”指定“文件” 完成镜像文件上传，
- 镜像名称：text-embeddings-inference-a40
- 版本号：v1
- 镜像用途：推理服务
- 导入方式：文件管理
- 文件：我的目录/text-embeddings-inference-a40.tar
- 点击“确定”，等待镜像上传完成，显示“状态”为“上传完成”

# 部署 embedding 模型
首页“模型”页签 -> 模型管理 -> 模型列表 -> 我的模型，创建模型，

【注】这里直接将模型创建到模型广场中，但这个并不影响在线服务中
- 模型名称：bge-m3
- 基础模型：bge-m3
- 模型类型：重排序
- 模型描述：为空
- 模型上传：使能
- 版本号：V0001
- 版本描述：为空
- 上传方式：文件管理导入
- 文件/文件夹：我的目录/bge-m3（这里规格实现有差异，.zip包解压缩之后进行关联吧）
- 点击“版本”，查看“状态”是否“成功”

首页“模型”页签 -> 推理服务 -> 在线服务，创建在线服务，
- 服务名称：fuyuxin-bge-m3
- 模型选择：bge-m3/V0001
- 服务端口号：8888
- API：/v1/embeddings
- API Key鉴权：默认，去使能

下一步，

- 服务场景：标准推理
- 节点：cp0
- 推理引擎：我的镜像 -> text-embeddings-inference-a40 / v1
- 实例数量：1
- 加速卡：英伟达 / A40  ----> 设置资源，加速卡数量：1，CPU核：10，内存：10，共享内存：9，节点数：1（内存不能小于共享内存）
- 使用模式：整卡
- 调度方式：性能

高级配置，默认挂进去的模型文件启动后的路径为 `/usr/local/serving/models/` 因此这一步也可以不做走系统默认，
- 文件管理挂载：文件管理路径->我的目录/bge-m3，容器内路径->/data/bge-m3

下一步，

- 运行命令：如下，
原先的命令为，
```shell
text-embeddings-router --model-id /data/bge-m3
```
根据实际环境，更新的命令为，
```shell
text-embeddings-router --model-id /usr/local/serving/models/ --port 8888
```
点击“下一步”，点击“确认”提交，点击运行中的服务，当状态变为“运行中”，进入预测界面，选择“json”，并输入预测内容，点击“预测”查看预测结果，
```json
{
    "input": "I like you."
}
```
预测结果，示例如下，
```json
{
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "embedding": [
                -0.025751725,
                0.023605749,
                -0.039064057,
                0.025806284,
                0.004937566,
                -0.013430543,
... ...
```
系统后台执行这个，
```shell
???
```
资源监控里面是这个，bge-m3-V0001: serving-rn-uvke27-f5hhzx6i-c898dfb8d-8nsd8（pod名称）

# 部署 rerank 模型
首页“模型”页签 -> 模型管理 -> 模型列表 -> 我的模型，创建模型，

【注】这里直接将模型创建到模型广场中，但这个并不影响在线服务中
- 模型名称：bge-reranker-v2-m3
- 基础模型：bge-reranker-v2-m3
- 模型类型：重排序
- 模型描述：为空
- 模型上传：使能
- 版本号：V0001
- 版本描述：为空
- 上传方式：文件管理导入
- 文件/文件夹：我的目录/bge-reranker-v2-m3（这里规格实现有差异，.zip包解压缩之后进行关联吧）
- 点击“版本”，查看“状态”是否“成功”

首页“模型”页签 -> 推理服务 -> 在线服务，创建在线服务，
- 服务名称：fuyuxin-bge-reranker-v2-m3
- 模型选择：bge-reranker-v2-m3/V0001
- 服务端口号：9999
- API：/rerank  ---> 如果选定模型之后，默认为：/v1/rerank，这里需手动改一下
- API Key鉴权：默认，去使能

下一步，

- 服务场景：标准推理
- 节点：cp0
- 推理引擎：我的镜像 -> text-embeddings-inference-a40 / v1
- 实例数量：1
- 加速卡：英伟达 / A40  ----> 设置资源，加速卡数量：1，CPU核：10，内存：10，共享内存：9，节点数：1（内存不能小于共享内存）
- 使用模式：整卡
- 调度方式：性能

高级配置，默认挂进去的模型文件启动后的路径为 `/usr/local/serving/models/` 因此这一步也可以不做走系统默认，
- 文件管理挂载：文件管理路径->我的目录/bge-reranker-v2-m3，容器内路径->/data/bge-reranker-v2-m3

下一步，

- 运行命令：如下，
原先的命令为，
```shell
text-embeddings-router --model-id /data/bge-reranker-v2-m3
```
根据实际环境，更新的命令为，
```shell
text-embeddings-router --model-id /usr/local/serving/models/ --port 9999
```
点击“下一步”，点击“确认”提交，点击运行中的服务，当状态变为“运行中”，进入预测界面，选择“json”，并输入预测内容，点击“预测”查看预测结果，
```json
{"query":"who are you?","texts":["tom","bob","alice"]}
```
预测结果，示例如下，
```json
[
    {
        "index": 2,
        "score": 0.0014607497
    },
    {
        "index": 0,
        "score": 0.0007321813
    },
... ...
```
系统后台执行这个，
```shell
???
```
资源监控里面是这个，bge-reranker-v2-m3-V0001: serving-rn-joze31-ftq8bla1-855bd9d6f7-jn2hn（pod名称）


# 对接应用使能平台
首页“应用”页签 -> 左侧导航“模型仓”，

## embedding
单击 “Open-API-Compatible” 下的 “添加模型”，输入 embedding 模型参数，
- 模型类型：embedding
- 模型名称：bge-m3
- 基础url：https://7.6.51.23/serving-gateway/6d57c192b32e4173960820b43e5f49e8/v1/

按照上面 embeddiubng 模型中“在线服务” -> “预测” 中的 POST URL，示例，
```shell
https://7.6.51.23/serving-gateway/6d57c192b32e4173960820b43e5f49e8/v1/embeddings
```
- 最大token数：512

## rerank
单击 “HuggingFace” 下的 “添加模型”，输入 rerank 模型参数，
- 模型类型：rerank
- 模型名称：bge-reranker-v2-m3
- 基础url：https://7.6.51.23/serving-gateway/589a90b7b04146eebe52cfbe4720fee2/

按照上面 rerank 模型中“在线服务” -> “预测” 中的 POST URL，示例，
```shell
https://7.6.51.23/serving-gateway/589a90b7b04146eebe52cfbe4720fee2/rerank
```
- 最大token数：1024

## 模型仓 -> 系统模型设置
这里，对模型仓中的模型指定默认，将对应的模型设置为添加的模型，比如
- 嵌入模型，设置为bge-m3；
- rerank模型，设置为bge-reranker-v2-m3；



