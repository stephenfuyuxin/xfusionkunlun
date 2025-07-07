# 参考链接
https://support.xfusion.com/support/#/zh/docOnline/DOC2020031774?path=zh-cn_topic_0000001157169951

从部署场景上来说，分为单机和多机。单机通过端口映射实现服务化，多机通过主机网络提供服务化。

从使用场景上来说，容器启动和服务启动，都在 `docker run` 中体现，通过 `bash wings_start.sh` 作为容器/服务启动的分界线。

# docker load
aarch64，
```shell
docker load -i AISpaceWings_25.0.0_aarch64.tgz
```

# docker images
```shell
docker images | grep wings
wings-npu-aarch64:v1.0
```

# driver
```shell
/usr/local/Ascend/driver# cat version.info
Version=25.0.rc1.1
ascendhal_version=7.35.23
aicpu_version=1.0
tdt_version=1.0
log_version=1.0
prof_version=2.0
dvppkernels_version=1.1
tsfw_version=1.0
Innerversion=V100R001C21SPC002B220
compatible_version=[V100R001C17],[V100R001C18],[V100R001C19],[V100R001C20],[V100R001C21]
compatible_version_fw=[6.4.0,6.4.99],[7.0.0,7.7.99]
package_version=25.0.rc1.1
```

# firmware
```shell
/usr/local/Ascend/firmware# cat version.info
Version=7.7.0.1.231
firmware_version=1.0
package_version=25.0.rc1.1
compatible_version_drv=[23.0.rc2,23.0.rc2.],[23.0.rc3,23.0.rc3.],[23.0.0,23.0.0.],[24.0,24.0.],[24.1,24.1.],[25.0,25.0.]
```

# docker-ce
```shell
~# docker version
Client: Docker Engine - Community
 Version:           24.0.7
 API version:       1.43
 Go version:        go1.20.10
 Git commit:        afdd53b
 Built:             Thu Oct 26 09:08:14 2023
 OS/Arch:           linux/arm64
 Context:           default

Server: Docker Engine - Community
 Engine:
  Version:          24.0.7
  API version:      1.43 (minimum version 1.12)
  Go version:       go1.20.10
  Git commit:       311b9ff
  Built:            Thu Oct 26 09:08:14 2023
  OS/Arch:          linux/arm64
  Experimental:     false
 containerd:
  Version:          v1.6.28
  GitCommit:        ae07eda36dd25f8a1b98dfbf587313b99c0190bb
 ascend:
  Version:          1.1.12
  GitCommit:        v1.1.12-0-g51d5e946
 docker-init:
  Version:          0.19.0
  GitCommit:        de40ad0
```

# ascend-docker-runtime
```shell
/usr/local/Ascend/Ascend-Docker-Runtime# cat ascend_docker_runtime_install.info
version=v5.0.RC3
arch=aarch64
os=linux
path=/usr/local/Ascend/Ascend-Docker-Runtime
build=Ascend-docker-runtime_5.0.RC3-aarch64
a500=n
a500a2=n
a200=n
a200isoc=n
a200ia2=n
```

# docker run
例子，
```shell
docker run -d --shm-size=512g \
--name AIspaceWings \
--device=/dev/davinci_manager \
--device=/dev/hisi_hdc \
--device=/dev/devmm_svm \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
-v /usr/local/sbin:/usr/local/sbin:ro \
-v {host_model_path}:/weights \
-v /var/log:/var/log \
-p {server_port}:18000 \
wings-npu-aarch64:v1.0 \
bash /opt/wings/wings_start.sh \
--model-name {model_name} \
--model-path /weights
```
`--device=/dev/davinci?`，本地持久化的物理卡数决定容器用于推理的卡数；

以下根据实际环境进行修改，
- {host_model_path}，宿主机上权重路径本地持久化到容器；
- {server_port}，设置宿主机上端口映射到容器端口（默认容器端口为18000）；
- {model_name}，设置为 wings 支持的基础模型，名称须严格跟**基础模型支持**中保持一致；

实际，
```shell
~# vim run.sh
#!/bin/bash
docker run -d --shm-size=512g \
--name AIspaceWings \
--device=/dev/davinci_manager \
--device=/dev/hisi_hdc \
--device=/dev/devmm_svm \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
-v /usr/local/sbin:/usr/local/sbin:ro \
-v /home/model/DeepSeek-R1-Distill-Qwen-7B:/weights \
-v /var/log:/var/log \
-p 9988:18000 \
wings-npu-aarch64:v1.0 \
bash /opt/wings/wings_start.sh \
--model-name DeepSeek-R1-Distill-Qwen-7B \
--model-path /weights
```
启动容器，
```shell
bash run.sh
```

# 服务拉起
服务拉起后使用标准 OpenAI 接口进行服务请求，例子如下，
```shell
curl -X 'POST' \
  'http://{host_ip}:{server_port}/v1/chat/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "model":{model_name},
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "What is the largest animal?"
        }
    ]
  }'
```
- {host_ip}，启动 wings 服务化机器的主节点IP地址；
- {server_port}，启动 wings 服务化的端口；
- {model_name}，设置为 wings 支持的基础模型，名称须严格跟**基础模型支持**中保持一致；

实际，
```shell
curl -X 'POST' \
  'http://7.6.50.21:9988/v1/chat/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "DeepSeek-R1-Distill-Qwen-7B",
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "What is the largest animal?"
        }
    ]
  }'
```

# 服务返回
对服务请求的推理相应，以上诉例子推理返回结果为例，
```shell
{"id":"endpoint_common_2","object":"chat.completion","created":1751890951,"model":"DeepSeek-R1-Distill-Qwen-7B","choices":[{"index":0,"message":{"role":"assistant","tool_calls":null,"content":"Okay, so I need to figure out what the largest animal is. Hmm, I remember hearing about blue whales being really big, but I'm not entirely sure if they're the biggest. Let me think. I know that blue whales are marine animals, so they must live in the ocean. I think they're called baleen whales because they have these big plates on their sides that they use to filter out plankton. That makes sense because if they're so large, they'd need a special way to feed.\n\nWait, but aren't there other large animals on land? Like maybe elephants or something. I think elephants are pretty big, but I'm not sure if they're the largest. I've heard that blue whales are bigger than any land animal, but I'm not certain. Maybe I should check the sizes. I think blue whales can grow up to 30 meters long and weigh around 200 tons. That's huge! On land, the largest animal I know is the blue whale, but wait, maybe there are other sea creatures that are even bigger.\n\nI recall something about whales being the largest animals because they can reach such massive sizes without their bodies breaking down. They have a special bone structure, I think, that supports their size. Maybe they're the clear winners here. But I should make sure. Are there any other marine animals that could be larger? I don't think so. Seals and sharks are big, but nothing comes close to the size of a blue whale.\n\nSo, putting it all together, blue whales are the largest animals because they can grow up to 30 meters long and weigh thousands of kilograms. They're the dominant species in the ocean, and their size is supported by their unique anatomy. I think that's the answer. I don't recall any land animals that are larger than blue whales, so they must be the largest overall.\n</think>\n\nThe largest animal is the blue whale. It is a marine mammal that can grow up to 30 meters in length and weigh approximately 200 tons. Blue whales are the largest animals on Earth, and their size is supported by their unique bone structure, which allows them to support their massive size without breaking down."},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":17,"completion_tokens":454,"total_tokens":471},"prefill_time":76,"decode_time_arr":[83,52,51,52,53,50,52,53,51,52,54,54,53,52,51,54,52,51,52,54,53,52,51,54,51,51,51,53,52,51,51,54,52,51,50,53,52,51,52,53,51,50,51,53,51,50,52,52,52,51,52,53,51,51,52,53,51,51,50,53,52,51,52,53,51,51,50,53,53,51,52,53,51,52,52,54,52,50,53,54,51,52,52,53,50,50,53,54,53,52,52,54,51,50,53,53,53,51,52,54,53,53,54,52,51,51,52,53,53,53,52,54,51,52,51,54,52,51,52,53,51,51,54,54,59,52,52,53,52,51,52,54,54,51,51,55,51,51,52,53,52,52,52,52,52,52,52,53,53,51,54,60,52,51,53,54,52,51,52,53,52,53,54,55,52,53,53,53,52,54,57,54,55,53,54,53,52,53,55,55,51,52,54,53,53,53,54,53,53,51,55,54,52,53,53,54,52,53,54,53,52,53,53,51,52,54,55,54,53,53,54,54,54,53,54,54,56,53,54,54,52,52,53,54,53,52,53,52,51,53,53,51,53,52,52,52,52,52,53,54,51,50,52,55,51,53,53,53,52,53,54,55,50,51,51,54,50,52,54,52,53,51,52,55,52,53,53,52,53,52,54,52,51,52,52,53,52,51,54,52,50,52,54,52,50,53,53,51,51,53,53,51,53,52,54,53,52,52,53,52,52,53,54,53,53,51,52,53,51,52,53,52,52,53,54,52,52,53,53,53,51,52,53,51,51,53,53,52,53,51,51,53,52,51,54,55,53,53,54,52,51,54,53,53,52,53,53,52,53,53,54,52,52,52,51,51,53,53,54,52,51,52,53,51,51,52,54,57,52,53,54,53,52,53,53,52,57,53,53,53,50,52,53,59,52,53,53,52,52,53,54,56,53,53,54,54,55,56,55,55,55,54,53,51,51,53,53,52,52,52,53,53,51,52,55,53,53,54,54,53,53,54,52,52,54,55,53,52,54,54,53,52,52,54,53,52,52,53,54,53,52,54,52,53,52,52,54,52,54,55,54,53,55]}
```
