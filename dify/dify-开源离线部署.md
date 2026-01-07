# 开源离线方式
这里不考虑在线的方式，需要基于开源离线方式进行本地私有化部署，

在线方式参考链接: https://github.com/langgenius/dify

# 私有化部署方案
这里，考虑到 ollama 能力有限，仅支持 LLM 以及有限的 Embedding / Rerank 相关模型，通过 xinference 来部署和管理 Embedding 和 Rerank 模型，甚至包括多模态模型等，
- 组件：dify, ollama, xinference，
- 模型类型：LLM, Embedding, Rerank，

进行 dify 私有化部署方案，图示如下，

ollama服务（管理LLM） -- 注册 --> dify服务 <-- 注册 -- xinference（管理Embedding和Rerank）

## ollama 与 xinference 差异
| 维度       | Ollama                  | Xinference                          |
| --------- | ----------------------- | ----------------------------------- |
| 一句话定位  | 个人零配置本地 LLM 工具    | 企业级多模态模型推理平台                |
| 上手难度   | 一条命令 10 秒启动         | 需装 Python/集群，首次略复杂           |
| 支持模型   | 仅文本 LLM（需 GGUF）      | 文本、Embedding、Rerank、语音、图像 100+ |
| 部署规模   | 仅单机                    | 原生分布式，K8s 一键弹性扩缩            |
| API 兼容  | 部分 OpenAI 接口           | 100% OpenAI REST/gRPC，直接切流量    |
| 性能优化   | 单机低延迟，无连续批处理     | 分布式调度，70B 提升 30-50% 吞吐       |
| 并发能力   | 笔记本 7B 可跑，高并发掉速   | 生产压测 P99 延迟 50-100 ms          |
| 生态集成   | 需额外封装 LangChain/Dify  | LangChain、Dify、私网网关零改动        |
| 运维功能   | 无                        | 模型版本、灰度、监控、权限、A/B 测        |
| 典型场景   | 本地 Demo/教育/个人知识库    | 生产 RAG、高并发聊天、多模型混合        |

# docker 相关安装及配置
docker 安装部署、修改工作目录等以及 docker-compose 安装部署等，这里不赘述，

# dify部署及服务启动与注册

## dify 本地私有化部署
通过 docker 进行 dify 本地私有化部署，示例使用的 dify 版本为 1.11.2，
```shell
# git clone https://github.com/langgenius/dify.git
# cd dify
dify]# cd docker
dify]# cp .env.example .env     # 需要 `ll -a` 才能看到对应的文件
dify]# vim .env
dify]# docker-compose up -d
```

编辑 `.env` 的目的主要是根据实际环境修改端口映射，示例未做任何修改，
```shell
NGINX_PORT=80            # 默认80
EXPOSE_NGINX_PORT=8099   # 默认80 -> 修改为8099
```

如果需要一把将所有 `docker-compose up -d` 的容器全部删除重来，则执行以下命令，将停掉并删掉当前目录 `docker-compose` 定义的所有容器、网络、卷、镜像，
```shell
# docker-compose down --volumes --rmi all --remove-orphans
```

想保留镜像，只删容器、网络、卷，
```shell
# docker-compose down --volumes --remove-orphans
```

## dify 私有化部署状态确认
执行 `docker-compose up -d` 成功之后，打印信息，示例类似如下，
```shell
docker]# docker-compose up -d
WARN[0000] No services to build
[+] up 12/12
 ✔ Container docker-redis-1             Running   0.0s
 ✔ Container docker-weaviate-1          Running   0.0s
 ✔ Container docker-db_postgres-1       Healthy   0.5s
 ✔ Container docker-ssrf_proxy-1        Running   0.0s
 ✔ Container docker-sandbox-1           Running   0.0s
 ✔ Container docker-web-1               Running   0.0s
 ✔ Container docker-worker_beat-1       Running   0.0s
 ✔ Container docker-plugin_daemon-1     Running   0.0s
 ✔ Container docker-worker-1            Running   0.0s
 ✔ Container docker-api-1               Running   0.0s
 ✔ Container docker-nginx-1             Running   0.0s
 ✔ Container docker-init_permissions-1  Running   0.5s
```

## 进行 dify 本地访问

首先跳转，通过链接设置管理员账户，
```html
http://<IP地址>:<端口>/install
```
完成管理员账户设置，访问方式，
```html
http://<IP地址>:<端口>
```

# FAQ

## 没有科学上网工具，执行 `docker-compose up -d` 超时报错
关键信息，如，
```shell
docker]# docker compose up -d
Running 9/9
✗ nginx       Error  Get "https://registry-1.docker.io/v2/": net/http: request canceled while waiting for connection (Client.Timeout exceeded while awaiting headers)   15.0s
✗ worker      Error  context canceled                                                                                     15.0s
✗ ssrf_proxy  Error  context canceled                                                                                     15.0s
✗ db          Error  context canceled                                                                                     15.0s
✗ web         Error  context canceled                                                                                     15.0s
✗ sandbox     Error  context canceled                                                                                     15.0s
✗ api         Error  context canceled                                                                                     15.0s
✗ redis       Error  context canceled                                                                                     15.0s
✗ weaviate    Error  context canceled                                                                                     15.0s
Error response from daemon:
Get "https://registry-1.docker.io/v2/": net/http: request canceled while waiting for connection (Client.Timeout exceeded while awaiting headers)
```
编辑 `/etc/docker/daemon.json` 文件，然后在里面加入下面的配置，保存完之后启停一下 docker 服务，
```json
{
  "registry-mirrors": [
    "https://docker.registry.cyou",
    "https://docker-cf.registry.cyou",
    "https://dockercf.jsdelivr.fyi",
    "https://docker.jsdelivr.fyi",
    "https://dockertest.jsdelivr.fyi",
    "https://mirror.aliyuncs.com",
    "https://dockerproxy.com",
    "https://mirror.baidubce.com",
    "https://docker.m.daocloud.io",
    "https://docker.nju.edu.cn",
    "https://docker.mirrors.sjtug.sjtu.edu.cn",
    "https://docker.mirrors.ustc.edu.cn",
    "https://mirror.iscas.ac.cn",
    "https://docker.rainbond.cc",
    "https://docker.unsee.tech",
    "https://dockerpull.org",
    "https://docker.1panel.live",
    "https://dockerhub.icu",
    "https://docker.m.daocloud.io",
    "https://docker.nju.edu.cn",
    "https://registry.docker-cn.com",
    "https://docker.mirrors.ustc.edu.cn",
    "https://hub-mirror.c.163.com",
    "https://mirror.baidubce.com",
    "https://5tqw56kt.mirror.aliyuncs.com",
    "https://docker.hpcloud.cloud",
    "http://mirrors.ustc.edu.cn",
    "https://docker.chenby.cn",
    "https://docker.ckyl.me",
    "http://mirror.azure.cn",
    "https://hub.rat.dev"
  ]
}
```

## 容器 `docker-init_permissions-1` 刚启动就 `Exited` 状态退出，
实测，aarch64架构存在这个问题，x86_64架构不存在这个问题（dify 1.11.0, 1.11.2 验证结论），
```shell
docker]# docker-compose up -d
WARN[0000] No services to build
[+] up 11/11 
 ✔ Container docker-web-1                Running   0.0s
 ✔ Container docker-redis-1              Running   0.0s
 ✔ Container docker-db_postgres-1        Healthy   0.5s
 ✔ Container docker-weaviate-1           Running   0.0s
 ✔ Container docker-sandbox-1            Running   0.0s
 ✔ Container docker-worker-1             Running   0.0s
 ✔ Container docker-plugin_daemon-1      Running   0.0s
 ✔ Container docker-worker_beat-1        Running   0.0s
 ✔ Container docker-api-1                Running   0.0s
 ✔ Container docker-nginx-1              Running   0.0s
 ✔ Container docker-init_permissions-1   Exited    0.5s
```

执行 `docker logs` 对特定的容器，查看报错日志信息，
```shell
# docker logs docker-init_permissions-1
Permissions already initialized. Exiting.
```
打印 `Permissions already initialized. Exiting.` 说明“权限初始化”容器本身并没有报错，它只是检测到上一次已经把目录 chown 好了，于是正常退出，

这是 Dify 官方 compose 文件里特意写的“一次性任务”容器，exit 0 是预期行为，不是故障，真正导致 docker compose up -d 失败的通常是它后面那个服务，

通过 `docker-compose ps -a` 找到状态一栏是 “Restarting” 或 “Exit (1)” 的那个容器名字
```shell
# docker-compose ps -a
NAME                        IMAGE                COMMAND                   SERVICE            CREATED        STATUS                          PORTS
docker-init_permissions-1   busybox:latest       "sh -c 'FLAG_FILE=\"/…"   init_permissions   23 hours ago   Exited (0) 23 hours ago
docker-ssrf_proxy-1         ubuntu/squid:latest  "sh -c 'cp /docker-e…"    ssrf_proxy         23 hours ago   Restarting (1) 28 seconds ago
```
发现，STATUS 栏里只有 docker-ssrf_proxy-1 在反复 Restarting (1)，其余全部是 Up 状态，

说明整栈其实早就起来了，只是 ssrf_proxy 这个容器一直退出，导致 compose 认为“服务未完全就绪”，

继续打印 ssrf_proxy 容器日志，这里，仅贴出关键报错信息，
```shell
# docker logs docker-ssrf_proxy-1
(logfile-daemon): error while loading shared libraries: libstdc++.so.6: cannot open shared object file: No such file or directory
```
squid 的日志守护进程因为找不到 libstdc++.so.6，pipe 一断，主进程跟着 FATAL 退出，于是容器无限重启，

这是 ubuntu/squid:6.x 镜像在 arm64(aarch64) 环境里的已知打包缺陷：官方把 squid 编译成需要 libstdc++.so.6，但基础镜像里没装 libstdc++6 包，

也有可能是，ubuntu/squid 是用 Snap 或容器内隔离方式运行的，实际运行时找不到那个库，通过在 docker-compose.yaml 文件中 apt 安装 squid libstdc++6 仍然失败，

彻底解决思路：彻底放弃 ubuntu/squid:latest，用原生“ubuntu + squid”方式，修改 docker-compose.yaml 文件，
```shell
docker]# cp docker-compose.yaml docker-compose.yaml.org
docker]# vim docker-compose.yaml
```
将 `image: ubuntu/squid:latest` 改成 `image: ubuntu:22.04` 之后，加上 `command` 构建，重新拉起试一把，
```yaml
  ssrf_proxy:
    image: ubuntu:22.04
    restart: always
    command: |
      bash -c "
        apt-get update &&
        apt-get install -y squid libstdc++6 &&
        # 用 awk 渲染模板到真实配置
        awk '{
            while(match($0, /\\$${[A-Za-z_][A-Za-z_0-9]*}/)) {
                var = substr($0, RSTART+2, RLENGTH-3)
                val = ENVIRON[var]
                $0 = substr($0, 1, RSTART-1) val substr($0, RSTART+RLENGTH)
            }
            print
        }' /etc/squid/squid.conf.template > /etc/squid/squid.conf &&
        # 启动 Squid
        /usr/sbin/squid -NYC -f /etc/squid/squid.conf
      "
    ports:
      - "3128:3128"
      - "8194:8194"
    volumes:
      - ./ssrf_proxy/squid.conf.template:/etc/squid/squid.conf.template:ro
    environment:
      HTTP_PORT: ${SSRF_HTTP_PORT:-3128}
      COREDUMP_DIR: ${SSRF_COREDUMP_DIR:-/var/spool/squid}
      REVERSE_PROXY_PORT: ${SSRF_REVERSE_PROXY_PORT:-8194}
      SANDBOX_HOST: ${SSRF_SANDBOX_HOST:-sandbox}
      SANDBOX_PORT: ${SANDBOX_PORT:-8194}
    networks:
      - ssrf_proxy_network
      - default
```
重新启动，
```shell
# docker-compose up -d --force-recreate ssrf_proxy
```
检查状态，`docker-compose logs -f ssrf_proxy` 不会再出现 libstdc++6 缺失的错误，且 ssrf_proxy 容器状态正常，
```shell
# docker-compose logs -f ssrf_proxy
# docker-compose ps
NAME                IMAGE        COMMAND                  SERVICE    CREATED            STATUS            PORTS
docker-ssrf_proxy-1 ubuntu:22.04 "bash -c '\n  apt-get…"  ssrf_proxy About a minute ago Up About a minute 0.0.0.0:3128->3128/tcp, [::]:3128->3128/tcp, 0.0.0.0:8194->8194/tcp, [::]:8194->8194/tcp
```
不过放置一段时间之后，仍然会出现 `docker-compose ps` 有部分容器的健康状态检查从 `healthy` 变为 `unhealthy`，后面更换 x86_64 架构环境进行安装部署，
