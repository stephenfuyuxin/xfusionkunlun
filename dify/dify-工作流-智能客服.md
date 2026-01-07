# dify-工作流-智能客服

将商城中用户对商品、服务的评价进行统计分析，推动营销策略的制定与改善，

原有的流程，运营团队需要通过登录商城后台管理系统查看用户对商品的评价，以人工的方式对用户评价进行统计分析，

当前，基于AI的应用，可使用 LLM 对客户评价进行分析、分类，区别出正面/负面评价，

- 正面评价，发送至公司营销群，

- 负面评价，通过进一步分类，区分出质量问题、物流问题或者其他问题，并将负面评价发送至公司营销群，

基于 AI 应用，相比原有的人工流程，实现效率提升，

在飞书上通过个人 bot 或者群组 bot 来接收 “工作流 - 智能客服” 的反馈消息，

以下为往飞书 bot 返回消息的 Python 代码参考，根据实测环境，修改飞书的 webhook 链接地址，
```python
import requests
import json
 
# 飞书 webhook 地址
WEBHOOK_URL = '输入实际使用的 Feishu webhook URL链接 https/http 形式'
 
# 要发送的消息内容
data = {
    "msg_type": "text",
    "content": {
        "text": "你好，这是一个测试消息。"
    }
}
 
headers = {
    'Content-Type': 'application/json',
    'Charset': 'UTF-8'
}
 
response = requests.post(WEBHOOK_URL, headers=headers, data=json.dumps(data))
 
if response.status_code == 200:
    print('消息发送成功。')
else:
    print('消息发送失败。')
```

## 开始节点
在开始节点，初始化定义信息，需要输入**商品编号**、**商品评价**信息，每个工作流都需要有一个开始节点，用于接收初始化输入信息，
```shell
商品编号
（待输入的商品编号，如 123, 123456等）
商品评价
（待输入的商品评价，如 物美价廉 等商品相关的描述性语言）
```

创建空白应用 -> 选择应用类型**工作流** -> 应用名称&图标，输入应用名称**fuyuxin-智能客服**，图片可自定义上传 -> 点击创建，

跳出**选择开始节点来开始** -> 通过**用户输入（原始开始节点）**，来创建开始节点，

在**用户输入**中，添加**输入字段**，这里，我们输入的字**字段类型**都是**文本**，依次创建标题和章节，
- 变量名称为`product_id`，显示名称为`商品编号`，
- 变量名称为`product_review`，显示名称为`商品评价`，
创建好之后，界面显示为，
```shell
用户输入
{x} product_id      商品编号  必填
{x} product_review  商品评价  必填
```

## 问题分类器
问题分类器的目的，根据⽤户的输⼊推理出与预设结果相匹配的分类结果，问题分类器核心，其实就是依赖 LLM ⼤语⾔模型的能⼒识别⽤户输⼊，

问题分类器通过将开始节点**product_review**字段的内容投⼊ LLM 中处理，并将处理结果与预设结果相匹配进行归类，这里，对商品评价进行归类及细分，可通过多级问题分类器对商品描述进行预置的类别筛选以及多次细分，按照预置的正面/负面评价进行多级归类，

- 正面评价，如果是正⾯评价则下⼀步流程交给正⾯评价分类器处理，反之则交给负⾯评价处理器处理，正⾯评价可以通过进⼀步问题分类器细分为质量好、性价⽐⾼等。示例对正向分类器不做二次细分，正向评价直接输出，

- 负面评价，如果是负面评价则进一步交给负⾯评价分类期处理，进⼀步细分为质量差、物流慢和其他等。示例重点实测这部分，

这里，用两个问题分类器级联，将 评价 输入 LLM 中处理，第一个问题分类仅做正向/负向分类，第二个问题分类对负向进行二次细分

```shell
模型
（关联 LLM 模型对评价进行处理）
输入
（关联 用户输入/{x} product_review String，输入评价文本信息）
分类
（根据预置的类别进行分类，如正向/负向评价，如对负向评价进行二次细分为质量差、物流慢、其他等）
```

## 代码执行
问题分类器经过 LLM 对评价进行处理，按照既定类别完成细分之后，需要将问题分类器的结果发送到 bot 中，即问题分类器的输出经过代码执行模块将特定的信息进行发送处理，

以下为不同类别、细分评价的自定义函数代码示例，

- 正面评价，正向评价不做二次细分直接输出，自定义函数代码示例，

代码执行-正面评价，
```python
import json

def main(product_id, product_review):
    message = "恭喜，你的产品收到了来自客户的正向反馈，用户对产品 {} 的评价是 {}".format(product_id, product_review)
    data = {
        "msg_type": "text",
        "content": {"text": message}
    }
    return {
        "result": json.dumps(data, ensure_ascii=False)
    }
```

- 负面评价，负向评价二次细分，根据不同细分结果（质量差、物流慢、其他）进行代码执行输出，自定义函数代码示例，

代码执行-负面评价-质量差，
```python
import json

def main(product_id, product_review):
    message = "请注意，你的产品收到了来自客户的负面评价-质量差，用户对产品 {} 的评价是 {}".format(product_id, product_review)
    data = {
        "msg_type": "text",
        "content": {"text": message}
    }
    return {
        "result": json.dumps(data, ensure_ascii=False)
    }
```

代码执行-负面评价-物流慢，
```python
import json

def main(product_id, product_review):
    message = "请注意，你的产品收到了来自客户的负面评价-物流慢，用户对产品 {} 的评价是 {}".format(product_id, product_review)
    data = {
        "msg_type": "text",
        "content": {"text": message}
    }
    return {
        "result": json.dumps(data, ensure_ascii=False)
    }
```

代码执行-负面评价-其他，
```python
import json

def main(product_id, product_review):
    message = "请注意，你的产品收到了来自客户的负面反馈-其他，用户对产品 {} 的评价是 {}".format(product_id, product_review)
    data = {
        "msg_type": "text",
        "content": {"text": message}
    }
    return {
        "result": json.dumps(data, ensure_ascii=False)
    }
```

## 变量聚合器
这里，不同的类别和细分的输出结果，返回值数据类型 `result` 是一样的，可以将不同分支的输出结果进行统一，

将多路分⽀的变量聚合为⼀个变量，以实现下游节点统⼀配置，在本例中将多个代码执⾏节点的输出结果 `result` 进⾏聚合，

示例如下，将**正面评价**以及**负面评价对应的二次细分**结果进行聚合，多个代码分支执行的结果可以进行统一输出，这里，聚合总计4个代码执行输出 `result` 结果，
```shell
变量赋值
正面评价 / {x} result string
负面评价-质量差 / {x} result String
负面评价-物流慢 / {x} result String
负面评价-其他 / {x} result String
```

## HTTP 请求节点
将变量聚合输出的结果通过 `http` 的方式发送给 bot，这里需要对 `result` 结果进行 `http` 形式处理并发送，

本节点的作⽤就是向外部发送 `http` 请求，给 bot 发消息，创建 HTTP 请求节点，需要设置的参数示例如下，

- **API**，调用 `POST` 方法，输入 bot 的 `http` URL链接，用于接收 `http` 格式的消息，

- **HEADERS**，输入 `http` 所需头部格式，如表所示，

| Key          | Value            |
| ------------ | ---------------- |
| Content-Type | application/json |
| Charset      | UTF-8            |

- **BODY**，选择 `raw`，输出代码中 `message` 变量所标识的文本格式 `RAW TEXT`，用于打印到 bot 中，这里，

通过 `按 '/' 键快速插入`，选择变量聚合器的输出作为最终输出 `{x}变量聚合器/{x} output`，

## 结束节点
定义⼀个⼯作流程结束的最终输出内容，每⼀个⼯作流在完整执⾏后都需要⾄少⼀个结束节点，⽤于输出完整执⾏的最终结果，

结束节点为流程终⽌节点，后⾯⽆法再添加其他节点，⼯作流应⽤中只有运⾏到结束节点才会输出执⾏结果，若流程中出现条件分叉，则需要定义多个结束节点，

设置输出变量，示例如下，可打印 HTTP 请求状态码、 body 输出状态以及自定义一个 `raw_text` 变量用于打印发给 bot 机器人的 `msg_type` 以及文本消息 ，
```shell
输出变量
status_code   HTTP 请求/{x} status_code Number
body          HTTP 请求/{x} body String
raw_text      {x}变量聚合器/{x} output
```
一条正面评价实测结果，在输出节点的结果展示，分别对应上述3条输出变量的结果
```shell
200
{"StatusCode":0,"StatusMessage":"success","code":0,"data":{},"msg":"success"}
{"msg_type": "text", "content": {"text": "恭喜，你的产品收到了来自客户的正向反馈，用户对产品 333 的评价是 物美价廉，购物体验极好"}}
```

# 智能客服实测留档
以下是我的飞书 bot 群机器人 -> 商品评价机器人，实测4条商品评价的效果以及文本记录（分别对应负面评价的其他、物流慢以及质量差3条，以及正面评价1条）
```shell
商品评价机器人
机器人
测试 - 通过webhook将自定义服务-商品评价测试的POST消息推送至飞书

请注意，你的产品收到了来自客户的负面反馈-其他，用户对产品 123 的评价是 客户小姐姐态度不好，购物体验差

请注意，你的产品收到了来自客户的负面评价-物流慢，用户对产品 111 的评价是 太慢了，10天之后才收到货物！

请注意，你的产品收到了来自客户的负面评价-质量差，用户对产品 222 的评价是 质感较差，给人一种很廉价的体验

恭喜，你的产品收到了来自客户的正向反馈，用户对产品 333 的评价是 物美价廉，购物体验极好
```
