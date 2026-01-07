# dify-工作流-应用构建
⼯作流通过将复杂的任务分解成较⼩的节点以降低系统复杂度，减少了对提示词技术和模型推理能⼒的依赖，提⾼了 LLM 应⽤⾯向复杂任务的性能，提升了系统的可解释性、稳定性和容错性，

⼯作流提供了丰富的逻辑节点，⽐如代码节点、流程控制节点、循环控制节点等，通过这些节点可解决⾃动化、批处理场景中的相对复杂的任务逻辑，

在 Dify 中，⼯作流分为以下两种类型， 
- Workflow：适⽤于⾼质量翻译、数据分析、内容⽣成等⾯向⾃动化和批处理场景；
- Chatflow：适⽤于客户服务、语义搜索等⾯向对话类的场景；

# 工作流 - 智能写作
输⼊⽂章标题以及相应的章节概述，从而生成长文本完整文章的应用，

用工作流设计一个智能写作，用户输入**文章标题**和**一级章节**，让长篇文章生成，通过工作流生成更多的子章节并最终输出长文本，

这里示例，

LLM 模型通过 `OpenAI-API-compatible` 的 MiniMax-M2 `/v1/chat/completions` 的api调用接口，

不考虑输出内容质量的前提下，相比 `OpenAI-API-compatible` 的 Qwen3-Next-80B-A3B-Instruct-AWQ-4bit `/v1/chat/completions`，这个推理输出的逻辑和层次更好，

## 开始节点
在开始节点，用户需要输入**文章标题**和**一级章节**概述，
```shell
# 文章标题
庄子的人生感悟
# 章节详情
1. 得失的故事
2. 困境的故事
3. 选择的故事
4. 评价的故事
5. ⼼态的故事
```
创建空白应用 -> 选择应用类型**工作流** -> 应用名称&图标，输入应用名称**fuyuxin-智能写作**，图片可自定义上传 -> 点击创建，

跳出**选择开始节点来开始** -> 通过**用户输入（原始开始节点）**，来创建开始节点，

在**用户输入**中，添加**输入字段**，这里，我们输入的字**字段类型**都是**文本**，依次创建标题和章节，
- 变量名称为`title`，显示名称为`文章标题`，
- 变量名称为`chapter`，显示名称为`一级章节`，
创建好之后，界面显示为，
```shell
用户输入
{x} title    必填
{x} chapter  必填
```

## LLM 节点
在本节点，接收开始节点的传递过来的参数 `title` 和 `chapter` ，并通过提示词工程 `prompt` 定义模型⾏为，在提示词中定义 **⻆⾊, 技能, ⽬标, 限制, 输出示例** 等 ，让 LLM 严格按照意图来输出⽂本，

在 LLM 节点中，**SYSTEM** 提示词示例如下，使用过程中需要注意，
- **文章标题**和**一级章节**通过按 `/` 键快速插入； 
- 示例中 **```** 前面的 `/` 转义在使用时需要删除；
- 输出示例注意保持 `json` 字符串格式，不要换行；

```shell
## ⻆⾊：⽂章撰写专家 
## 技能：你根据⽤户输⼊的⽂章标题 /{x}title 和各章节名称 /{x}chapter ，⽣成各个章节及⼦章节 
## ⽬标： 
- 确保⽣成的每个⼦章节和⽗章节紧密相关 
- 纵观整体章节，必须保证各章节过渡连贯流畅 
- 最终输出json字符串，详细请看以下输出示例 
## 限制： 
- 输出内容必须是标准json字符串，不要包含任何与json字符串⽆关的内容 
- 请严格按照输出示例中的数据格式输出json字符串，不要输出其他任何与json字符串 
⽆关⽂本、以及特殊字符 
- 不要输出任何与json⽆关的特殊符号，⽐如\n或者是#或者是\```
- 请将位于输出内容开头或结尾的任何与json⽆关的特殊符号都删掉 
## 输出示例： 
[{"chapter": "引⾔", "subchapter": ["1. ⽓候变化对沿海城市影响的概述",  "2. 理解这些影响的重要性"]}, {"chapter": "海平⾯上升", "subchapter":  ["1. 海平⾯上升的原因", "2. 对沿海基础设施和社区的影响" ,"3. 受影响城市的例⼦"}]
```

## 代码执行模块
本节点⽀持运⾏ Python / NodeJS 代码以在⼯作流程中执⾏数据转换，⾮常适合⽤于 json 转换、 ⽂本处理等情景，

该节点极⼤地增强开发灵活性，能够在⼯作流程中嵌⼊⾃定义的 Python 或 Javascript 脚本，实现预设节点⽆法完成的⼯作任务，本节点的输出类型包括 string 、 Array[Object] 等，需选择适合⾃⼰的数据类型；

示例中，
- 输入变量设置为 `arg1`，设置类型为 `LLM / {x} text string`，
- 输出变量设置为 `result`，设置类型为 `Array[Object]`，

代码执行模块的核心在于，将 LLM 节点输出从 `json` 的 `string` 类型转换为 `` 类型，
```python
import json 
def main(arg1): 
    data = json.loads(arg1.strip()) 
    return { 
        "result": data 
    }
```

对于 json string 最外层带 `<think>…</think>` 标记的部分，导致 `json.loads` 从第 1 个字符开始就遇到了非 json 内容，于是抛出以下异常（详情见 FAQ 章节对应问题），
```shell
“Expecting value: line 1 column 1 (char 0)”
```

代码修改如下，实测可用，
```python
import json
import re

def main(arg1: str):
    # 去掉 <think>...</think> 段落（非贪婪匹配）
    json_part = re.sub(r'<think>.*?</think>', '', arg1, flags=re.S).strip()
    
    # 如果去掉后为空，直接抛异常
    if not json_part:
        raise ValueError("No JSON content found after removing <think> block")
    
    data = json.loads(json_part)
    return {"result": data}
```

输出，示例如下，
```shell
{
  "result": [
    {
      "chapter": "得失的故事",
      "subchapter": [
        "1. 什么是得失",
        "2. 庄子论得失的智慧",
        "3. 生活中的得失实例"
      ]
    },
    {
      "chapter": "困境的故事",
      "subchapter": [
        "1. 困境的定义",
        "2. 庄子如何面对困境",
        "3. 困境中的选择与智慧"
      ]
    },
    {
      "chapter": "选择的故事",
      "subchapter": [
        "1. 选择的本质",
        "2. 庄子关于选择的寓言",
        "3. 如何做出明智选择"
      ]
    },
    {
      "chapter": "评价的故事",
      "subchapter": [
        "1. 评价的标准",
        "2. 庄子对评价的思考",
        "3. 自我评价与他人评价"
      ]
    },
    {
      "chapter": "心态的故事",
      "subchapter": [
        "1. 心态的重要性",
        "2. 庄子保持平和心态的方法",
        "3. 心态调整的实践"
      ]
    }
  ]
}
```

## 迭代节点
上一节点代码执⾏的输出数据类型为 `Array[object]` 作为本节点迭代节点的输⼊，

迭代节点的核心在于，通过循环 `Array[Object]` ，取出其中每⼀个 `Object` ，⼀个 `Object` 代表了⼀个章节（包含⽗章节和各个⼦章节），总共5个章节，所以迭代节点循环迭代5次，

根据每⼀个 `Object` ⽣成章节详细内容，如何⽣成？在迭代节点中嵌套 LLM 节点，使⽤ LLM 节点编写每⼀个章节的具体内容，

示例中，
- 新增后续迭代节点，代码生成节点，点击新建 -> 逻辑 -> 迭代，新建迭代节点，
- 在迭代节点中，新增嵌套 LLM 节点，设置方式与上述 LLM 节点类似，不过上述 LLM 节点用于根据提示词意图生成标题、章节和子章节等，这里根据提示词进一步生成各章节文本内容，
- 生成文本内容，在 **SYSTEM** 和 **USER** 中都分别编写 `prompt` 根据提示词意图生成文章内容，
```shell
# SYSTEM提示词 
你是⼀位⽂章撰写专家，擅⻓写有吸引⼒的⻓篇⽂章

# USER提示词 
你正在写⼀篇名为 /{x}title 的⽂章，请根据以下信息 /{x}item 写每⼀个章节，⽣成全⽂时，请以完整的⼤纲作为参考 /{x}chapter
```
- 迭代节点设置输入输出数据类型，输入为代码执行的 `{x} result Array[Object]`，输出为迭代内层嵌套 LLM 的 `{x} text String`，

每⼀个迭代内层嵌套 LLM 节点输出的都是 `String` 类型⽂本，所以最终迭代节点输出的就是 `Array[String]` 数组，这是⼀个包含每个章节详细内容的数组

## 代码执行模块
在本节点中，将迭代节点产⽣的 `Array[String]` 数组中的 `String` 数据进⾏拼接，形成完整的文章，

示例中，
- 新增代码执行模块，用于将迭代节点产⽣的 `Array[String]` 数组中的 `String` 数据进⾏拼接，点击新建 -> 转换 -> 代码执行，新建代码执行，
- 在代码执行中，设置输入变量为 `arg1`，设置变量值为迭代的输出 `{x} output Array[String]`，
- 通过 Python 代码执行 `Array[String]` 数组中的 `String` 数据拼接操作，
```python
def main(arg1): 
    return { 
        "result": "\n\n\n---华丽分割线----\n\n\n".join(arg1) 
    }
```

这里，同样碰到了迭代节点中内层嵌套 LLM 输出结果带 `<think>…</think>` 标记的问题，虽不报错，但这属于无需给用户的正文，如果直接拼接，就会把大量内部推理文本也塞进结果里，既占 token 又干扰阅读，

在 `join` 之前先做一次过滤即可，修改执行代码如下，实测可用，
```python
import re
def main(arg1: list[str]) -> dict:
    cleaned = []
    for text in arg1:
        # 去掉 <think>...</think> 整块
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
        if text:                       # 防止出现空串
            cleaned.append(text)
    return {"result": "\n\n\n---华丽分割线----\n\n\n".join(cleaned)}
```

- 设置输出变量为 `result`，数据类型为 `String` 即可，

## 结束节点
⽤于定义⼀个 workflow 流程结束的最终输出内容，在本例中将输出上一节点代码执行 `String` 数据拼接的完整⽂本内容，

示例中，
- 新建输出节点，作为结束节点，
- 新增输出变量 `result`，设置变量值为拼接代码执行的输出 `{x} result String`

通过全流程**测试运行**，无报错，结束节点有对应的输出，通过**发布**完成发布上线或发布更新，状态变成**已发布**之后，点击**运行**上线，

注，点击运行上线之后，链接有问题，需要在<IP地址>后面加上 **:<端口>** 才可用（不用管弹窗要求输入用户名/密码的操作），

# FAQ

## 从 LLM 节点输出 json string 最外层带 `<think>…</think>` 标记，导致代码执行模块报错
报错如下，
```shell
raceback (most recent call last): File "/var/sandbox/sandbox-python/tmp/b88211c3_66af_42c0_913e_31678e9a54ab.py", line 48, in <module> File "<string>", line 15, in <module> File "<string>", line 3, in main File "/usr/local/lib/python3.10/json/__init__.py", line 346, in loads return _default_decoder.decode(s) File "/usr/local/lib/python3.10/json/decoder.py", line 337, in decode obj, end = self.raw_decode(s, idx=_w(s, 0).end()) File "/usr/local/lib/python3.10/json/decoder.py", line 355, in raw_decode raise JSONDecodeError("Expecting value", s, err.value) from None json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0) error: exit status 255
```
根源在于，解析时 `json.loads` 从第 1 个字符开始就遇到了非 json 内容，于是抛出异常，
```shell
“Expecting value: line 1 column 1 (char 0)”
```
需要先剥离 json string 最外层带 `<think>…</think>` 标记，之后再进行类型转换，

代码执行模块源代码，
```python
import json 
def main(arg1): 
    data = json.loads(arg1.strip()) 
    return { 
        "result": data 
    }
```
修改代码，
```python
import json
import re

def main(arg1: str):
    # 去掉 <think>...</think> 段落（非贪婪匹配）
    json_part = re.sub(r'<think>.*?</think>', '', arg1, flags=re.S).strip()
    
    # 如果去掉后为空，直接抛异常
    if not json_part:
        raise ValueError("No JSON content found after removing <think> block")
    
    data = json.loads(json_part)
    return {"result": data}
```
