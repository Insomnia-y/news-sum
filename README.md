

# 新闻摘要生成

<img src="https://cdn.jsdelivr.net/gh/1candoallthings/figure-bed@main/img/202211281927594.png" alt="image-20221128192713487" style="zoom: 25%;" />


## 参考资料

赛题来自[DataFountain比赛官网链接](https://www.datafountain.cn/competitions/541)

### 模型和预训练参数

mt5模型的实现来自[huggingface的transformers库](https://github.com/huggingface/transformers)

[预训练模型参数](https://huggingface.co/csebuetnlp/mT5_multilingual_XLSum)经过在包含45种语言的[XL-Sum](https://huggingface.co/datasets/csebuetnlp/xlsum)数据集上预训练得到

### 代码编写

加载transformers库中mt5预训练模型进行测试，参考博客[稀土掘金baseline](https://juejin.cn/post/7026590075051851789)

在赛题数据集上对mt5预训练模型进行微调，参考博客[Hugging Face 的 Transformers 库快速入门（八）：文本摘要任务](https://xiaosheng.run/2022/03/29/transformers-note-8.html)

## 小组成员与分工

| 姓名 | 学号 | 分工                                        |
| ---- | ---- | ------------------------------------------- |
| 李润X  | -    | 本项目中，transformer框架                         |
| 梁XX  | -    | 本项目中，解决方案一：mt5预训练模型的调研       |
| 袁XX  | -    | 本项目解决方案一：mt5预训练模型的测试、微调 |
| 卢XX  | -    | 本项目解决方案二：2022年ACL文本摘要sota     |
| 李子X  | -    | discussion中，“MRC介绍”部分                   |
| 宁XX  | -    | discussion中，“VQA介绍”部分                   |
| 叶XX  | -    | discussion中，“用MRC的思路实现VQA”部分        |



## 问题描述

依据真实的新闻文章，利用机器学习相关技术，建立高效的摘要生成模型，为新闻文档生成相应的内容摘要。

## 技术方案介绍

> 写方法介绍，不涉及具体代码

### 模型

#### 1. mt5

（todo lwj）

#### 2. 2022ACL

（todo lpr）

### 框架

#### 1. transformers

Transformers 提供了数以千计的预训练模型，包括本项目的文本摘要任务，提供了便于快速下载和使用的API，可以将预训练模型用在本项目给定的以CNN 和Daily Mail的报刊新闻为文章基础的数据集上微调，且模型文件可单独使用，方便魔改和快速实验。

在本项目中，直接使用 Transformers 库自带的 MT5ForConditionalGeneration类来构建模型，并加载预训练模型：
```python 
from transformers import AutoModelForSeq2SeqLM
model_name = "mT5_multilingual_XLSum"
model = MT5ForConditionalGeneration.from_pretrained(model_name)
```

本项目使用的Transformers库的类图如下：
![image](https://user-images.githubusercontent.com/60568578/207641118-ec4bac0d-ada5-4567-ba33-a07aa2912468.png)


## 技术方案实现

> 简单介绍一下代码即可，例如主要的API分别实现了什么功能

#### 1. mt5

（todo ypw）

#### 2. 2022ACL

（todo lpr）

## 性能测试

### 数据集说明

数据是以CNN 和Daily Mail的报刊新闻为文章基础的，包含新闻文章和摘要等，该类数据集被广泛应用于摘要生成和阅读理解等应用场景。

数据文件夹包含4个文件，依次为：

| 文件类别 | 文件名         | 文件内容                    |
| -------- | -------------- | --------------------------- |
| 训练集   | train.csv      | 训练数据集，对应的摘要      |
| 测试集   | test.csv       | 测试数据集，无对应的摘要    |
| 提交样例 | submission.csv | 仅有两个字段Index \t Target |

### 评价指标

对于文本摘要任务，常用评估指标是 [ROUGE 值](https://en.wikipedia.org/wiki/ROUGE_(metric)) (short for Recall-Oriented Understudy for Gisting Evaluation)，它可以度量两个词语序列之间的词语重合率。ROUGE 值的召回率表示参考摘要在多大程度上被生成摘要覆盖，如果我们只比较词语，那么召回率就是：

![image-20221128192133540](https://cdn.jsdelivr.net/gh/1candoallthings/figure-bed@main/img/202211281921624.png)

准确率则表示生成的摘要中有多少词语与参考摘要相关：

![image-20221128192157919](https://cdn.jsdelivr.net/gh/1candoallthings/figure-bed@main/img/202211281921942.png)

最后再基于准确率和召回率来计算 F1 值。

ROUGE-1 度量 uni-grams 的重合情况，ROUGE-2 度量 bi-grams 的重合情况，而 ROUGE-L 则通过在生成摘要和参考摘要中寻找最长公共子串来度量最长的单词匹配序列。实际操作中，我们可以通过 [rouge 库](https://github.com/pltrdy/rouge)来方便地计算这些 ROUGE 值。

---

本赛题采用ROUGE-L值进行评价，详细评分算法如下：

![image-20221128200848145](https://cdn.jsdelivr.net/gh/1candoallthings/figure-bed@main/img/202211282008169.png)

其中 $\operatorname{LCS}(X, Y)$ 是 $X$ 和 $Y$ 的最长公共子序列的长度， $m$ 和 $n$ 分别表示人工标注摘要和机器自动摘要的长度（一般就是所含词的个数）， $R_{lcs}$ 和 $P_{lcs}$ 分别表示召回率和准确率， $F_{lcs}$ 表示ROUGE-L。

### 实验结果

将**预训练模型**和**微调后模型**进行性能对比：

|                           | ROUGE-1 | ROUGE-2 | ROUGE-L |
| ------------------------- | ------- | ------- | ------- |
| 预训练模型                | 80.03   | 39.91   | 67.68   |
| 微调后模型（Epoch=2）     | 87.91   | 53.17   | 77.47   |
| mt5微调后模型（训练结束） | （todo ypw）    | todo    | todo    |
| 2022ACL                   | （todo lpr）    | todo    | todo    |

预训练模型：

![image-20221128195145622](https://cdn.jsdelivr.net/gh/1candoallthings/figure-bed@main/img/202211281951652.png)

微调后模型（Epoch=2）：

![image-20221128195334931](https://cdn.jsdelivr.net/gh/1candoallthings/figure-bed@main/img/202211281953957.png)



## Readme

### 环境配置

```sh
// 安装依赖
pip install -r requirements.txt  
// 下载预训练模型参数
git lfs install
git lfs pull
```

### 启动训练

测试mt5预训练模型在赛题数据集上的效果：

```
python main.py
```

对mt5预训练模型进行微调，拆分train_dataset.csv中的9000条数据，其中8500条用于train，500条用于valid：

```sh
python demo.py
```
