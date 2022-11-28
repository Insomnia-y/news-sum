

# 新闻摘要生成

<img src="https://cdn.jsdelivr.net/gh/1candoallthings/figure-bed@main/img/202211281927594.png" alt="image-20221128192713487" style="zoom: 25%;" />

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20221128192846738.png" alt="image-20221128192846738" style="zoom: 25%;" />

[TOC]

## 参考资料

赛题来自[DataFountain比赛官网链接](https://www.datafountain.cn/competitions/541)

### 模型和预训练参数

mt5模型的实现来自[huggingface的transformers库](https://github.com/huggingface/transformers)

[预训练模型参数](https://huggingface.co/csebuetnlp/mT5_multilingual_XLSum)经过在包含45种语言的[XL-Sum](https://huggingface.co/datasets/csebuetnlp/xlsum)数据集上预训练得到

### 代码编写

加载transformers库中mt5预训练模型进行测试，参考博客[稀土掘金baseline](https://juejin.cn/post/7026590075051851789)

在赛题数据集上对mt5预训练模型进行微调，参考博客[Hugging Face 的 Transformers 库快速入门（八）：文本摘要任务](https://xiaosheng.run/2022/03/29/transformers-note-8.html)

## 小组成员与分工

| 姓名 | 学号 | 分工                              |
| ---- | ---- | --------------------------------- |
| yyx  | -    | 编写文档                          |
| ypw  | -    | 编写mt5预训练模型的测试、微调代码 |
| lza  | -    | 实现模型改进                      |
| nyk  | -    | 实现模型改进                      |
| lrf  | -    |                                   |
| lwj  | -    |                                   |
| xxx  | -    |                                   |

## 问题描述

依据真实的新闻文章，利用机器学习相关技术，建立高效的摘要生成模型，为新闻文档生成相应的内容摘要。

## 技术方案

> 使用的模型 & 使用的框架

### 模型

mt5

TODO

### 框架

transformers

TODO

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
$$
\begin{equation}
\begin{aligned}
\mathrm{R}_{\mathrm{lcs}} & =\frac{\operatorname{LCS}(X, Y)}{m} \\
\mathrm{P}_{\mathrm{lcs}} & =\frac{\operatorname{LCS}(X, Y)}{n} \\

\mathrm{F}_{\mathrm{lcs}} & =\frac{\left(1+\beta^{2}\right) \mathrm{R}_{\mathrm{lcs}} \mathrm{P}_{\mathrm{lcs}}}{\mathrm{R}_{\mathrm{lcs}}+\beta^{2} \mathrm{P}_{\mathrm{lcs}}}
\end{aligned}
\end{equation}
$$
其中$\operatorname{LCS}(X, Y)$是$X$和$Y$的最长公共子序列的长度，$m$和$n$分别表示人工标注摘要和机器自动摘要的长度（一般就是所含词的个数），$\mathrm{R}_{\mathrm{lcs}}$和$\mathrm{P}_{\mathrm{lcs}}$ 分别表示召回率和准确率，$\mathrm{F}_{\mathrm{lcs}}$ 表示ROUGE-L。

### 实验结果

将**预训练模型**和**微调后模型**进行性能对比：

|                                             | ROUGE-1 | ROUGE-2 | ROUGE-L |
| ------------------------------------------- | ------- | ------- | ------- |
| 预训练模型                                  | 80.03   | 39.91   | 67.68   |
| 微调后模型（Epoch==2，时间为20221128-1950） | 87.91   | 53.17   | 77.47   |
| 微调后模型（训练结束）                      |         |         |         |

预训练模型：

![image-20221128195145622](https://cdn.jsdelivr.net/gh/1candoallthings/figure-bed@main/img/202211281951652.png)

微调后模型：

![image-20221128195334931](https://cdn.jsdelivr.net/gh/1candoallthings/figure-bed@main/img/202211281953957.png)

## 问题及解决方法

TODO

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

