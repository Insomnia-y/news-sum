# 新闻摘要生成

<img src="https://cdn.jsdelivr.net/gh/1candoallthings/figure-bed@main/img/202211281927594.png" alt="image-20221128192713487" style="zoom: 25%;" />

<img src="C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20221128192846738.png" alt="image-20221128192846738" style="zoom: 25%;" />

## 参考文献

赛题来自[DataFountain比赛官网链接](https://www.datafountain.cn/competitions/541)

### 模型和预训练参数

mt5模型的实现来自[huggingface的transformers库](https://github.com/huggingface/transformers)

[预训练模型参数](https://huggingface.co/csebuetnlp/mT5_multilingual_XLSum)经过在包含45种语言的[XL-Sum](https://huggingface.co/datasets/csebuetnlp/xlsum)数据集上预训练得到

### 代码编写

加载transformers库中mt5预训练模型进行测试，参考博客[稀土掘金baseline](https://juejin.cn/post/7026590075051851789)

在赛题数据集上对mt5预训练模型进行微调，参考博客[Hugging Face 的 Transformers 库快速入门（八）：文本摘要任务](https://xiaosheng.run/2022/03/29/transformers-note-8.html)

## 小组分工

| 姓名 | 学号 | 分工                              |
| ---- | ---- | --------------------------------- |
| yyx  | -    | 编写文档                          |
| ypw  | -    | 编写mt5预训练模型的测试、微调代码 |
| lza  | -    | 实现模型改进                      |
| nyk  | -    | 实现模型改进                      |
| lrf  | -    |                                   |
| lwj  | -    |                                   |
| xxx  | -    |                                   |



## 环境配置

```sh
// 安装依赖
pip install -r requirements.txt  
// 下载预训练模型参数
git lfs install
git lfs pull
```



## 启动训练

测试mt5预训练模型在赛题数据集上的效果：

```
python main.py
```

对mt5预训练模型进行微调，拆分train_dataset.csv中的9000条数据，其中8500条用于train，500条用于valid：

```sh
python demo.py
```



## 评价指标

对于文本摘要任务，常用评估指标是 [ROUGE 值](https://en.wikipedia.org/wiki/ROUGE_(metric)) (short for Recall-Oriented Understudy for Gisting Evaluation)，它可以度量两个词语序列之间的词语重合率。ROUGE 值的召回率表示参考摘要在多大程度上被生成摘要覆盖，如果我们只比较词语，那么召回率就是：

![image-20221128192133540](https://cdn.jsdelivr.net/gh/1candoallthings/figure-bed@main/img/202211281921624.png)

准确率则表示生成的摘要中有多少词语与参考摘要相关：

![image-20221128192157919](https://cdn.jsdelivr.net/gh/1candoallthings/figure-bed@main/img/202211281921942.png)

最后再基于准确率和召回率来计算 F1 值。实际操作中，我们可以通过 [rouge 库](https://github.com/pltrdy/rouge)来方便地计算这些 ROUGE 值，例如 ROUGE-1 度量 uni-grams 的重合情况，ROUGE-2 度量 bi-grams 的重合情况，而 ROUGE-L 则通过在生成摘要和参考摘要中寻找最长公共子串来度量最长的单词匹配序列。