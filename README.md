# Text_summarization-Using-BART

本科期间的毕业设计，功能是实现一个文本摘要工作，将一段新闻（约3-500字）进行摘要生成标题。

# Run Environment

```powershell
conda install --file requirements.txt
```

> 除此之外，需要安装Streamlit库用于生成web界面。
> `pip install streamlit`


## 实验环境

|系统|python|pytorch|CUDA|GPU
|---|---|---|---|---|
|Ubuntu 22.04 LTS|Python 3.10.9|2.0.0|11.7|RTX 3090

## [数据集](https://pan.baidu.com/s/18nXMGWpnaGo0PeK8tT2TUA?pwd=cccc)

训练语料库采用NLPCC2017，总共有50000条新闻文本摘要对。
> 来源: 自然语言处理与中文计算会议（CCF Conference on Natural Language Processing &. Chinese Computing，NLPCC）。

# Train

1. 在train目录下，`train_bart.ipynb`是一个Step by Step的训练过程。 
2. main.py是一个使用stream-lite库生成简单Web界面，并且可以演示模型效果的库。
2. [数据集的百度云连接! 不维护其持久有效]

# 实现思路

使用一个中文数据集对预训练模型BART进行微调，使其能够在文本摘要这个下游任务中应用。
