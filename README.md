
# graph-analytics-starter-pack

[![Awesome](https://awesome.re/badge.svg)](https://github.com/guaiyoui/graph-analytics-starter-pack) 
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Made With Love](https://img.shields.io/badge/Made%20With-Love-red.svg)](https://github.com/chetanraj/awesome-github-badges)


Graph Analytics和AI4DB相关学习资料/路径

---

## 目录
- [导论](#导论)
- [视频课程](#视频课程)
- [研究领域](#研究领域)
    - [Cohesive Subgraph Discovery](#CohesiveSubgraphDiscovery)
    - [Generalized Anomaly Detection](#GeneralizedAnomalyDetection)
    - [Graph Similarity Computation](#GraphSimilarityComputation)
    - [Cardinality Estimation](#CardinalityEstimation)
    - [Subgraph Matching/Counting](#SubgraphMatching)
    - ......
- [其他](#其他)

## 导论
Graph Analytics和AI for Database是当今数据分析和人工智能领域中的两个重要分支。Graph Analytics通过分析图形数据来揭示数据背后的模式和关系，帮助人们更好地理解和利用数据。AI4DB则利用机器学习和人工智能技术来处理和管理大规模数据库、解决NP-hard的图相关问题，提高数据处理的效率和准确性。

在Graph Analytics和AI for Database领域，我们一般关注来自以下会议的学术论文:

<table>
    <tr>
        <th>Category</th><th>Conference</th><th>Link</th><th>Comment</th>
    </tr>
    <tr>
        <td rowspan="3">Database</td><td>SIGMOD</td><td> <a href="https://dblp.org/db/conf/sigmod/sigmod2022.html" target="_blank">DBLP, </a>  <a href="https://2022.sigmod.org/" target="_blank">官网</a>  </td> <td> Pioneering conference in Database</td>
    </tr>
    <tr>
        <td>VLDB</td><td><a href="http://vldb.org/pvldb/volumes/16/" target="_blank">VLDB</a></td> <td> </td>
    </tr>
    <tr>
        <td>ICDE</td><td> <a href="https://icde2023.ics.uci.edu/" target="_blank">ICDE</a> </td>  <td> </td>
    </tr>
    <tr>
        <td rowspan="3">AI/ML/DL</td><td>ICML</td><td> <a href="https://dblp.org/db/conf/icml/icml2022.html" target="_blank">ICML</a> </td>  <td> </td>
    </tr>
    <tr>
        <td>ICLR</td><td> <a href="https://dblp.org/db/conf/iclr/iclr2022.html" target="_blank">ICLR</a> </td>  <td> </td>
    </tr>
    <tr>
        <td>NeurIPS</td><td> <a href="https://papers.nips.cc/paper/2022" target="_blank">NeurIPS</a> </td>  <td> </td>
    </tr>
    <tr>
        <td rowspan="1">Data Mining</td><td>KDD</td><td> <a href="https://kdd.org/kdd2022/paperRT.html" target="_blank">KDD</a> </td>  <td> </td>
    </tr>
</table>



## 视频课程
### 主要课程
- [Stanford CS224W Machine Learning with Graphs 课程网址](http://web.stanford.edu/class/cs224w/)
- [Stanford CS224W Machine Learning with Graphs 课程视频](https://www.bilibili.com/video/BV1RZ4y1c7Co/?spm_id_from=333.337.search-card.all.click&vd_source=eb83fc5d65c5d8ce4504000a8b1a7056)
- [UNSW COMP9312 Data Analytics for Graphs](https://github.com/guaiyoui/awesome-graph-analytics/tree/main/COMP9312)

### 参考课程
- [Stanford CS520 Knowledge Graphs (2021)](https://www.bilibili.com/video/BV1hb4y1r7fF/?from=search&seid=6234955209527085652&spm_id_from=333.337.0.0&vd_source=eb83fc5d65c5d8ce4504000a8b1a7056)
- [Stanford CS246 大数据挖掘 (2019)](https://www.bilibili.com/video/BV1SC4y187x1/?from=search&seid=1692751967493851255&spm_id_from=333.337.0.0&vd_source=eb83fc5d65c5d8ce4504000a8b1a7056)
- [Stanford Course Explore](https://explorecourses.stanford.edu/search?view=catalog&academicYear=&page=0&q=CS&filter-departmentcode-CS=on&filter-coursestatus-Active=on&filter-term-Autumn=on)

### 重点章节
| Week  | Content  | Reading List  | Material  |
|---|---|---|---|
|1| Node Embedding  | [1: DeepWalk: Online Learning of Social Representations](https://arxiv.org/pdf/1403.6652.pdf)<br>[2: node2vec: Scalable Feature Learning for Networks](https://arxiv.org/pdf/1607.00653.pdf) | 节点表征是learning中最基础的一个问题。可以参考CS224W 3rd和COMP9312 week6的内容。传统的矩阵分解方法可以参考：[矩阵分解的python实现](https://blog.csdn.net/qq_43741312/article/details/97548944) | 
|2| Graph Neural Networks  | [1: Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/pdf/1609.02907.pdf)<br>[2: Graph Attention Networks](https://arxiv.org/pdf/1710.10903.pdf)   | 模型的结构是learning的核心(CS224W 4th)，reading list给了两个经典模型: GCN和GAT。关于神经网络中每一层的结构，可以参考tutorial: [Learning basic](https://github.com/guaiyoui/awesome-graph-analytics/blob/main/tutorials/Tutorial5_inclasscode.ipynb) | 
|3| GNN Augmentation and Training   |  [1: Hierarchical Graph Representation Learning with Differentiable Pooling](https://arxiv.org/pdf/1806.08804.pdf)<br>[2: Hyper-Path-Based Representation Learning for Hyper-Networks.](https://arxiv.org/abs/1908.09152) | 介绍整个GNN的流程，可以参考CS224W 6th。reading list给了其他representation learning的方法和在hypergraph上的拓展。怎么embedding可以参考tutorial: [Node embedding](https://github.com/guaiyoui/awesome-graph-analytics/blob/main/tutorials/tutorial6_Node_Embedding.ipynb)  |
|4| Theory of Graph Neural Networks  | [1: Anomaly Detection on Attributed Networks via Contrastive Self-Supervised Learning](https://arxiv.org/abs/2103.00113)<br>[2: Sub-graph Contrast for Scalable Self-Supervised Graph Representation Learning](https://arxiv.org/abs/2009.10273)  | 对神经网络的分析，可以参考CS224w 7th. reading list给了目前主流的self supervised的embedding方法和网络框架。怎么用learning来做基础任务，参考tutorial: [Downstream Application: node classification, link prediction and graph classification](https://github.com/guaiyoui/awesome-graph-analytics/blob/main/tutorials/Tutorial7_Downstream_Applications_Template.ipynb)  |
|5| Label Propagation on Graphs |  [1: GLSearch: Maximum Common Subgraph Detection via Learning to Search](http://proceedings.mlr.press/v139/bai21e/bai21e.pdf)<br>[2: Computing Graph Edit Distance via Neural Graph Matching](https://www.vldb.org/pvldb/vol16/p1817-cheng.pdf) | 参考CS224W 8th. reading list给了graph similarity computation (NP hard)问题的两种解决方法。自己实现GCN的结构，可以参考tutorial: [Graph Convolutional Network](https://github.com/guaiyoui/awesome-graph-analytics/blob/main/tutorials/tutorial_8.ipynb)  |
|6| Subgraph Matching and Counting |  [1: A Learned Sketch for Subgraph Counting](https://dl.acm.org/doi/pdf/10.1145/3448016.3457289)<br>[2: Neural Subgraph Counting with Wasserstein Estimator](https://dl.acm.org/doi/pdf/10.1145/3514221.3526163) | 参考CS224W 12nd. reading list给了subgraph counting (NP hard)问题的两篇文章(from sigmod 2021 and sigmod 2022)。subcon论文的仓库代码在 [Contrastive Learning on Graph](https://github.com/yzjiao/Subg-Con)  |


## 研究领域

<p id="CohesiveSubgraphDiscovery"></p>

### 1: Cohesive Subgraph Discovery
Cohesive Subgraph Discovery是一种在图形数据中寻找具有高度内聚性的子图的问题。

#### 1.1 Subgraph-model-based Community Search

#### 1.2 Metric-based Community Search
- DMCS : Density Modularity based Community Search [[paper]](https://dl.acm.org/doi/abs/10.1145/3514221.3526137)
- ..... To be continued


#### 1.3 Learning-based Community Search
- Query Driven-Graph Neural Networks for Community Search [[code]](https://github.com/lizJyl/Codes-for-Peer-Review-of-VLDB-August-337) [[paper]](https://arxiv.org/abs/2104.03583)
- ICS-GNN: lightweight interactive community search via graph neural network [[code]](https://github.com/guaiyoui/awesome-graph-analytics/blob/main/files/ics-gnn.zip) [[paper]](https://dl.acm.org/doi/pdf/10.14778/3447689.3447704)
- Community Search: A Meta-Learning Approach [[paper]](https://arxiv.org/abs/2201.00288)

<p id="GeneralizedAnomalyDetection"></p>

### 2: Generalized Anomaly Detection
Generalized Anomaly Detection包括了很多类似的问题，比如: anomaly detection, novelty detection, open set recognition, out-of-distribution detection 和 outlier detection.
#### 综述
- Generalized Out-of-Distribution Detection: A Survey [[paper]](https://arxiv.org/pdf/2110.11334.pdf)

#### Anomaly Detection
- ICLR2022 anomaly detection for tabular data with internal contrastive learning. [[paper]](https://openreview.net/forum?id=_hszZbt46bT)
- ..... To be continued

### Fraud Detection
- [入门综述论文](https://github.com/safe-graph/graph-fraud-detection-papers#survey-paper-back-to-top)
- [入门论文列表](https://github.com/safe-graph/graph-fraud-detection-papers)
- [入门代码demo](https://github.com/finint/antifraud)
- [TKDE Community Aware反洗钱 Anti-Money Laundering by Group-Aware Deep Graph Learning](https://ieeexplore.ieee.org/document/10114503)
- [AAAI Risk-Aware反诈骗 Semi-Supervised Credit Card Fraud Detection via Attribute-Driven Graph Representation](https://arxiv.org/pdf/2003.01171.pdf)
- [TKDE Spatial-Aware反诈骗 Graph Neural Network for Fraud Detection via Spatial-temporal Attention](https://ieeexplore.ieee.org/abstract/document/9204584)



<p id="GraphSimilarityComputation"></p>

### 3: Graph Similarity Computation

<p id="CardinalityEstimation"></p>

### 4: Cardinality Estimation

<p id="SubgraphMatching"></p>

### 5: Subgraph Matching/Counting









## 其他
- [PKU Lanco Lab 入门指导](https://github.com/guaiyoui/awesome-graph-analytics/blob/main/files/NLP%E5%85%A5%E9%97%A8%E6%8C%87%E5%AF%BC.pdf)
- [CS224w学习资料](https://yasoz.github.io/cs224w-zh/#/Introduction-and-Graph-Structure)

### 论文写作
- [Writing tips from MLNLP](https://github.com/MLNLP-World/Paper-Writing-Tips#%E7%99%BE%E5%AE%B6%E4%B9%8B%E8%A8%80)
- [在线latex编辑器](https://www.latexlive.com/)
### 画图
- [https://github.com/guanyingc/python_plot_utils](https://github.com/guanyingc/python_plot_utils)
### 工具
- [谷歌学术](https://scholar.google.com.hk/)
- [ChatGPT](https://chat.openai.com/), [POE: 集成多个语言模型](https://poe.com/ChatGPT)
- [ChatGPT学术润色的prompt参考](https://github.com/ashawkey/chatgpt_please_improve_my_paper_writing)