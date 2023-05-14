# awesome-graph-analytics
Graph Analytics和AL4DB相关学习资料/路径

---

## 目录
- [视频课程](#视频课程)
- [理论及论文](#理论及论文)
- [学习资料](#学习资料)
- [相关会议](#相关会议)
- [论文写作](#论文写作)
- [其他](#其他)


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
|1| Node Embedding  | [1: DeepWalk: Online Learning of Social Representations](https://arxiv.org/pdf/1403.6652.pdf)<br>[2: node2vec: Scalable Feature Learning for Networks](https://arxiv.org/pdf/1607.00653.pdf) |  [矩阵分解的python实现](https://blog.csdn.net/qq_43741312/article/details/97548944) | 
|2| Graph Neural Networks  | [1: Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/pdf/1609.02907.pdf)<br>[2: Graph Attention Networks](https://arxiv.org/pdf/1710.10903.pdf)   |  [Learning basic](https://github.com/guaiyoui/awesome-graph-analytics/blob/main/tutorials/Tutorial5_inclasscode.ipynb) | 
|3| GNN Augmentation and Training   |  [1: Hierarchical Graph Representation Learning with Differentiable Pooling](https://arxiv.org/pdf/1806.08804.pdf)<br>[2: Hyper-Path-Based Representation Learning for Hyper-Networks.](https://arxiv.org/abs/1908.09152) | [Node embedding](https://github.com/guaiyoui/awesome-graph-analytics/blob/main/tutorials/tutorial6_Node_Embedding.ipynb)  |
|4| Theory of Graph Neural Networks  | [1: Anomaly Detection on Attributed Networks via Contrastive Self-Supervised Learning](https://arxiv.org/abs/2103.00113)<br>[2: Sub-graph Contrast for Scalable Self-Supervised Graph Representation Learning](https://arxiv.org/abs/2009.10273)  | [Downstream Application](https://github.com/guaiyoui/awesome-graph-analytics/blob/main/tutorials/Tutorial7_Downstream_Applications_Template.ipynb)  |
|5| Label Propagation on Graphs |  [1: GLSearch: Maximum Common Subgraph Detection via Learning to Search](http://proceedings.mlr.press/v139/bai21e/bai21e.pdf)<br>[2: Computing Graph Edit Distance via Neural Graph Matching](https://www.vldb.org/pvldb/vol16/p1817-cheng.pdf) |  [Graph Convolutional Network](https://github.com/guaiyoui/awesome-graph-analytics/blob/main/tutorials/tutorial_8.ipynb)  |
|6| Subgraph Matching and Counting |  [1: A Learned Sketch for Subgraph Counting](https://dl.acm.org/doi/pdf/10.1145/3448016.3457289)<br>[2: Neural Subgraph Counting with Wasserstein Estimator](https://dl.acm.org/doi/pdf/10.1145/3514221.3526163) | [Contrastive Learning on Graph](https://github.com/yzjiao/Subg-Con)  |


## 理论及论文

### Traditional Community Search
- [DMCS : Density Modularity based Community Search](https://dl.acm.org/doi/abs/10.1145/3514221.3526137)
- .... TBD

### Learning based Community Search
- Query Driven-Graph Neural Networks for Community Search [[code]](https://github.com/lizJyl/Codes-for-Peer-Review-of-VLDB-August-337) [[paper]](https://arxiv.org/abs/2104.03583)
- ICS-GNN: lightweight interactive community search via graph neural network [[code]](https://github.com/lizJyl/Codes-for-Peer-Review-of-VLDB-August-337) [[paper]](https://dl.acm.org/doi/pdf/10.14778/3447689.3447704)

### Anomaly Detection
- 

### Fraud Detection
- [入门综述论文](https://github.com/safe-graph/graph-fraud-detection-papers#survey-paper-back-to-top)
- [入门论文列表](https://github.com/safe-graph/graph-fraud-detection-papers)
- [入门代码demo](https://github.com/finint/antifraud)
- [TKDE Community Aware反洗钱 Anti-Money Laundering by Group-Aware Deep Graph Learning](https://ieeexplore.ieee.org/document/10114503)
- [AAAI Risk-Aware反诈骗 Semi-Supervised Credit Card Fraud Detection via Attribute-Driven Graph Representation](https://arxiv.org/pdf/2003.01171.pdf)
- [TKDE Spatial-Aware反诈骗 Graph Neural Network for Fraud Detection via Spatial-temporal Attention](https://ieeexplore.ieee.org/abstract/document/9204584)


## 学习资料
- [https://yasoz.github.io/cs224w-zh/#/Introduction-and-Graph-Structure](https://yasoz.github.io/cs224w-zh/#/Introduction-and-Graph-Structure)


## 相关会议

### Database三大会
#### 1: SIGMOD
- DBLP [https://dblp.org/db/conf/sigmod/sigmod2022.html](https://dblp.org/db/conf/sigmod/sigmod2022.html)
- 官网 [https://2022.sigmod.org/](https://2022.sigmod.org/)
- 如果要在DBLP中切换成SIGMODxxxx, 其中xxxx代表年份，只需要把上面最后的sigmod2022变成sigmodxxxx

#### 2: VLDB
- [http://vldb.org/pvldb/volumes/16/](http://vldb.org/pvldb/volumes/16/)

#### 3: ICDE
- [https://icde2023.ics.uci.edu/](https://icde2023.ics.uci.edu/)

### 机器/深度学习三大会

#### 1: ICML
- [https://dblp.org/db/conf/icml/icml2022.html](https://dblp.org/db/conf/icml/icml2022.html)
- 要修改年份的话，与上面类似


#### 2: ICLR
- [https://dblp.org/db/conf/iclr/iclr2022.html](https://dblp.org/db/conf/iclr/iclr2022.html)


#### 3: NeurIPS
- [https://papers.nips.cc/paper/2022](https://papers.nips.cc/paper/2022)

### Data mining会议

#### 1: KDD
- [https://kdd.org/kdd2022/paperRT.html](https://kdd.org/kdd2022/paperRT.html)


## 论文写作

### 写作
- [Writing tips from MLNLP](https://github.com/MLNLP-World/Paper-Writing-Tips#%E7%99%BE%E5%AE%B6%E4%B9%8B%E8%A8%80)
- [在线latex编辑器](https://www.latexlive.com/)


### 画图
- [https://github.com/guanyingc/python_plot_utils](https://github.com/guanyingc/python_plot_utils)


### 工具
- [谷歌学术](https://scholar.google.com.hk/)
- [ChatGPT](https://poe.com/ChatGPT)
- [ChatGPT学术润色的prompt参考](https://github.com/ashawkey/chatgpt_please_improve_my_paper_writing)




## 其他
- [PKU Lanco Lab 入门指导](https://github.com/guaiyoui/awesome-graph-analytics/blob/main/files/NLP%E5%85%A5%E9%97%A8%E6%8C%87%E5%AF%BC.pdf)
