
# graph-analytics-starter-pack

[![Awesome](https://awesome.re/badge.svg)](https://github.com/guaiyoui/graph-analytics-starter-pack) 
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Made With Love](https://img.shields.io/badge/Made%20With-Love-red.svg)](https://github.com/chetanraj/awesome-github-badges)


Learning materials/paths related to Graph Analytics and AI4DB.

## <img src="./pics/organizer.png" width="25" />Main Contributor

Thanks to the following people for organizing and guiding this project

<a href="https://github.com/guaiyoui"> <img src="pics/jianwei.jpeg"  width="80" >  </a> 
<a href="https://github.com/cs-kaiwang">  <img src="pics/kaiwang.png"  width="80" ></a> 
<a href="https://github.com/valleysprings">  <img src="pics/jiawei.png"  width="80" ></a> 
<a href="https://github.com/shenmuxing">  <img src="pics/jingye.jpeg"  width="80" > </a> 
<a href="https://github.com/ShunyangLi">  <img src="pics/shunyang.jpeg"  width="80" >  </a> 
<a href="hhttps://github.com/jerryUNSW">  <img src="pics/yizhang.png"  width="80" >  </a> 
<a href="https://github.com/Helloat123">  <img src="pics/ruicheng.jpeg"  width="80" >  </a> 
<a href="https://github.com/YingAU">  <img src="pics/yingzhang.jpeg"  width="80" > </a> 
<a href="https://github.com/AllanJinYosoro">  <img src="pics/AllanJinYosoro.jpg"  width="80" > </a>


<!-- - [graph-analytics-starter-pack](#graph-analytics-starter-pack) -->
  <!-- - [Table-of-Contents](#Table-of-Contents) -->
---
<!-- 1: Cohesive Subgraph Discovery -->

## Table-of-Contents

- [graph-analytics-starter-pack](#graph-analytics-starter-pack)
  - [Main Contributor](#main-contributor)
  - [Table-of-Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Courses](#courses)
    - [Main Courses](#main-courses)
    - [Reference Courses](#reference-courses)
    - [Key Chapters](#key-chapters)
  - [1: Cohesive Subgraph Discovery](#1-cohesive-subgraph-discovery)
  - [2: Generalized Anomaly Detection](#2-generalized-anomaly-detection)
    - [2.1 Survey of anomaly detection and Benchmarks](#21-survey-of-anomaly-detection-and-benchmarks)
    - [2.2 Anomaly Detection](#22-anomaly-detection)
    - [2.3 Fraud Detection](#23-fraud-detection)
  - [3: AIGC-LLM](#3-aigc-llm)
  - [4: Differential Privacy](#4-differential-privacy)
  - [5: Graph Analytics on GPUs](#5-graph-analytics-on-gpus)
  - [6: Graph Similarity Computation](#6-graph-similarity-computation)
  - [7: Subgraph Matching and Counting](#7-subgraph-matching-and-counting)
  - [8: Cardinality Estimation](#8-cardinality-estimation)
  - [9: Graph for DB and tabular data](#9-graph-for-db-and-tabular-data)
  - [10: Vector Database](#10-vector-database)
  - [12: GNN-based recommendation system](#12-gnn-based-recommendation-system)
    - [GNN-based Collaborative Filtering](#gnn-based-collaborative-filtering)
    - [GNN-based session-based/sequence-based Rec.Sys.](#gnn-based-session-basedsequence-based-recsys)
    - [GNN-based substitute \& complement Rec.Sys.](#gnn-based-substitute--complement-recsys)
    - [GNN-based cold-start Rec.Syc.](#gnn-based-cold-start-recsyc)
  - [13: Others](#13-others)
    - [PaperWriting](#paperwriting)
    - [FigureDrawing](#figuredrawing)
    - [Tools](#tools)


## Introduction
Graph Analytics and AI for Database are two important branches in today's data analysis and artificial intelligence fields. Graph Analytics helps people better understand and utilize data by analyzing graph data to reveal patterns and relationships behind the data. AI4DB uses machine learning and artificial intelligence techniques to process and manage large-scale databases, solve NP-hard graph-related problems, and improve the efficiency and accuracy of data processing.

In the fields of Graph Analytics and AI for Database, we generally focus on academic papers from the following conferences:

<table>
    <tr>
        <th>Category</th><th>Conference</th><th>Link</th><th>Comment</th>
    </tr>
    <tr>
        <td rowspan="3">Database</td><td>SIGMOD</td><td> <a href="https://dblp.org/db/conf/sigmod/sigmod2022.html" target="_blank">DBLP, </a>  <a href="https://2022.sigmod.org/" target="_blank">Official website</a>  </td> <td> Pioneering conference in Database</td>
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
[A Comprehensive Survey on Graph Neural Networks] https://ieeexplore.ieee.org/document/9046288

GNN is commonly implemented using PyTorch. Here's a series of tutorials using PyTorch: https://pytorch-geometric.readthedocs.io/en/latest/


If you're not familiar enough with deep learning, you can check out (the following are for reference, you don't need to look at all of them):

This book: Deep Learning https://github.com/exacity/deeplearningbook-chinese
This course: Search for "李宏毅 机器学习" (Hung-yi Lee Machine Learning) on Bilibili
Course + Book: Hands-on Deep Learning

Other potentially valuable Stanford University courses:
cs224n (NLP), cs224w (graph), cs229 (ML), cs231n (CV), cs285 (RL)




## Courses
### Main Courses
- [Stanford CS224W Machine Learning with Graphs: Course Website](http://web.stanford.edu/class/cs224w/)
- [Stanford CS224W Machine Learning with Graphs: Course Video](https://www.bilibili.com/video/BV1RZ4y1c7Co/?spm_id_from=333.337.search-card.all.click&vd_source=eb83fc5d65c5d8ce4504000a8b1a7056)
- [UNSW COMP9312 Data Analytics for Graphs](https://github.com/guaiyoui/awesome-graph-analytics/tree/main/COMP9312)

### Reference Courses
- [Stanford CS520 Knowledge Graphs (2021)](https://www.bilibili.com/video/BV1hb4y1r7fF/?from=search&seid=6234955209527085652&spm_id_from=333.337.0.0&vd_source=eb83fc5d65c5d8ce4504000a8b1a7056)
- [Stanford CS246 Big Data Mining (2019)](https://www.bilibili.com/video/BV1SC4y187x1/?from=search&seid=1692751967493851255&spm_id_from=333.337.0.0&vd_source=eb83fc5d65c5d8ce4504000a8b1a7056)
- [Stanford Course Explore](https://explorecourses.stanford.edu/search?view=catalog&academicYear=&page=0&q=CS&filter-departmentcode-CS=on&filter-coursestatus-Active=on&filter-term-Autumn=on)

### Key Chapters
| Week  | Content  | Reading List  | Material  |
|---|---|---|---|
|1| Node Embedding  | [1: DeepWalk: Online Learning of Social Representations](https://arxiv.org/pdf/1403.6652.pdf)<br>[2: node2vec: Scalable Feature Learning for Networks](https://arxiv.org/pdf/1607.00653.pdf) | Node representation is one of the most fundamental problems in graph learning. You can refer to the content of CS224W 3rd and COMP9312 week 6. For traditional matrix factorization methods, you can refer to:：[matrix factorization](https://blog.csdn.net/qq_43741312/article/details/97548944) | 
|2| Graph Neural Networks  | [1: Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/pdf/1609.02907.pdf)<br>[2: Graph Attention Networks](https://arxiv.org/pdf/1710.10903.pdf)   | The structure of the model is the core of learning (CS224W 4th). The reading list provides two classic models: GCN and GAT. Regarding the structure of each layer in neural networks, you can refer to the tutorial: [Learning basic](https://github.com/guaiyoui/awesome-graph-analytics/blob/main/tutorials/Tutorial5_inclasscode.ipynb) | 
|3| GNN Augmentation and Training   |  [1: RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space](https://arxiv.org/pdf/1902.10197.pdf)<br>[2: Hyper-Path-Based Representation Learning for Hyper-Networks.](https://arxiv.org/abs/1908.09152) | For an introduction to the entire GNN process, you can refer to CS224W 6th. The reading list provides other representation learning methods and their extensions on knowledge graphs/hypergraphs. For how to do embedding, you can refer to the tutorial: [Node embedding](https://github.com/guaiyoui/awesome-graph-analytics/blob/main/tutorials/tutorial6_Node_Embedding.ipynb)  |
|4| Theory of Graph Neural Networks  | [1: Anomaly Detection on Attributed Networks via Contrastive Self-Supervised Learning](https://arxiv.org/abs/2103.00113)<br>[2: Sub-graph Contrast for Scalable Self-Supervised Graph Representation Learning](https://arxiv.org/abs/2009.10273)  | For analysis of neural networks, you can refer to CS224W 7th. The reading list provides current mainstream self-supervised embedding methods and network frameworks. For how to use learning to perform basic tasks, refer to the tutorial: [Downstream Application: node classification, link prediction and graph classification](https://github.com/guaiyoui/awesome-graph-analytics/blob/main/tutorials/Tutorial7_Downstream_Applications_Template.ipynb)  |
|5| Label Propagation on Graphs |  [1: GLSearch: Maximum Common Subgraph Detection via Learning to Search](http://proceedings.mlr.press/v139/bai21e/bai21e.pdf)<br>[2: Computing Graph Edit Distance via Neural Graph Matching](https://www.vldb.org/pvldb/vol16/p1817-cheng.pdf) | Refer to CS224W 8th. The reading list provides two methods for solving the graph similarity computation (NP-hard) problem. To implement the GCN structure yourself, you can refer to the tutorial: [Graph Convolutional Network](https://github.com/guaiyoui/awesome-graph-analytics/blob/main/tutorials/tutorial_8.ipynb)  |
|6| Subgraph Matching and Counting |  [1: A Learned Sketch for Subgraph Counting](https://dl.acm.org/doi/pdf/10.1145/3448016.3457289)<br>[2: Neural Subgraph Counting with Wasserstein Estimator](https://dl.acm.org/doi/pdf/10.1145/3514221.3526163) | Refer to CS224W 12th. The reading list provides two papers on the subgraph counting (NP-hard) problem (from SIGMOD 2021 and SIGMOD 2022). The repository code for the SubCon paper is at [Contrastive Learning on Graph](https://github.com/yzjiao/Subg-Con)  |


<p id="CohesiveSubgraphDiscovery"></p>

## 1: Cohesive Subgraph Discovery

Cohesive Subgraph Discovery is a problem of finding highly cohesive subgraphs in graph data. This survey: [A Survey on Machine Learning Solutions for Graph Pattern Extraction](https://arxiv.org/abs/2204.01057) (pay close attention to Ch2.6 community search) clearly explains several baseline articles for our work.

For more details, please refer to [1: Cohesive Subgraph Discovery](./sections/CohesiveSubgraph/)


<p id="GeneralizedAnomalyDetection"></p>

## 2: Generalized Anomaly Detection
Generalized Anomaly Detection包括了很多类似的问题，比如: anomaly detection, novelty detection, open set recognition, out-of-distribution detection 和 outlier detection.

### 2.1 Survey of anomaly detection and Benchmarks
- Generalized Out-of-Distribution Detection: A Survey [[paper]](https://arxiv.org/pdf/2110.11334.pdf)

- DGraph: A Large-Scale Financial Dataset for Graph Anomaly Detection [[paper]](https://arxiv.org/abs/2207.03579) [[project page]](https://dgraph.xinye.com/dataset)
- ADBench: Anomaly Detection Benchmark [[paper]](https://proceedings.neurips.cc/paper_files/paper/2022/hash/cf93972b116ca5268827d575f2cc226b-Abstract-Datasets_and_Benchmarks.html) [[project page]](https://github.com/Minqi824/ADBench/)

### 2.2 Anomaly Detection

| Conference | Paper  |  Material | Abstract | Highlights |
|---|---|---|---|---|
|ICLR2022|[Anomaly detection for tabular data with internal contrastive learning.](https://openreview.net/forum?id=_hszZbt46bT)|[[code]](https://openreview.net/forum?id=_hszZbt46bT)| KNN and Contrastive Learning for Tabular data| -- |
|ICDE2023|[Unsupervised Graph Outlier Detection: Problem Revisit, New Insight, and Superior Method.](https://fanzhangcs.github.io/)|---| --- | --- |


### 2.3 Fraud Detection
- [入门综述论文](https://github.com/safe-graph/graph-fraud-detection-papers#survey-paper-back-to-top)
- [入门论文列表](https://github.com/safe-graph/graph-fraud-detection-papers)
- [入门代码demo](https://github.com/finint/antifraud)
- [TKDE Community Aware反洗钱 Anti-Money Laundering by Group-Aware Deep Graph Learning](https://ieeexplore.ieee.org/document/10114503)
- [AAAI Risk-Aware反诈骗 Semi-Supervised Credit Card Fraud Detection via Attribute-Driven Graph Representation](https://arxiv.org/pdf/2003.01171.pdf)
- [TKDE Spatial-Aware反诈骗 Graph Neural Network for Fraud Detection via Spatial-temporal Attention](https://ieeexplore.ieee.org/abstract/document/9204584)

## 3: AIGC-LLM

Please refer to [3: AIGC-LLM](./sections/LLM/)




## 4: Differential Privacy

Please refer to [4: Differential Privacy](./sections/DifferentialPrivacy/)

and 

DP & ML:
https://github.com/JeffffffFu/Awesome-Differential-Privacy-and-Meachine-Learning



## 5: Graph Analytics on GPUs

Please refer to [5: Graph Analytics on GPUs](./sections/GPU/)




## 6: Graph Similarity Computation

<p id="SubgraphMatching"></p>

## 7: Subgraph Matching and Counting

<p id="CardinalityEstimation"></p>

## 8: Cardinality Estimation

<p id="Graph4DB"></p>

## 9: Graph for DB and tabular data

Graphs are a valuable tool for representing connections between entities, while tabular or relational data is a convenient and user-friendly way to store information. Researchers frequently employ graphs to depict interdependencies among records, attributes, elements, and schemas within and across tables. It is worth noting that in contemporary usage, the term "tabular deep learning" is often used to refer to the application of deep learning techniques to relational data organized as records, while the term "database" is often reserved to refer specifically to the software and infrastructure used to manage and manipulate such data.


| Conference | Paper  |  Material | Abstract | Highlights |
|---|---|---|---|---|
|PODS2023|[Databases as Graphs: Predictive Queries for Declarative Machine Learning.](https://dl.acm.org/doi/abs/10.1145/3584372.3589939)|---| Using hypergraph to model the relationship behind the records| -- |
|---|[Enabling tabular deep learning when d ge n with an auxiliary knowledge graph](https://arxiv.org/abs/2306.04766)|---| Capture the relation between two attributes by KG | -- |
|CIKM22|[Local Contrastive Feature learning for Tabular Data](https://dl.acm.org/doi/10.1145/3511808.3557630)|---| Capture the relation between two attributes by maximum spanning tree | -- |
|NIPS22|[Learning enhanced representations for tabular data via neighborhood propagation](https://arxiv.org/abs/2206.06587)|---| --- | -- |
|dlpkdd2021|[TabGNN: Multiplex Graph Neural Network for Tabular Data Prediction](https://arxiv.org/abs/2108.09127)|---| --- | -- |






<p id="VectorDB"></p>

## 10: Vector Database

Similarity search at a very large scale.

| Conference | Paper  |  Material | Abstract | Highlights |
|---|---|---|---|---|
|SIGMOD2023|[Near-Duplicate Sequence Search at Scale for Large Language Model Memorization.](https://dl.acm.org/doi/abs/10.1145/3589324)|---| ---| -- |

Talk 1: [Vector Database for Large Language Models in Production (Sam Partee)](https://www.youtube.com/watch?v=9VgpXcfJYvw)

## 12: GNN-based recommendation system

推荐系统涵盖众多子领域，难免挂一漏万，仅在此介绍部分应用GNN的推荐系统子领域。

Recommender systems encompass numerous sub-domains, and it's inevitable to miss some areas. Here we only introduce some sub-domains of recommender systems that apply GNN.

### GNN-based Collaborative Filtering

Collaborative Filtering (CF) is one of the most classic and widely used methods in recommender systems. Its core idea is to find association patterns between similar users or items based on historical user behavior data. It mainly falls into two categories: User-based CF makes recommendations by finding similar user groups, while Item-based CF makes recommendations through similarity relationships between items.

Here are some cornerstone papers of GNN-based CF. Some of which, like LightGCN, still serves as baseline or even benchmark component in many SOTA researches.

| Paper  | Conference  | Year  | Highlights |
|---|---|---|---|
| Graph Convolutional Neural Networks for Web-Scale Recommender Systems| KDD|2018 |First paper of GNN-based Collaborative Filtering, PinSAGE |
| Neural Graph Collaborative Filtering | SIGIR|2019 |Refinement of PinSAGE, NGCF |
|LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation | SIGIR| 2020| The most popular benchmark model, a refinement version of NGCF, LightGCN|
| Self-supervised Graph Learning for Recommendation| SIGIR|2021 |One of the first papers in GNN-based CF utilizing self-supervised method, SGL |

### GNN-based session-based/sequence-based Rec.Sys.

Session-based Recommender Systems focus on predicting the next item within an ongoing session, where a session is typically a short-term interaction sequence (such as a browsing session) with no user identification required. The key problem is: Given a sequence of user interactions $S = \{i_1, i_2, \cdots, i_t\}$ in the current session, predict the next item $i_{t+1}$ that the user is likely to interact with.

Sequence-based Recommender Systems consider the complete historical sequence of user interactions across multiple sessions over time. The problem is defined as: Given a user u and their entire historical interaction sequence $H = \{i_1, i_2, \cdots, i_n\}$ ordered by timestamp, predict the next items that the user will interact with in the future.

For detail information about session-based Rec.Sys. and Sequence-based Rec.Sys., please refer to:
-  [Graph and Sequential Neural Networks in Session-based Recommendation: A Survey](https://arxiv.org/abs/2408.14851)

Here are some cornerstone papers of GNN-based session-based/sequence-based Rec.Sys.

| Paper  | Conference  | Year  | Highlights |
|---|---|---|---|
| Session-Based Recommendation with Graph Neural Networks| AAAI|2018 |SR-GNN, still a popular baseline choice. |

虽然但是，如果您想在此领域做出贡献，三思而后行：我相比序列模型，尤其是Transformer架构模型，有什么优势？

### GNN-based substitute & complement Rec.Sys.
Substitute recommendation aims to suggest interchangeable items for a given query item.  Most traditional methods infer substitute relationships through item similarities, and extract semantic information from consumer reviews for this purpose. Recently, due to the semantic connection between substitutes and complements (e.g., a substitute’s complement is often also a complement of the original item) , and with the development of GNN, current mainstream models primarily use networks of co-view and co-purchase relationships to learn substitute relationships . They employ various methods to explore the latent relationships between different item interactions.

Complement recommendation aims to suggest complement items (like mouse for a computer, game handle for a PS5) for a given query item. Due to semantic complextity, most methods utilize GNN-based methods just like Sub. Rec.Sys.

Here are some cornerstone papers of GNN-based substitute & complement Rec.Sys.

| Paper  | Conference  | Year  | Highlights |
|---|---|---|---|
| Inferring Networks of Substitutable and Complementary Products| KDD|2015 |Most popular dataset: Amazon datasets|
| Measuring the Value of Recommendation Links on Product Demand| ISR| 2019| A Bussiness paper (ISR, UTD24) if you need|
| Decoupled Graph Convolution Network for Inferring Substitutable and Complementary Items| CIKM| 2020| One of the first GNN-based Sub. & Com. Rec.Sys., predicting substitute and complement relationships simultaneously|
| Heterogeneous graph neural networks with neighbor-SIM attention mechanism for substitute product recommendation| AAAI| 2021| |
| Decoupled Hyperbolic Graph Attention Network for Modeling Substitutable and Complementary Item Relationships| CIKM| 2022| |
| Enhanced Multi-Relationships Integration Graph Convolutional Network for Inferring Substitutable and Complementary Items| AAAI| 2023| |

### GNN-based cold-start Rec.Syc.
Please refer to [Awesome-Cold-Start-Recommendation](https://github.com/YuanchenBei/Awesome-Cold-Start-Recommendation)
## 13: Others
- [PKU Lanco Lab's Introductory Guide](https://github.com/guaiyoui/awesome-graph-analytics/blob/main/files/NLP%E5%85%A5%E9%97%A8%E6%8C%87%E5%AF%BC.pdf)
- [CS224w Learning Materials](https://yasoz.github.io/cs224w-zh/#/Introduction-and-Graph-Structure)

### PaperWriting
- [Writing tips from MLNLP](https://github.com/MLNLP-World/Paper-Writing-Tips#%E7%99%BE%E5%AE%B6%E4%B9%8B%E8%A8%80)
- [Online LaTeX editor](https://www.overleaf.com/)
### FigureDrawing
- [https://github.com/guanyingc/python_plot_utils](https://github.com/guanyingc/python_plot_utils)
### Tools
- [Google Scholar](https://scholar.google.com.hk/)
- [ChatGPT](https://chat.openai.com/), [Claude](https://claude.ai/), [POE: Integrating multiple language models](https://poe.com/ChatGPT)
- [Reference prompts for academic editing with ChatGPT](https://github.com/ashawkey/chatgpt_please_improve_my_paper_writing)
