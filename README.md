
# graph-analytics-starter-pack

[![Awesome](https://awesome.re/badge.svg)](https://github.com/guaiyoui/graph-analytics-starter-pack) 
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Made With Love](https://img.shields.io/badge/Made%20With-Love-red.svg)](https://github.com/chetanraj/awesome-github-badges)


Graph Analytics和AI4DB相关学习资料/路径

---

## 目录
- [graph-analytics-starter-pack](#graph-analytics-starter-pack)
  - [目录](#目录)
  - [导论](#导论)
  - [视频课程](#视频课程)
    - [主要课程](#主要课程)
    - [参考课程](#参考课程)
    - [重点章节](#重点章节)
  - [1: Cohesive Subgraph Discovery](#1-cohesive-subgraph-discovery)
    - [1.1 Subgraph-model-based Community Search](#11-subgraph-model-based-community-search)
    - [1.2 Metric-based Community Search](#12-metric-based-community-search)
    - [1.3 Learning-based Community Search](#13-learning-based-community-search)
  - [2: Generalized Anomaly Detection](#2-generalized-anomaly-detection)
    - [2.1 Survey of anomaly detection and Benchmarks](#21-survey-of-anomaly-detection-and-benchmarks)
    - [2.2 Anomaly Detection](#22-anomaly-detection)
    - [2.3 Fraud Detection](#23-fraud-detection)
  - [3: AIGC-LLM](#3-aigc-llm)
    - [3.1 Survey of AIGC-LLM](#31-survey-of-aigc-llm)
    - [3.2 Theory of AIGC-LLM](#32-theory-of-aigc-llm)
    - [3.3 Prompt Learning](#33-prompt-learning)
      - [Prompt **Engineering Techniques**](#prompt-engineering-techniques)
      - [In-context Learning](#in-context-learning)
      - [Parameter-Efficient Fine-Tuning (PEFT)](#parameter-efficient-fine-tuning-peft)
      - [Reasoning with Large Language Models](#reasoning-with-large-language-models)
      - [Multimodal Prompt](#multimodal-prompt)
      - [Evaluation \& Reliability](#evaluation--reliability)
      - [Others](#others)
    - [3.4 Foundation Models](#34-foundation-models)
      - [3.4.1 Encoder-only Architecture](#341-encoder-only-architecture)
      - [3.4.2 Decoder-only Architecture](#342-decoder-only-architecture)
      - [3.4.3 Encoder-decoder Architecture](#343-encoder-decoder-architecture)
      - [3.4.4 Other](#344-other)
    - [3.5 Related Repos](#35-related-repos)
    - [3.6 Datasets of LLM-AIGC](#36-datasets-of-llm-aigc)
    - [3.7 Tools for LLM-AIGC](#37-tools-for-llm-aigc)
      - [Open-Source LLMs](#open-source-llms)
      - [Prompt Learning](#prompt-learning)
      - [CoT](#cot)
      - [Development](#development)
      - [ChatBots](#chatbots)
  - [4: Graph Similarity Computation](#4-graph-similarity-computation)
  - [5: Subgraph Matching and Counting](#5-subgraph-matching-and-counting)
  - [6: Cardinality Estimation](#6-cardinality-estimation)
  - [7: Graph for DB and tabular data](#7-graph-for-db-and-tabular-data)
  - [8: Vector Database](#8-vector-database)
  - [9: Differential Privacy](#9-differential-privacy)
  - [10: 其他](#9-其他)
    - [论文写作](#论文写作)
    - [画图](#画图)
    - [工具](#工具)
- [graph analytics on GPUs](./GPU/)

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
GNN的全面介绍[A Comprehensive Survey on Graph Neural Networks](https://ieeexplore.ieee.org/document/9046288)

GNN常用pytorch实现，这里有一系列使用pytorch的教程https://pytorch-geometric.readthedocs.io/en/latest/



如果对深度学习不够了解，可以看（以下为参考，不需要都看）

* 这本书：Deep Learninghttps://github.com/exacity/deeplearningbook-chinese
* 这门课：Bilibili搜索 李宏毅 机器学习
* 课+书：动手学深度学习



其它可能有参考价值的斯坦福大学课程：cs224n(NLP) cs224w(graph) cs229(ML) cs231n(CV) cs285(RL)




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
|3| GNN Augmentation and Training   |  [1: RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space](https://arxiv.org/pdf/1902.10197.pdf)<br>[2: Hyper-Path-Based Representation Learning for Hyper-Networks.](https://arxiv.org/abs/1908.09152) | 介绍整个GNN的流程，可以参考CS224W 6th。reading list给了其他representation learning的方法和在knowledge graph/hypergraph上的拓展。怎么embedding可以参考tutorial: [Node embedding](https://github.com/guaiyoui/awesome-graph-analytics/blob/main/tutorials/tutorial6_Node_Embedding.ipynb)  |
|4| Theory of Graph Neural Networks  | [1: Anomaly Detection on Attributed Networks via Contrastive Self-Supervised Learning](https://arxiv.org/abs/2103.00113)<br>[2: Sub-graph Contrast for Scalable Self-Supervised Graph Representation Learning](https://arxiv.org/abs/2009.10273)  | 对神经网络的分析，可以参考CS224w 7th. reading list给了目前主流的self supervised的embedding方法和网络框架。怎么用learning来做基础任务，参考tutorial: [Downstream Application: node classification, link prediction and graph classification](https://github.com/guaiyoui/awesome-graph-analytics/blob/main/tutorials/Tutorial7_Downstream_Applications_Template.ipynb)  |
|5| Label Propagation on Graphs |  [1: GLSearch: Maximum Common Subgraph Detection via Learning to Search](http://proceedings.mlr.press/v139/bai21e/bai21e.pdf)<br>[2: Computing Graph Edit Distance via Neural Graph Matching](https://www.vldb.org/pvldb/vol16/p1817-cheng.pdf) | 参考CS224W 8th. reading list给了graph similarity computation (NP hard)问题的两种解决方法。自己实现GCN的结构，可以参考tutorial: [Graph Convolutional Network](https://github.com/guaiyoui/awesome-graph-analytics/blob/main/tutorials/tutorial_8.ipynb)  |
|6| Subgraph Matching and Counting |  [1: A Learned Sketch for Subgraph Counting](https://dl.acm.org/doi/pdf/10.1145/3448016.3457289)<br>[2: Neural Subgraph Counting with Wasserstein Estimator](https://dl.acm.org/doi/pdf/10.1145/3514221.3526163) | 参考CS224W 12nd. reading list给了subgraph counting (NP hard)问题的两篇文章(from sigmod 2021 and sigmod 2022)。subcon论文的仓库代码在 [Contrastive Learning on Graph](https://github.com/yzjiao/Subg-Con)  |



<p id="CohesiveSubgraphDiscovery"></p>

## 1: Cohesive Subgraph Discovery
Cohesive Subgraph Discovery是一种在图形数据中寻找具有高度内聚性的子图的问题。这篇survey：[A Survey on Machine Learning Solutions for Graph Pattern Extraction](https://arxiv.org/abs/2204.01057)（pay close attention to Ch2.6 **community search**）很好的讲清楚了我们工作的几篇baseline文章。

### 1.1 Subgraph-model-based Community Search
Subgraph-model-based community search model the community as various subgraph models, e.g., k-core, k-truss, and k connected component.

### 1.2 Metric-based Community Search
The metric-based community search methods aims to find a connected subgraph that contains the query nodes and has the largest metric, e.g., density, modularity....

| Conference | Paper  |  Material | Abstract | Highlights |
|---|---|---|---|---|
|SIGMOD2022|[DMCS: Density Modularity based Community Search](https://dl.acm.org/doi/abs/10.1145/3514221.3526137)|---| maximize the density modularity| Propose a new modulariity called density modularity to alleviate free-rider effect and resolution limit problem. |
|VLDB2015|[Robust local community detection: on free rider effect and its elimination](http://www.vldb.org/pvldb/vol8/p798-wu.pdf)|---| maximize the query biased density | Systematically study the many goodness functions, and provide detailed proof. |




### 1.3 Learning-based Community Search
Learning-based community search的方法，一般把问题model成node classification的任务。

| Conference | Paper  |  Material | Abstract | Highlights |
|---|---|---|---|---|
|VLDB2021|[ICS-GNN: lightweight interactive community search via graph neural network](https://dl.acm.org/doi/pdf/10.14778/3447689.3447704)|[[code]](https://github.com/guaiyoui/awesome-graph-analytics/blob/main/files/ics-gnn.zip)| Find a community in multiple iterations (i.e., hops)|---|
|VLDB2022|[Query Driven-Graph Neural Networks for Community Search](https://arxiv.org/abs/2104.03583)|[[code]](https://github.com/lizJyl/Codes-for-Peer-Review-of-VLDB-August-337)|QD-GNN and AQD-GNN for community search and attributed community search respectively|Take query into account. Study multiple CS-related settings.|
|ICDE2023|[Community Search: A Meta-Learning Approach](https://arxiv.org/abs/2201.00288)|---|CS using small data|---|
|ICDE2023|[COCLEP: Contrastive Learning-based Semi-Supervised Community Search](https://siqiangluo.com/docs/COCLEP__Contrastive_Learning_based_Semi_Supervised_Community_Search__camera_ready_.pdf)|[[code]](https://github.com/guaiyoui/awesome-graph-analytics/blob/main/files/COCLEP.zip)|论文针对目前基于深度学习的社区搜索模型依赖大量实际难以获取的标记数据进行训练的问题，提出了一种基于对比学习和数据分区的社区搜索方法(COCLEP)，只需极少量的标签即可实现高效且有效的社区查询任务，其基本原理是通过所提出的图神经网络和标签感知对比学习器来学习查询依赖的模型。此外，论文从理论上证明了可以利用最小割将COCLEP扩展用于大型数据集。|---|

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


<p id="AIGCLLM"></p>

## 3: AIGC-LLM

### 3.1 Survey of AIGC-LLM

- [**Augmented Language Models: a Survey](https://doi.org/10.48550/arXiv.2302.07842), Arxiv, 2023.02.15**
- [**A Survey for In-context Learning](https://doi.org/10.48550/arXiv.2301.00234), Arxiv, 2022.12.31**
- [**Reasoning with Language Model Prompting: A Survey](https://doi.org/10.48550/arXiv.2212.09597), Arxiv, 2022.12.19**
- [**Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing](https://doi.org/10.1145/3560815), Arxiv, 2021.07.28**
- [**Emergent Abilities of Large Language Models](https://doi.org/10.48550/arXiv.2206.07682), Arxiv, 2022.06.15**
- [**Towards Reasoning in Large Language Models: A Survey](https://doi.org/10.48550/arXiv.2212.10403), Arxiv, 2022.12.20**

### 3.2 Theory of AIGC-LLM

- **[A Mathematical Exploration of Why Language Models Help Solve Downstream Tasks](https://arxiv.org/abs/2010.03648), 2020.10.7**
- **[Why Do Pretrained Language Models Help in Downstream Tasks? An Analysis of Head and Prompt Tuning](https://arxiv.org/abs/2106.09226), 2021.6.17**

### 3.3 Prompt Learning

#### Prompt **Engineering Techniques**

- **[Automatic Prompt Augmentation and Selection with Chain-of-Thought from Labeled Data](https://doi.org/10.48550/arXiv.2302.12822)** （**2023.02.24**）
- **[Guiding Large Language Models via Directional Stimulus Prompting](https://doi.org/10.48550/arXiv.2302.11520)** （**2023.02.22**）
- **[Progressive Prompts: Continual Learning for Language Models](https://doi.org/10.48550/arXiv.2301.12314), 2023.01.29**
- **[Batch Prompting: Efficient Inference with Large Language Model APIs](https://doi.org/10.48550/arXiv.2301.08721)** （**2023.01.19**）
- **[One Embedder, Any Task: Instruction-Finetuned Text Embeddings](https://doi.org/10.48550/arXiv.2212.09741)** （**2022.12.19**）
- **[Successive Prompting for Decomposing Complex Questions](https://doi.org/10.48550/arXiv.2212.04092)** （**2022.12.08**）
- **[Promptagator: Few-shot Dense Retrieval From 8 Examples](https://doi.org/10.48550/arXiv.2209.11755)** （**2022.09.23**）
- **[Black-box Prompt Learning for Pre-trained Language Models](https://arxiv.org/abs/2201.08531)** （**2022.01.21**）
- **[Design Guidelines for Prompt Engineering Text-to-Image Generative Models](https://doi.org/10.1145/3491102.3501825)** （**2021.09.14**）
- **[Prompt Programming for Large Language Models: Beyond the Few-Shot Paradigm](https://doi.org/10.1145/3411763.3451760)** （**2021.02.15**）
- [**Making Pre-trained Language Models Better Few-shot Learners](https://doi.org/10.18653/v1/2021.acl-long.295), ACL, 2021.01.01**
- [**Eliciting Knowledge from Language Models Using Automatically Generated Prompts](https://doi.org/10.18653/v1/2020.emnlp-main.346), EMNLP, 2020.10.29**
- **[Automatically Identifying Words That Can Serve as Labels for Few-Shot Text Classification](https://doi.org/10.5282/UBM/EPUB.74034)** （**2020.10.26**）

#### In-context Learning

- **[Larger language models do in-context learning differently](https://doi.org/10.48550/arXiv.2303.03846)** （**2023.03.07**）
- **[Language Model Crossover: Variation through Few-Shot Prompting](https://doi.org/10.48550/arXiv.2302.12170)** （**2023.02.23**）
- **[How Does In-Context Learning Help Prompt Tuning?](https://doi.org/10.48550/arXiv.2302.11521)** （**2023.02.22**）
- **[Large Language Models Are Implicitly Topic Models: Explaining and Finding Good Demonstrations for In-Context Learning](https://doi.org/10.48550/arXiv.2301.11916)** （**2023.01.27**）
- **[Transformers as Algorithms: Generalization and Stability in In-context Learning](https://arxiv.org/abs/2301.07067)** （**2023.01.17**）
- **[OPT-IML: Scaling Language Model Instruction Meta Learning through the Lens of Generalization](https://doi.org/10.48550/arXiv.2212.12017), Meta,**（**2022.12.22**）
- [**Finetuned Language Models are Zero-Shot Learners](https://arxiv.org/abs/2109.01652),** ICLR, （2021.9.3）
    
    FLAN，多任务 instruction tuning
    
- [**Learning To Retrieve Prompts for In-Context Learning](https://arxiv.org/abs/2112.08633),** NAACL, （2022.12.16）
    
    Prompt 选择对模型效果有影响，之前的方法利用相似度的方式挑选合适的样本，本文提出用一个单独的模型对训练集中的每个样例进行评分，选取最合适的作为 demonstration
    

#### Parameter-Efficient Fine-Tuning (PEFT)

- [**LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS**](https://arxiv.org/pdf/2106.09685.pdf)
    
    **LoRA** 
    
- [**Prefix-Tuning: Optimizing Continuous Prompts for Generation**](https://aclanthology.org/2021.acl-long.353/)
    
    **Prefix Tuning** 
    
- [**GPT Understands, Too**](https://arxiv.org/pdf/2103.10385.pdf)
    
    **P-Tuning V1** 
    
- [**P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks**](https://arxiv.org/pdf/2110.07602.pdf)
    
    **P-Tuning V2** 
    
- [**The Power of Scale for Parameter-Efficient Prompt Tuning**](https://arxiv.org/pdf/2104.08691.pdf)
    
    **Prompt Tuning** 
    

#### Reasoning with Large Language Models

- [**Automatic Chain of Thought Prompting in Large Language Models**](https://arxiv.org/abs/2210.03493)
    
    Auto-CoT
    
- **[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)**
    
    Manual-CoT
    
- **[Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916), NeurIPS, 2022**
    
    Zero-CoT
    

#### Multimodal Prompt

- [**Multimodal Chain-of-Thought Reasoning in Language Models**](https://arxiv.org/pdf/2302.00923.pdf)

#### Evaluation & Reliability

#### Others

- RPT: Relational Pre-trained Transformer Is Almost All You Need towards Democratizing Data Preparation, VLDB, 2021
- Can Foundation Models Wrangle Your Data?, VLDB, 2023

### 3.4 Foundation Models


#### 3.4.1 Encoder-only Architecture

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) (2018.10.11)
- [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942) （2019.09.26）
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) （2019.07.26）
- [ERNIE 3.0 Titan: Exploring Larger-scale Knowledge Enhanced Pre-training for Language Understanding and Generation](https://arxiv.org/abs/2112.12731)  (2021.12.23)

#### 3.4.2 Decoder-only Architecture

- [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774) （2023.03.15）
- GPT-3 [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) （2020.05.28）
- [JURASSIC-1: TECHNICAL DETAILS AND EVALUATION](https://assets.website-files.com/60fd4503684b466578c0d307/61138924626a6981ee09caf6_jurassic_tech_paper.pdf) (2021.08)
- Gopher [Scaling Language Models: Methods, Analysis & Insights from Training Gopher](https://arxiv.org/abs/2112.11446) （2021.12.08）
- [LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) （2022.01.20）
- Chinchilla [Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556) （2022.03.29）
- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311.pdf) （2022.04.05）
- [BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/abs/2211.05100) (2022.11.09)
- [OPT-IML: Scaling Language Model Instruction Meta Learning through the Lens of Generalization](https://arxiv.org/abs/2212.12017) (2022.12.22)
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) （2023.02.27）

#### 3.4.3 Encoder-decoder Architecture

- [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) （2019.10.29）
- T5 [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) （2019.10.23）
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) （2021.01.11）

#### 3.4.4 Other
- [GLM-130B: An Open Bilingual Pre-trained Model](https://arxiv.org/abs/2210.02414)  (2022.10.05)


### 3.5 Related Repos

- [Awesome-LLM: a curated list of Large Language Model](https://github.com/Hannibal046/Awesome-LLM)  
- [Awesome resources for in-context learning and prompt engineering: Mastery of the LLMs such as ChatGPT, GPT-3, and FlanT5, with up-to-date and cutting-edge updates](https://github.com/EgoAlpha/prompt-in-context-learning) 
- [This repository contains a hand-curated resources for Prompt Engineering with a focus on Generative Pre-trained Transformer (GPT), ChatGPT, PaLM etc](https://github.com/promptslab/Awesome-Prompt-Engineering) 
- [A trend starts from "Chain of Thought Prompting Elicits Reasoning in Large Language Models"](https://github.com/Timothyxxx/Chain-of-ThoughtsPapers) 
- [Collection of papers and resources on Reasoning in Large Language Models, including Chain-of-Thought, Instruction-Tuning, and others.](https://github.com/atfortes/LLM-Reasoning-Papers) 


### 3.6 Datasets of LLM-AIGC

- [Alpaca dataset from Stanford, cleaned and curated](https://github.com/gururise/AlpacaDataCleaned)  
- [Alpaca is a dataset of 52,000 instructions and demonstrations generated by OpenAI's `text-davinci-003` engine. This instruction data can be used to conduct instruction-tuning for language models and make the language model follow instruction better.](https://huggingface.co/datasets/tatsu-lab/alpaca/tree/main/data) 


### 3.7 Tools for LLM-AIGC

- [Awesome-LLM: a curated list of Large Language Model](https://github.com/Hannibal046/Awesome-LLM)  
- [Awesome resources for in-context learning and prompt engineering: Mastery of the LLMs such as ChatGPT, GPT-3, and FlanT5, with up-to-date and cutting-edge updates](https://github.com/EgoAlpha/prompt-in-context-learning) 
- [This repository contains a hand-curated resources for Prompt Engineering with a focus on Generative Pre-trained Transformer (GPT), ChatGPT, PaLM etc](https://github.com/promptslab/Awesome-Prompt-Engineering) 
- [A trend starts from "Chain of Thought Prompting Elicits Reasoning in Large Language Models"](https://github.com/Timothyxxx/Chain-of-ThoughtsPapers) 
- [Collection of papers and resources on Reasoning in Large Language Models, including Chain-of-Thought, Instruction-Tuning, and others.](https://github.com/atfortes/LLM-Reasoning-Papers) 


#### Open-Source LLMs

- [Inference code for LLaMA models](https://github.com/facebookresearch/llama)
- [Code and documentation to train Stanford's Alpaca models, and generate the data.](https://github.com/tatsu-lab/stanford_alpaca)
- [Port of Facebook's LLaMA model in C/C++](https://github.com/ggerganov/llama.cpp)
- [Locally run an Instruction-Tuned Chat-Style LLM](https://github.com/antimatter15/alpaca.cpp)
- [Instruct-tune LLaMA on consumer hardware](https://github.com/tloen/alpaca-lora)
- [骆驼:A Chinese finetuned instruction LLaMA](https://github.com/LC1332/Chinese-alpaca-lora)
- [Alpaca-LoRA as Chatbot service](https://github.com/deep-diver/Alpaca-LoRA-Serve)
- his fine-tunes the [GPT-J 6B](https://huggingface.co/EleutherAI/gpt-j-6B) model on the [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) dataset using a Databricks notebook [The repo](https://github.com/databrickslabs/dolly)
- [ChatGLM-6B：开源双语对话语言模型 | An Open Bilingual Dialogue Language Model](https://github.com/THUDM/ChatGLM-6B)
- GPT-J 6B is a transformer model trained using Ben Wang's **[Mesh Transformer JAX](https://github.com/kingoflolz/mesh-transformer-jax/)**. "GPT-J" refers to the class of model, while "6B" represents the number of trainable parameters. [EleutherAI/gpt-j-6B · Hugging Face](https://huggingface.co/EleutherAI/gpt-j-6B)
    
- [一种平价的chatgpt实现方案, 基于ChatGLM-6B + LoRA](https://github.com/mymusise/ChatGLM-Tuning)
- [Open Academic Research on Improving LLaMA to SOTA LLM](https://github.com/AetherCortex/Llama-X)

#### Prompt Learning

- [An Open-Source Framework for Prompt-Learning](https://github.com/thunlp/OpenPrompt)
- [PEFT: State-of-the-art Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)

#### CoT

- [Benchmarking LLM reasoning performance w. chain-of-thought prompting](https://github.com/FranxYao/chain-of-thought-hub)

#### Development

- [Examples and guides for using the OpenAI API](https://github.com/openai/openai-cookbook)
- [A gradio web UI for running Large Language Models like GPT-J 6B, OPT, GALACTICA, LLaMA, and Pygmalion.](https://github.com/oobabooga/text-generation-webui)
- [GUI for ChatGPT API](https://github.com/GaiZhenbiao/ChuanhuChatGPT)
- [LlamaIndex (GPT Index) is a project that provides a central interface to connect your LLM's with external data.](https://github.com/jerryjliu/llama_index)
- [The ChatGPT Retrieval Plugin lets you easily search and find personal or work documents by asking questions in everyday language.](https://github.com/openai/chatgpt-retrieval-plugin)
- [Building applications with LLMs through composability](https://github.com/hwchase17/langchain)
- [Containers for machine learning](https://github.com/replicate/cog)
    
    

#### ChatBots

- [An open-source ChatGPT UI.](https://github.com/mckaywrigley/chatbot-ui) 
- [Create UIs for your machine learning model in Python in 3 minutes](https://github.com/gradio-app/gradio)
- [A web interface for chatting with Alpaca through llama.cpp. Fully dockerized, with an easy to use API.](https://github.com/nsarrazin/serge)
- [ChatLLaMA 📢 Open source implementation for LLaMA-based ChatGPT runnable in a single GPU. 15x faster training process than ChatGPT](https://github.com/juncongmoo/chatllama)
- [Locally running, hands-free ChatGPT](https://github.com/yakGPT/yakGPT)
- [An editor made for programming with AI](https://github.com/getcursor/cursor)
- [ChatGPT 学术优化](https://github.com/binary-husky/chatgpt_academic)
- [myGPTReader is a slack bot that can read any webpage, ebook, video(YouTube) or document and summarize it with chatGPT. It can also talk to you via voice using the content in the channel.](https://github.com/madawei2699/myGPTReader)
- [Use ChatGPT to summarize the arXiv papers.](https://github.com/kaixindelele/ChatPaper)
- [基于 ChatGPT API 的划词翻译浏览器插件和跨平台桌面端应用 - Browser extension and cross-platform desktop application for translation based on ChatGPT API.](https://github.com/yetone/openai-translator)
- [LLM Chain for answering questions from documents with citations](https://github.com/whitead/paper-qa)
- [Grounded search engine (i.e. with source reference) based on LLM / ChatGPT / OpenAI API. It supports web search, file content search etc.](https://github.com/michaelthwan/searchGPT)
- [An open-source LLM based research assistant that allows you to have a conversation with a research paper](https://github.com/mukulpatnaik/researchgpt)
- [A simple command-line interface tool that allows you to interact with ChatGPT from OpenAI.](https://github.com/p208p2002/heygpt)

<p id="GraphSimilarityComputation"></p>

## 4: Graph Similarity Computation

<p id="SubgraphMatching"></p>

## 5: Subgraph Matching and Counting

<p id="CardinalityEstimation"></p>

## 6: Cardinality Estimation

<p id="Graph4DB"></p>

## 7: Graph for DB and tabular data

Graphs are a valuable tool for representing connections between entities, while tabular or relational data is a convenient and user-friendly way to store information. Researchers frequently employ graphs to depict interdependencies among records, attributes, elements, and schemas within and across tables. It is worth noting that in contemporary usage, the term "tabular deep learning" is often used to refer to the application of deep learning techniques to relational data organized as records, while the term "database" is often reserved to refer specifically to the software and infrastructure used to manage and manipulate such data.


| Conference | Paper  |  Material | Abstract | Highlights |
|---|---|---|---|---|
|PODS2023|[Databases as Graphs: Predictive Queries for Declarative Machine Learning.](https://dl.acm.org/doi/abs/10.1145/3584372.3589939)|---| Using hypergraph to model the relationship behind the records| -- |
|---|[Enabling tabular deep learning when d ge n with an auxiliary knowledge graph](https://arxiv.org/abs/2306.04766)|---| Capture the relation between two attributes by KG | -- |
|CIKM22|[Local Contrastive Feature learning for Tabular Data](https://dl.acm.org/doi/10.1145/3511808.3557630)|---| Capture the relation between two attributes by maximum spanning tree | -- |
|NIPS22|[Learning enhanced representations for tabular data via neighborhood propagation](https://arxiv.org/abs/2206.06587)|---| --- | -- |
|dlpkdd2021|[TabGNN: Multiplex Graph Neural Network for Tabular Data Prediction](https://arxiv.org/abs/2108.09127)|---| --- | -- |






<p id="VectorDB"></p>

## 8: Vector Database

Similarity search at a very large scale.

| Conference | Paper  |  Material | Abstract | Highlights |
|---|---|---|---|---|
|SIGMOD2023|[Near-Duplicate Sequence Search at Scale for Large Language Model Memorization.](https://dl.acm.org/doi/abs/10.1145/3589324)|---| ---| -- |

Talk 1: [Vector Database for Large Language Models in Production (Sam Partee)](https://www.youtube.com/watch?v=9VgpXcfJYvw)

## 9: Differential Privacy
DP & Graph:

DP & ML:
https://github.com/JeffffffFu/Awesome-Differential-Privacy-and-Meachine-Learning


## 10: 其他
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
