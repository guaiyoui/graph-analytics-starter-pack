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