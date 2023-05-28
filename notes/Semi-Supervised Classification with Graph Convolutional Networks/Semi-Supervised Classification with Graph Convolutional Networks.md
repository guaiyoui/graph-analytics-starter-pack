# [ Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/pdf/1609.02907.pdf)(用于半监督节点分类的图卷积神经网络GCN)

## 介绍

本文考虑半监督节点分类问题。

一种传统的方式是通过平滑化方式来给没有标签的节点进行预测：

$$
\mathcal{L}=\mathcal{L}_0+\lambda \mathcal{L}_{\text {reg }}, \quad \text { with } \quad \mathcal{L}_{\text {reg }}=\sum_{i, j} A_{i j}\left\|f\left(X_i\right)-f\left(X_j\right)\right\|^2=f(X)^{\top} \Delta f(X)
$$

其中$\mathcal{L}$是最终的标签信息（TODO：笔者看了参考文献，他们大概就是在表达这个意思）。$A \in \mathbb{R}^{N \times N}$是邻接矩阵；D是对角矩阵且$D_{i i}=\sum_j A_{i j}$，也就是$D_{ii}$是第i个节点的度数；$X$是节点的属性矩阵，$X\in\mathbb{R}^{N\times C}$；$\Delta = D - A$ 是拉普拉斯算子。就反正这一顿操作下来，没有标签的节点也可以硬给他加上标签，这未尝不是一种半监督学习的方法。

在本文中，不乱给节点加标签，而是直接使用一个神经网络$f(X,A)$对标签$\mathcal{L_0}$进行预测。

## 图卷积网络的卷积方法

节点包括属性$X$和连接关系$A$两种信息，通过结合图结构信息和节点属性可以学习有效的节点表征$H\in\mathbb{R}^{N\times F}$。每一层的节点表征通过以下规则进行更新:

$$
H^{(l+1)}=\sigma\left(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)}\right)
$$

其中，$H^{(l+1)}$是第l+1层节点表征，另外说明$H^{(0)}=X$,也就是第0层节点表征就是节点的属性。$\tilde{A}=A+I_N$是邻接矩阵+单位矩阵，如此$\tilde{A}$第i行就包括了所有邻居和自己的信息。$\tilde{D}$对应$D$,依然是一个对角矩阵$\tilde{D_{ii}}=\sum_j\tilde{A_{ij}}$,表达了节点$i$的度数。$\sigma(\cdot)$是一个激活函数，可以是ReLU或者Tanh。$W^{(l)}\in \mathbb{R}^{F\times F'}$是一个分层线性变换矩阵，是一个参数矩阵,F和F'分别是第l层和第l+1层的表征数量。

以上公式继续推导，就会发现$H^{(l)}$的第i个节点可以进行如下表达：

$$
\begin{gathered}
H_i^{(l)}=\sigma\left(\sum_{j \in\{N(i) \cup i\}} \frac{\tilde{A}_{i j}}{\sqrt{\tilde{D}_{i i} \tilde{D}_{j j}}} H_j^{(l-1)} W^{(l)}\right) \\
H_i^{(l)}=\sigma\left(\sum_{j \in N(i)} \frac{A_{i j}}{\sqrt{\tilde{D}_{i i} \tilde{D}_{j j}}} H_j^{(l-1)} W^{(l)}+\frac{1}{\tilde{D}_i} H_i^{(l-1)} W^{(l)}\right)
\end{gathered}
$$

其中$N(i)$是i节点的邻居节点。上式可以很直观的看出，$\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}$是在对$A_{ij}$做归一化，用这种方式在度量哪些结点和我这个节点关系“更近”，近的节点获得更高的权重。这里归一化的作用更详细的说明可以参考[一篇知乎上的回答](https://www.zhihu.com/question/426784258)。然后利用加权平均的方法处理上一层的输入信号。最后进行一个线性+非线性变换获得新的输出信号。

综上所述，这种方法其实是利用节点连接关系或者说远近进行属性的聚合，如果没有属性可以认为所有节点都有一个值为1的属性，此时也可以聚合节点连接关系本身。利用新聚合的属性作为节点的“特征”，也就是节点表征，就可以进行预测。

### 与谱图卷积的联系

图上的谱卷积可以定义如下：先不假设像之前那样输入一个矩阵 $H\in\mathbb{R}^{N\times F}$，而是输入一个节点信号$x\in \mathbb{R}^N$,带有傅里叶域上的卷积滤波器$g_\theta=\operatorname{diag}(\theta)$，其中$\theta \in \mathbb{R}^N$是滤波器的参数，一个普通的谱卷积可以如下表示，其中 `=`左边代表这是用$g_\theta$对x进行卷积, 右边是具体的卷积公式。

$$
g_\theta \star x=U g_\theta U^{\top} x
$$

其中U是归一化的拉普拉斯矩阵L的特征向量矩阵，也就是满足：

$$
L=I_N-D^{-\frac{1}{2}} A D^{-\frac{1}{2}}=U \Lambda U^{\top}
$$

其中$\Lambda$是特征值的对角矩阵，观察$U\Lambda U^{\top}$和$Ug_{\theta}U^{\top}$,并且$g_\theta$和$\Lambda$都是对角矩阵，也就可以认为$g_\theta$以$\Lambda$为自变量的一个函数$g_\theta(\Lambda)$。

直接计算卷积公式右边则复杂度高达$O(N^2)$,如果x是一个矩阵，那么就更难以接受；另外，计算L的特征值分解也有比较高的计算代价。因此应该进行近似计算。这里用切比雪夫多项式$T_k(x)$近似$g_\theta(\Lambda)$:

$$
g_{\theta^{\prime}}(\Lambda) \approx \sum_{k=0}^K \theta_k^{\prime} T_k(\tilde{\Lambda})
$$

其中, $\tilde{\Lambda}=\frac{2}{\lambda_{\max }} \Lambda-I_N$,$\lambda_{\max }$是L的最大特征值，有性质：标准化拉普拉斯矩阵的的特征值范围在\[0,2\]之间。$\theta^{\prime} \in \mathbb{R}^K$是切比雪夫系数向量。$T_k(x)= 2 x T_{k-1}(x)-T_{k-2}(x)$，$T_0(x) = 1$ 且 $T_1(x) = x$。

上面那个近似公式其实是选择了一组多项式基底{$T_0(x)$,$T_1(x)$,..., $T_k(x)$ }近似原来的$g_\theta$，因此参数量也就从原来的n个变成了k个。

另外定义$\tilde{L}=\frac{2}{\lambda_{\max }} L-I_N$。因为 $(U\Lambda U^{\top})^k= U\Lambda^k U^{\top}$,所以

$$
T_k(\tilde{L})=T_k(\frac{2}{\lambda_{\max }} L-I_N )\\ =T_k( \frac{2}{\lambda_{\max }} U \Lambda U^{\top}-UU^{\top}) \\ = U T_k(\tilde{\Lambda}) U^{\top}
$$

所以

$$
g_{\theta^{\prime}} \star x \approx \sum_{k=0}^K \theta_k^{\prime} T_k(\tilde{L}) x,
$$

由此可以看出，

* 每个节点只取决于K阶邻域内的信息。因为切比雪夫的递推公式。
* 新式子总体复杂度O(N)（TODO 我想了半天都没看出来）。
* 而且新公式没有对L进行分解。

作者认为应该把卷积数限制为K=1，并且用多个神经网络隐藏层来覆盖图上的丰富的卷积滤波函数类。作者希望通过这种方法缓解过拟合问题，因为大多数图节点度数分布不均衡(高变异性)，对于节点度分布具有高变异性的图来说，由于某些节点的度数非常高，因此在这些节点的邻域中可能存在一些过于复杂的结构，这些结构可能会导致机器学习算法在训练时出现过拟合的问题。

作者还简化把$\lambda_{\max } \approx 2$(TODO:怎么是约等于？这不就相当于设置一个超参数了吗？直接设置成等于不行吗？)。综合上面两个简化，新的卷积滤波器就是：

$$
g_{\theta^{\prime}} \star x \approx \theta_0^{\prime} x+\theta_1^{\prime}\left(L-I_N\right) x=\theta_0^{\prime} x-\theta_1^{\prime} D^{-\frac{1}{2}} A D^{-\frac{1}{2}} x,
$$

作者又一次简化，令$\theta=\theta_0^{\prime}=-\theta_1^{\prime}$,则

$$
g_\theta \star x \approx \theta\left(I_N+D^{-\frac{1}{2}} A D^{-\frac{1}{2}}\right) x
$$

此时，使用一个trick(TODO具体怎么推我看不出来)：$I_N+D^{-\frac{1}{2}} A D^{-\frac{1}{2}} \rightarrow \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}$,其中$\tilde{A}=A+I_N$ and $\tilde{D}_{i i}=\sum_j \tilde{A}_{i j}$

以上是对于输入如果是向量$x\in\mathbb{R}^{N}$的分析，那么如果说输入是矩阵$H\in\mathbb{R}^{N\times F}$，则：

$$
Z=\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H \Theta
$$

其中$\Theta \in \mathbb{R}^{F \times F^{\prime}}$是参数矩阵，这里计算复杂度是$\mathcal{O}(N F C)$，这就和最开始的公式一样，只是缺一个激活函数。

综上所述，本文提出的方法是谱图卷积的简化和近似。

## 给闲人看的术语

* 谱图卷积：图或者图神经网络中，“谱”就是指图拉普拉斯算子L的特征值分解。可以认为图拉普拉斯算子L是一种特殊方式归一化的邻接矩阵A，而特征值分解是一种找到构成图基本正交分量的方法。谱图卷积一个介绍：https://towardsdatascience.com/spectral-graph-convolution-explained-and-implemented-step-by-step-2e495b57f801
  * 图的谱分析是一个大领域，虽然现在用的不太多，有耶鲁大学的课程讲了这个，课程讲义在此列出：http://www.cs.yale.edu/homes/spielman/561/
* 图拉普拉斯正则化: 是一种在图上进行正则化的方法。在这种方法中，我们使用图的拉普拉斯矩阵来对模型进行正则化。拉普拉斯矩阵是一个对称矩阵，它描述了图的拓扑结构。通过对拉普拉斯矩阵进行特征分解，我们可以得到一组正交的基函数，这些基函数可以用于对图上的信号进行变换。Graph Laplacian regularization可以用于约束模型参数，以使其在图上的变化尽可能平滑。这种正则化方法在图像和信号处理领域得到了广泛应用。https://zhuanlan.zhihu.com/p/362416124 。 这里先不管了，就是一种传统方法。
* 图傅里叶变换：
* 切比雪夫多项式：

## 贡献和优势

* 作者们引入了一种简单且良好的逐层传播规则，用于直接在图上操作的神经网络模型，并展示了这种方法其实是谱图卷积的近似。
* 作者们展示了这种基于图的神经网络模型可以用于快速和可扩展的半监督节点分类。在多个数据集上的实验证明，与半监督学习的最新方法相比，该模型在分类精度和效率方面表现良好。
* 这个神经网络的参数$W\in \mathbb{R}^{F\times F'}$和图本身的节点数量无关，本文的网络只用了节点的若干阶邻域信息，普通的网络应该也就是用5阶邻域信息封顶，应该会有比较好的泛化性。

## 劣势和讨论

* 这种聚合方法过分粗暴。笔者记得有一个很著名的问题是怎么把不同人的声音完美的区分开，目前似乎没有用线性方法可以很好解决的，非线性方法似乎也没有解决的很好(要不然降噪耳机早就能做很好了)。而这样把属性信号混合在一起之后，很难再用线性的方法把这些信号再分开，也就是说在聚合过程中丢失了很多非常重要的信息。

这种半监督的方式其实效果存疑，我曾经帮忙复现过本科一个学长发表的顶会论文，使用推特社交网络数据在图神经网络上(我记得就是这个类似的GNN)给账号分类，分成机器人账号和真人账号。当时原始数据大概10万条，最后有1000条(或者10000条？)数据用来测试，从训练集到测试集都是均衡样本，原始数据大概是标注了10000条，显然是半监督学习。他们顶会论文用了10万条数据进行训练，最后测试集上精度0.8+，我只用了有标签的1万条数据训练，最后测试集上精度0.9+。。。。反正现在对这个方法印象不太好。

## 不会的地方

切比雪夫这里到底是怎么近似的

在线学习的可能性，可不可以在新节点加入后只学习新节点可以得到比较好的结果？

## 参考文献

[图神经网络：基础、前沿与应用](https://graph-neural-networks.github.io/index.html) ，参考第四章

[康奈尔大学，谱图理论](https://people.orie.cornell.edu/dpw/orie6334/Fall2016/lecture7.pdf)：主要参考了标准化拉普拉斯方面的理论，讲述了为什么特征值取值范围在\[0,2\]之间。
