# ROTATE: KNOWLEDGEGRAPHEMBEDDING BY RELA-TIONALROTATION  INCOMPLEXSPACE

开会要问的：那些横向相关的论文有必要复现吗？还是读读代码就够了？

给后来者：本笔记用来帮助同学们了解DeepWalk论文，**遵循[Mozilla Public License 2.0](https://www.mozilla.org/en-US/MPL/2.0/)协议** ，本部分的 `作者`和 `修改人`信息请保留，后面任何内容请后来者按照认为最合适的方式排布、修改。

* 笔记作者：赵敬业
* 修改人：

## 研究问题

本文主要是研究的用Knowledge graph embedding(KGE)来预测缺失连接的问题。本文重点在于提出一个新的KGE算法。

知识图谱的连接可以表示为(h,r,t),其中h和t属于实体空间$h,t\in \mathcal{E}$，r属于关系空间$r\in\mathcal{R}$。需要设定$f_r(h,t)$函数作为score function来评判h,t在r关系下是否足够近。那么一个优秀的知识图谱应该能让有关系r的h和t在$f_r(h,t)$上足够大，让没有关系r的h',t在$f_r(h',t)$上足够小。同时，最好能够预测缺失的连接。

## 贡献

提出了复数空间表示知识图谱的实体和关系的方法，并且解决了已有的知识图谱嵌入算法不能完全表示对称/反对称/可逆/组合等关系模式的问题。

## 知识图谱的关系模式

知识图谱三元组存在一些特殊的关系模式，包括对称/反对称/可逆和组合等：

* 对称关系(如结婚关系)： r(x,y) -> r(y,x)
* 反对称关系(如亲子关系)：r(x,y) -> -r(y,x)
* 可逆关系(如上位词和下位词（水果和苹果），这是两个关系互相可逆，不是一个关系)：$r_2(x,y)$ -> $r_1(y,x)$
* 组合关系(妈妈的丈夫=父亲)：$r_2(x,y)\wedge r_3(y,x)$ -> $r_1(x,z)$

很多现有的方法试图对上面的几种关系进行建模表示，比如有SE、TransE、TranX、DistMulti、ComlEx等方法。

## 不同KGE算法的比较

|  Model  |         Score Function（打分函数）<br />这一项越高，代表h和t在关系r上很近         | Symmetry<br />（对称） | Antisymmetry<br />（反对称） | Inversion<br />（可逆） | Composition<br />（组合） |
| :------: | :--------------------------------------------------------------------------------: | :--------------------: | :--------------------------: | :---------------------: | :-----------------------: |
|    SE    |      $-||\boldsymbol{W}_{r, 1} \mathbf{h}-\mathbf{W}_{r, 2} \mathbf{t}|| $      |   $\boldsymbol{X}$   |      $\boldsymbol{X}$      |   $\boldsymbol{X}$   |    $\boldsymbol{X}$    |
|  TransE  |                     $-||\mathbf{h}+\mathbf{r}-\mathbf{t}|| $                     |  $\boldsymbol{X} $  |       $\checkmark $       |     $\checkmark$     |      $\checkmark$      |
|  TransX  |    $-|| g_{r, 1}(\mathbf{h})+\mathbf{r}-g_{r, 2} \left(\mathbf{t}\right) || $    |     $\checkmark$     |        $\checkmark$        |   $\boldsymbol{X}$   |    $\boldsymbol{X}$    |
| DistMult |                $\langle\mathbf{h}, \mathbf{r}, \mathbf{t}\rangle$                |    $\checkmark $    |      $\boldsymbol{X}$      |   $\boldsymbol{X}$   |    $\boldsymbol{X}$    |
| ComplEx | $\operatorname{Re}(\langle\mathbf{h}, \mathbf{r}, \overline{\mathbf{t}}\rangle $ |     $\checkmark$     |        $\checkmark$        |     $\checkmark$     |    $\boldsymbol{X}$    |
|  RotatE  |                  $-||\mathbf{h} \circ \mathbf{r}-\mathbf{t}|| $                  |     $\checkmark$     |        $\checkmark$        |     $\checkmark$     |      $\checkmark$      |

一看就知道RotatE表现很好。

TransE可以建模除了对称关系之外的其他所有关系。

## RotatE的核心想法

本文方法的动机来源于欧拉公式：

$$
e^{i \Theta}=\cos \Theta+i \sin \Theta
$$

这说明酉(unitary)复数$r=e^{i\theta}$可以认为是复数空间中的旋转操作。

我们可以将h和t表示成k维的复数向量，并且让关系r的模长为1，且$r=\left(r_1, r_2, \ldots, r_k\right), r_i=e^{i \theta_{r, i}}$。那么定义如果(h,r,t)满足下面的公式，就说h和t有r关系：

$$
\mathbf{t}=\mathbf{h} \circ \mathbf{r}, \quad \text { where }\left|r_i\right|=1,
$$

其中$\circ$是阿玛达乘积，或称逐元素相乘。即$t_i=h_i r_i$。显然不太可能所有满足有这种r关系的h和t都完全满足这个等式，那么可以要求有r关系的h和t应该在进行r操作后距离足够近。定义如下的距离函数，之后机器学习的目标就呼之欲出了：

$$
d_r(h, t)=\|h \circ r-t\|
$$

另外再讨论之前图表中的TransE操作。TransE操作如下图(a)所示，就是一个普通的平移操作，因此这种实际上除了r=0这种特例，没办法保证找到对称关系，即任何不是0向量的r都不满足h+r=t且t+r=h。而RotatE采用复数向量空间来表示知识图谱中的实体，并且将关系看成是复数空间中的实体的旋转，这样如果$r_i=\pm 1$，则r关系就能满足对称性要求。图(c)表示了满足一种对称关系的r=(-1,-1)

![image-20210731092705213](https://zhang-each.github.io/2021/08/08/reading8/image-20210731092705213.png)

## 优化算法：自对抗负采样和损失函数

RotatE模型依然采用基于负采样的损失函数来优化模型参数，其形式如下：

$$
minimize \quad L=-\log \sigma\left(\gamma-d_r(\mathbf{h}, \mathbf{t})\right)-\sum_{i=1}^n \frac{1}{k} \log \sigma\left(d_r\left(\mathbf{h}_i^{\prime}, \mathbf{t}_i^{\prime}\right)-\gamma\right)
$$

$\sigma$是sigmoid函数，这里的k是模型选取的嵌入维度，同时每次负采样我们使用n个负样本,$\gamma$是一个固定的参数,保证。

负采样算法也是来源于近似softmax函数的需要。其他近似方法是试图估计log-softmax函数，负采样方法放弃了估计log-softmax函数，而是直接使用新的目标函数来达到同样的最终最大化log-softmax函数的目的。这里可以简单理解：等式右边第一项$-\log \sigma\left(\gamma-d_r(\mathbf{h}, \mathbf{t})\right)$中的$d_r(h,t)$取样于真实关系生成器，是正样本；等式右边第二项$-\sum_{i=1}^n \frac{1}{k} \log \sigma\left(d_r\left(\mathbf{h}_i^{\prime}, \mathbf{t}_i^{\prime}\right)-\gamma\right)$中的$d_r(h_i^\prime,h_i^\prime)$取样于一个另外的均匀分布，是负样本。讲到这里目标函数的定义在这里就很明显了。另外softmax的各种近似方法，这篇[博客](https://www.ruder.io/word-embeddings-softmax/)笔者认为写的非常好。

作者认为上面那种从均匀随机分布中抽取样本的方法非常低效，因为这样抽出来的很多负样本显然是负样本，不用机器学习都知道，这就导致很低效。因此，本文提出自对抗负采样[Self-adversarial negative sampling]，这种方法也可以应用到其他知识图谱上。具体而言，自对抗负采样使用目前已经学到的机器学习模型来计算负样本的分布，也就是从下面的分布中抽取负样本（TODO 下面的公式似乎是写错了）：

$$
p\left(h_j^{\prime}, r, t_j^{\prime} \mid\left\{\left(h_i, r_i, t_i\right)\right\}\right)=\frac{\exp (\alpha f_r\left(\mathbf{h}_j^{\prime}, \mathbf{t}_j^{\prime}\right))}{\sum_i \exp \alpha f_r\left(\mathbf{h}_i^{\prime}, \mathbf{t}_i^{\prime}\right)}
$$

这里的$\alpha$是一个超参数，可以表示采样的temperature（TODO？？），同时如果直接用这个方法去一个个找负采样的样本，会让计算变的很复杂，我们可以直接将这个概率作为负采样的权重，将损失函数变成下面的形式：

$$
L=-\log \sigma\left(\gamma-d_r(\mathbf{h}, \mathbf{t})\right)-\sum_{i=1}^n p\left(h_i^{\prime}, r, t_i^{\prime}\right) \log \sigma\left(d_r\left(\mathbf{h}_i^{\prime}, \mathbf{t}_i^{\prime}\right)-\gamma\right)
$$

## 优势

* 可以扩展到大型知识图，因为它在时间和内存上都保持线性。
* 模型达到了SOTA（State-of-the-Art，在特定任务上已达到了最高水平的性能表现）

## 参考

[那颗名为现在的星](https://zhang-each.github.io/2021/08/08/reading8/)

[论文翻译](https://zhuanlan.zhihu.com/p/426764321)

[知识图谱图嵌入](https://towardsdatascience.com/introduction-to-knowledge-graph-embedding-with-dgl-ke-77ace6fb60ef)
