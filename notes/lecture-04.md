# Lecture4: Attention Alternatives&MoE

## Attention Alternatives

目前存在的问题是，现代大模型一般需要支持比较长的上下文窗口，但是随着序列最长，`Attention` 部分的时间复杂度可以达到 $O(n^2)$ ，所以我们需要考虑使用新的注意力架构来降低时间复杂度

<div class="feishu-grid" style="grid-template-columns: 0.5fr 0.5fr;">

<div class="feishu-column">

![大模型上下文增长(目前大部分模型 1M)](assets/images/lecture-04-img-01.png)

</div>

<div class="feishu-column">

![序列长度和计算耗时增长关系](assets/images/lecture-04-img-02.png)

</div>

</div>

解决上述问题的思路主要分为两种:

1. 注意力本身架构需要修改: 例如 `Local Attention` 和 `Global Attention` 结合
2. 从系统工程的角度考虑问题，例如 `FlashAttention` 算法

<div class="feishu-grid" style="grid-template-columns: 0.5fr 0.5fr;">

<div class="feishu-column">

![图片展示了Transformer及其变种的注意力机制。上半部分从左至右依次为Transformer、Sparse Transformer（strided）和Sparse Transformer（fixed）的注意力图，显示了不同模型在不同位置的注意力分布。下半部分是Sparse Transformer的架构图，标注了Local Attention和Global Attention，说明其结合了局部和全局注意力机制，且Global Attention层的维度是Local Attention层的两倍，且其查询数量是Key数量的16倍，强调了其在系统工程角度解决注意力时间复杂度问题的思路。](assets/images/lecture-04-img-03.png)

</div>

<div class="feishu-column">

![图片](assets/images/lecture-04-img-04.png)

</div>

</div>

### Linear Attention

首先我们考虑普通的注意力操作，其中存在 $Q \in R^{n \times d_k}、K \in R^{n \times d_k}、V \in R^{n \times d_v}$，普通注意力运算为:

$$Attn(Q,K,V) = \rho(QK^T)V$$

如果其中 $\rho$ 是线性函数，那么我们可以做如下变换:

$$(QK^T)V = Q(K^TV)$$

我们这里可以比较两种运算，下面描述的单位为乘积运算(1 `MAC` 相当于一次乘加运算)

- $(QK^T)V$: 第一次运算相当于 $[n, d_k] \times [d_k, n] \rightarrow [n, n]$ 相当于输出 $n^2$ 个元素，每一个元素进行 $d_k$ 次 `MAC` 运算，同时第二次运算相当于 $[n,n] \times [n, d_v] \rightarrow [n, d_v] $ 相当于输出 $nd_v$ 个元素，每一个元素进行 $n$次 `MAC` 运算，所以总共进行 $n^2d_v$次运算，所以相当于总共进行 $n^2 d_k + n^2 d_v$ 次 `MAC` 运算
- $Q(K^TV)$: 第一次运算相当于 $[d_k, n] \times [n, d_v] \rightarrow [d_k, d_v]$ 相当于输出 $d_kd_v$个元素，每一个元素进行 $n$ 次运算，第二次运算相当于 $[n, d_k] \times [d_k, d_v] \rightarrow [n, d_v]$ 相当于输出 $nd_v$ 个元素，每一个元素进行 $d_k$ 次运算，所以进行 $nd_kd_v$次运算，所以总共进行 $2nd_kd_v$ 次运算

这里 $n$ 序列长度一般是一个比较大的值，但是向量维度可能不是一个比较大的值，所以第二种情况的计算次数可能远远小于第一种形式

另外，我们可以考虑推理情况下对于上述线性注意力机制的形式重写，例如对于第 $t$个位置(这里序列长度为 $n$)，那么对应位置的输出值计算如下:

$$y_t = \sum_{j \leq t} (q_t^Tk_j) v_j$$

同时利用卷积交换律我们可以得到:

$$y_t = \sum_{j \leq t} q_t^T (k_j v_j^T) = q_t^T (\sum_{j \leq t} k_j v_j^T) = q_t^T S_t$$

由此我们就可以定义状态 $S_t$，同时根据 $S_t = \sum_{j \leq t} k_j v_j^T$ 我们可以得到状态递推公式:

$$S_t = S_{t-1} + k_t v_t^T$$

所以我们可以得到如果可以使用这种线性注意力机制，那么我们在推理的时候就可以得到：

$$S_t = S_{t-1} + k_tv_t^T, y_t = q_t^T S_t$$

> **上述形式类似于 RNN，利于推理但是不利于训练，所以利用这种形式我们可以:**
> 
> - **训练时使用稠密形式，显著减少 `FLOPs`**
> - **推理时使用递推形式(串行线性形式)，推理甚至不需要随着序列长度增长的 `KV Cache`，每一个状态在每一个 `token` 上递推更新一次即可**

`MiniMax M1` 使用 7:1(7 `Linear`、1 `Full`) 的混合线性注意力架构，同时在不同任务上有不错的表现，`FLOPs` 机会随着上个下文窗口线性增长

![这张图片展示了MiniMax M1相关的信息，包含三个核心部分：左侧是不同模型在各类推理任务上的准确率对比柱状图，中间是各模型FLOPs随生成长度变化的折线图，右侧是混合线性注意力架构的结构示意图。图片底部的表格列出了Softmax与Hybrid-lightning两类架构在多个基准任务上的性能指标，底部文字明确说明MiniMax M1及minimax-text-01采用7:1的混合线性注意力架构，即7个线性注意力模块搭配1个全注意力模块。该图直观呈现了文档中提及的MiniMax M1架构的结构特点与性能表现，对应文档中关于混合线性注意力的相关内容。](assets/images/lecture-04-img-05.png)

#### From Linear Attention to Mamba2

- `Linear Attention`:

$$S_t = S_{t-1} + k_t v_t^T \ and \ y_t = q_t^T S_t$$

- `Mamba-2`:

$$S_t = \gamma_t S_{t-1} + k_tv_t^T \ and \ y_t = q_t^T S_t + v_t^T D \ with \ \gamma_t = f(x_t)$$

相较于基本的 `Linear Attention`, `Mamba-2` 在状态更新参数上加上了门控机制，同时在输出上加上了一个来自于当前位置的直接输出项(有点类似于残差项)，提升了传统 `Linear Attention` 的表达能力，同时公式中的 $\gamma_t = f(x_t)$ 不依赖于状态，可以进行并行计算

同时我们可以对于 `Mamba-2` 串行公式进行变形:

$$S_t = \gamma_t S_{t-1} + k_tv_t^T \\ = \gamma_t \gamma_{t-1} S_{t-2} + \gamma_t k_{t-1} v_{t-1}^T + B_t x_t^T\\ = \sum_{j=1}^T (\prod_{i = j + 1}^t a_i) k_j v_j^T $$

同时最后的输出为:

$$y_t = q_t^T S_t + v_t^T D \\ = \sum_{j = 1}^t (\prod_{i=j+1}^t a_i) (q_t^T k_j)v_j + v_t^T D$$

经典的 `SSM` 运算公式一般可以写成如下形式:

$$Y = (L \odot CB^T) X$$

同时 `Mamba-2` 中的 `SSM` 定义如下:

![图片展示了`Mamba-2`中`SSM`运算的符号、形状及含义。其中，$x_t$为输入，形状为$\\\\mathbb R^P$；$h_t$为状态，形状为$\\\\mathbb R^{N\\\\times P}$；$a_t$为标量衰减（输入依赖），形状为$\\\\mathbb R$；$B_t$为输入矩阵（向量），形状为$\\\\mathbb R^N$；$C_t$为输出矩阵（向量），形状为$\\\\mathbb R^N$。递推公式为$h_t = a_t h_{t-1} + B_t x_t^\\\\top$，$y_t = C_t^\\\\top h_t$。该图与上文介绍的`Mamba-2`中`SSM`定义相关，是对`SSM`运算具体符号和计算方式的说明。](assets/images/lecture-04-img-06.png)

同时对应的掩码衰减 $L \in \mathbb R^{T \times T}$:

$$L_{tj} = \begin {cases} \prod_{i=j+1}^t a_i, j \leq t \\0, j \gt t\end {cases}$$

如果加上残差项，那么对应的 `SSM` 操作可以写成:

$$Y = (L \odot CB^T)X + X diag(D)$$

#### Gated delta net

`Gated delta net` 门控机制和 `Linear Attention`、`Mamba-2` 存在一些区别，对应的状态递推公式为:

$$S_t = \gamma_t (I - \beta_t k_t k_t^T) S_{t-1} + \beta_tk_tv_t^T \\y_t = q_t^T S_t \ with \ \gamma_t = f(x_t), \beta_t = f(x_t)$$

递推公式中，门控机制中的 $\gamma_t $用于表示过去状态的权重，同时我们考虑 $\beta_t $ 的作用，当 $\beta_t = 0$ 的时候，表示几乎不接受当前输入

其中的 $I - \beta_t k_t k_t^T $ 我觉得可以这样理解，注意到 $\beta_t = f(x_t)$ 和原始输入 $x_t $ 有关，同时注意到 $k_tk_t^T $ 是外积，作用是正交投影，考虑使用该变换 $(k_tk_t^T)u = k_t(k_t^Tu)$ 我们可以发现这个集合变换的意义其实是首先求解 $u$ 在 $k_t$ 方向上的投影，然后放回到 $k_t $方向上得到向量，所以利用这一个性质，我们可以得到:

- 与 $k_t$ 正交的方向效果是几乎为 0,完全不变
- 和 $k_t$ 相近的方向相当于缩放为原来的 $I - \beta_t$

所以相当于任意和 $k_t $ 正交的量，存在 $S_t u = S_{t-1}u$，所以我们可以发现 $\beta_t $ 的准确含义应该是同时控制擦除强度( $k_t$ 方向保留 $1 - \beta_t$，同时新值占比 $\beta_t$)

> 我觉得比较通俗的解释是: **如果当前输入 $x_t$ 和 $k_t$ 几乎正交，那么相当于 $S_{t-1}$ 过去的状态几乎不衰减，才出强度低，同时如果当前输入 $x_t$ 和 $k_t$ 相关性高的话，相当于次数的输出需要更多当前信息，那么相当于当前对于过去状态的衰减要尽可能大，同时对于当前输入的权重尽可能多(两个作用，写入&擦除过去权重)**

另外一种直观解释:

> 实际上我们的目标就是利用当前状态和值可以确定对应的输入: $S_t k_t = v_t$，所以我们可以使用梯度下降的方式求解

![图片展示了推导B：一步梯度下降（最直观）的内容。将用状态在key $k_t$处重构 $v_t$看成最小二乘损失，给出损失函数表达式。对S求梯度，得到梯度表达式。做一步梯度下降，步长取$\\\\beta_t/2$，给出状态更新公式。同样得到另一个状态更新公式。最后说明 $k_t k_t^\\\\top$是“在当前key上的重构误差”的梯度方向，DeltaNet的“delta”指的就是这个误差修正项。该图片与上下文紧密相关，是对上文提到的梯度下降求解方法的数学推导说明。](assets/images/lecture-04-img-07.png)

`Qwen3.5` 使用 3:1(GDN/Attenton hybrids Arch)，极大程度上提升了长上下文下的吞吐量和在各种任务上的性能，可见这种混合架构还是具有优势的:

![图片展示了Qwen 3.5 / Qwen Next模型架构及性能。左侧是模型架构图，包含Gated Delta Net、Linear Attention等组件。中间是不同模型在不同任务上的性能对比柱状图，如Speech2Text、Text2Text等任务的吞吐量、准确率等指标。右侧是Decode Throughput vs Context Length（Normalized）折线图，展示了不同模型在不同上下文长度下的解码吞吐量。底部文字总结Qwen 3.5是3:1 GDN / Attention hybrids模型，性能合理且具备良好推理特性。](assets/images/lecture-04-img-08.png)

### DeepSeek Sparse Attention

`DeepSeek V3.2` 中提出 `DSA`,核心思想是对于一段序列，不需要做全局的 `Attention` 操作，而是根据一定规则选出序列的一部分子集，并且在这个子集上做全量的 `Attention` 操作，同时子集的选择依据可以是，首先利用一个模型输入 $Q、K$ 并且的到一个激活值，然后选取 `top-k` 激活值做 `Full Attention` 运算即可，同时这一个模型比较轻量:

$$I_{t,s} = \sum_{j=1}^{H^I} w_{t,j}^I \cdot ReLU(q_{t,j}^I \cdot k_s^t)$$

$$u_l = Attn(h_t, \{c_s | I_{t,s} \in Top-k(I_t')\})$$

> **这里我其实有一个疑问：这里所引起应该也会做全量的内积，不是也会消耗很多资源吗？ 答案是 索引器其实不需要那么精确，可以对于 q、k进行降维等方案做的比较轻量化，同时也可以使用低精度的数据类型，例如 FP8等，同时训练过程可以使用已经训练好的 Teacher Model，优化目标为完整的全注意力机制**

同时 `DeepSeek V3.2、GLM5` 中都使用了 `DSA` 并且取得了不错的效果，资源消耗减少，同时模型表达能力和相关的测评指标也不错

![图片](assets/images/lecture-04-img-09.png)

## Mixture of experts(MoE)

`MoE` 模型结构 vs 原始 `MLP` 模型结构如下:

> `MoE` 核心思想: 使用 4 个和原来一样大的 `FFN`，并且针对于输入选择对应的 `FFN` 做前向和反向传播，这样实现了计算的 `FLOPs` 不变，但是 `FFN` 部分的参数量增加到了原来的4倍；同时路由专家的粒度为  `token` 级别，也就是每一个 `token` 都会路由到一个 `expert`

![图片展示了Dense模型和Sparse模型的结构对比。Dense模型中，输入x经Self - Attention后，分别与FFN Layer 1和FFN Layer 2相加并归一化，再经Self - Attention，最后输出y1和y2。Sparse模型中，输入x经Self - Attention后，路由到FFN 1、FFN 2、FFN 3、FFN 4，分别与FFN 1、FFN 2、FFN 3、FFN 4相加并归一化，再经Self - Attention，最后输出y1和y2。该图与上下文介绍的MoE模型结构相关，直观呈现了Dense和Sparse模型的差异。](assets/images/lecture-04-img-10.png)

同时大量实验表示，训练 `MoE` 模型比训练 `Dense` 模型更快，同时对应的模型的表达能力也会更好，例如部分实验结果如下(相同 `FLOPs` 的情况下，更多参数的 `MoE` 模型表现更好):

<div class="feishu-grid" style="grid-template-columns: 0.5fr 0.5fr;">

<div class="feishu-column">

![MoE模型Loss更低、表达能力更好](assets/images/lecture-04-img-11.png)

</div>

<div class="feishu-column">

![MoE模型训练速度更快](assets/images/lecture-04-img-12.png)

</div>

</div>

从系统层面看，`MoE` 还存在另外一个优点，`MoE` 中的每个 `expert` 天然分块，所以可以放在不同设备上进行推理训练(专家并行)

> 计算、显存、通信 `trade-off`; 此时有更高的总算力、降低显存，但是可能存在通信瓶颈；

![MoE专家并行(类似于 AlexNet)](assets/images/lecture-04-img-13.png)

`MoE` 模型虽然训练更快、表达能力更好，但是由于参数量比较大，所以训练/推理可能放在不同设备上，但是此时可能会出现专家并行利用率低等问题，同时可能出现训练崩溃的问题

![MoE为什么没有变得流行](assets/images/lecture-04-img-14.png)

### MoE parameters

`MoE` 中可以变化的组件/参数:

- 路由函数(`Routing function`)
- 专家大小(`Expert Sizes`)
- 训练方式

#### Routing function

`MoE` 的专家路由是 `token` 级别的，将 `token` 路由到指定的专家进行运算，存在三种可能的路由方案: 针对每一个 `token` 选择合适的 `expert`；针对每一个 `expert` 选择合适的 `token`；全局进行 `expert` 和 `token` 之间的分配

![图片展示了三种专家路由算法，对应“Token chooses expert”“Expert chooses token”“Global routing via optimization”三种方案。左侧为“Token chooses expert”，每个Token向量与Expert向量求内积，选择Top-k Expert；中间为“Expert chooses token”，每个Expert向量与Token向量求内积，选择Top-k Token；右侧为“Global routing via optimization”，通过优化全局决定Expert Assignment。图片与上下文紧密相关，直观呈现了MoE中专家路由的三种方案。](assets/images/lecture-04-img-15.png)

通过消融实验证明，`token` 选择 `expert` 的形式，验证损失更低，同时对应的表达能力可能更好，所以目前的主流做法是每一个 `token` 选择对应的 `expert`

![图片展示了MoE的专家路由为“token choice topk”形式的实验结果。图中包含四个子图，分别对应训练损失、验证损失、HellaSwag和MMLU Var指标。每个子图横轴均为Tokens（B），纵轴为Performance。图中用粉色和蓝色分别表示TC（token choice）和EC（expert choice）两种路由方式，TC的性能曲线整体优于EC。该图与上下文紧密相关，通过实验结果验证了“token choice topk”路由形式的验证损失更低，表达能力可能更好的结论。](assets/images/lecture-04-img-16.png)

`Routing function` 目前存在如下几种方案:

- **`Top-k` 方案: 每一个 `token` 向量和每一个 `expert` 对应的向量求内积，并且得到 `Top-k` 的专家，该 `token` 和每一个专家计算的结果经过门控(门控参数通过内积决定)，输出最终的激活值**
- `Hashing` 方案: 直接使用 `hash` 算法进行每一个 `token` 的路由
- `RL`
- `Solve a matching problem`: 当成一个匹配问题(可能是全局进行 `token` 和 `expert` 的匹配，但是这个模型好像不好训练)

<div class="feishu-grid" style="grid-template-columns: 0.5fr 0.5fr;">

<div class="feishu-column">

![Top-k & Hash](assets/images/lecture-04-img-17.png)

</div>

<div class="feishu-column">

![RL & Match Problem](assets/images/lecture-04-img-18.png)

</div>

</div>

`Top-k` 算法具体实现:

$$\mathbf h_t^l = \sum_{i=1}^N (g_{i,t} FFN_i(\mathbf u_i^l)) + \mathbf u_t^l, \\g_{i,t} = \begin {cases} s_{i,t}, \ s_{i,t} \in Topk(\{s_{j,t}|1 \leq j \leq N\}, K) \\\ 0, otherwise \end {cases} \\s_{i,t} = Softmax_i(\mathbf {u_t^l}^T \mathbf e_i^l)$$

> 输入 `token` 向量表示为 $\mathbf u_i^l$，每个专家的向量表示为 $\mathbf e_i^l$

`DeepSeek MoE` 中引入了共享专家，核心思想是，对于每一个 `token` 除了路由专家层面的特征提取，可能还需要更加通用的一些特征提取方案，所以 `DeepSeek MoE` 更洗颗粒度进行专家切分，同时固定一部分专家作为共享专家，每一个 `token` 公用，同时另外的作为路由专家，根据路由函数调用:

> 但是国外的一些研究例如 `OlMoE` 表明: 可能不是共享专家导致的表达能力的提升，而是多专家本来就可以带来表达能力的提升

<div class="feishu-grid" style="grid-template-columns: 0.333333fr 0.333333fr 0.333333fr;">

<div class="feishu-column">

![DeepSeek MoE 共享专家架构](assets/images/lecture-04-img-19.png)

</div>

<div class="feishu-column">

![DeepSeek MoE消融实验](assets/images/lecture-04-img-20.png)

</div>

<div class="feishu-column">

![OlMoE消融实验](assets/images/lecture-04-img-21.png)

</div>

</div>

#### Train MoE

##### Load-balancing auxiliary loss

> `Top-k` 算法只是提供了一种通过分数选择对应指定专家和最后组合输出激活值的方案，但是其中的一些向量还是需要训练，例如专家嵌入 $\mathbf e_i^l$ 等；下面我们会讨论对应的训练方案分以及这些训练方案可以解决的问题

`MoE` 训练过程中可能存在的问题: 如果大量 `token` 都路由到一个专家，那么这一个专家就会成为性能评价(专家塌陷)；**路由器的训练目标是语言建模损失，但是并不关心负载是否均衡，`Switch Transformer` 中额外添加了负载均衡辅助损失，强行把负载拉平**

总共存在 $N$ 个专家，编号为 $i = 1, \cdots, N$，同时批次大小为 $\mathcal B$, 该批次中的所有 `token` 总数为 $T$，同时使用 $x$ 表示该批次中的一个 `token`，使用  $p_i(x)$ 表示路由器给 `token` $x$ 分配专家 $i$的概率，对应的 $\argmax p(x)$ 表示该 `token` 实际选中的专家，所以我们可以做如下定义:

- $f_i = \frac {1}{T} \sum_{x \in \mathcal B} 1\{\argmax p(x) = i\}$  **$f_i$ 是硬负载，表示有多少比例的 `token` 实际分配给了专家 $i$**
- $P_i = \frac {1}{T} \sum_{x \in \mathcal B} p_i(x)$ ， **$P_i$ 表示软偏好，表示路由器平均给专家 $i$分配了多少概率**

最后的损失函数定义为:

$$loss = \alpha \cdot N \cdot \sum_{i=1}^N f_i \cdot P_i$$

上式中 $\sum_i f_i P_i$ 是两个分布的点积，如果点积越大，说明二者重叠程度越高，如果大部分 `token` 都倾向于选择一个专家，极端情况下 $f_i = 1、P_i = 1, i 取 [1,N] 中特定值，其他为 0$，所以此时 $loss = \alpha \cdot N $ 取得最大值；如果每一个 `token` 倾向于选择一个专家，那么相当于 $f_i = P_i = \frac {1}{N}$，那么此时 $loss = \alpha$ 取得最小值，所以可以防止专家塌陷问题

同时我们可以考虑计算 $loss$ 函数的梯度， $\frac {\partial loss}{\partial p_i(x)} = \alpha N f_i \frac {\partial P_i}{\partial p_i(x)} = \alpha N f_i \frac {1}{T}$，同时我们带入 $f_i$ 可以得到对应的梯度为 $\frac {\alpha N}{T^2} \sum_{x' \in \mathcal B} 1{\argmax p(x') = i}$，所以相当于梯度下降的变化量正比于 $f_i$，也就是:

- 如果专家 $i$ 被分配到的 `token` 越多( $f_i$ 越大)，说明其对应的概率 $p_i(x)$ 减少的速度就会越快(概率向下压得更厉害)
- 如果专家 $i$ 使用比较少，相当于对应分配概率 $p_i(x)$ 减少的速度就会越慢(相对使用多的专家来说就是概率有一定抬升)

> **PS: 启发式算法 -> 使用经验规则快速找到一个足够好的解，但是不保证最优**

`DeepSeek V1-2`中使用了类似于上面的启发式算法来负载均衡专家的选择和设备的选择，对于设备的损失函数中，对应的 $f_i$ 和 $p_i$ 实际上换成了每一个设备上的 `token` 分配比例

- 专家负载均衡损失函数:

>  $K'$ 是 `top-k` 选择的数量

$$\mathcal L_{ExpBal} = \alpha_1 \sum_{i = 1}^{N'} f_i P_i, \\f_i = \frac {N'}{K'T} \sum_{t=1}^T 1 (Token \ t \ selects\ Expert\ i), \\P_i = \frac {1}{T} \sum_{t=1}^T s_{i,t}$$

- 设备负载均衡损失函数:

> 其中 $\epsilon_i$ 表示每个设备上的专家数量

$$\mathcal L_{DevBal} = \alpha_2 \sum_{i=1}^D f_i'P_i', \\f_i' = \frac {1}{|\mathcal \epsilon_i|  }\sum_{j \in \epsilon_i}f_j, \\P_i' = \sum_{j \in \epsilon_i} P_j,$$

`DeepSeek V3` 中在门控中加上了偏置避免了一些负载均衡损失项，这个偏置位置的思想就是，如果专家过于热门，那么就降低这个偏置的大小使得权重更小，如果专家比较冷门，那么增加偏置提升权重

`OlMoE` 做了是否存在负载均衡损失的消融实验，发现如果没有负载均衡损失，那么可能出现训练不稳定以及专家塌陷(也就是几乎所有 `token` 都倾向于选择同样的专家)，实验结果如下:

![图片](assets/images/lecture-04-img-22.png)

> 同时上述的 `Top-K` + 负载均衡损失的算法也应用在了 `DeepSeek Sparse Attention(DSA)` 中，用于平衡子序列的选择

### System Size Problem

#### Parallelism

`MoE` 专家并行:

> 按照每一层计算 `FFN` 参数量比 `Attention` 大很多，甚至在现代大模型中占 80% 以上，`Attention` 主要负责混合、`FFN` 负责变换

- `Attention`: 复制到每台设备，或者做张量并行
- `Model FFN`: 包含门控和多个专家 `FFN`，专家分散到不同设备上
- `All-to-All Dispatch`: 根据 `gating` 结果，将每一个 `token` 隐状态发送到它选中的专家所在的设备
- `All-to-All Combine`: 专家计算完成之后，将结果送回到原设备，并且按照门控权重加权求和

![图片](assets/images/lecture-04-img-23.png)

其他常见并行模式:

![图片](assets/images/lecture-04-img-24.png)

#### Kernel Optimization

`MoE` 存在的另外一个问题是会让每一个专家分配到的 `token` 数量出现不一致的情况，但是 `GPU` 的 `tensor core` 只擅长固定形状的大稠密矩阵乘法，所以针对于这些问题存在一些系统/算子层面的优化算法

例如，假设存在 3 个专家，同时每一个专家的 `FFN` 是 $W_e \in \mathbb R^{d \times d_{ff}}$，分到的 `token` 矩阵是 $X_e \in \mathbb R^{C_e \times d}$，其中 $C_e$ 是一个动态的值，存在如下三种优化方式:

1. **`Batched Matrix Multiplication`(批矩阵乘)**: 把么一个专家的输入 `padding` 到固定容量，然后做批量矩阵乘法，可能存在的问题是 `padding` 浪费算力，同时超过容量的 `token` 可能会被丢弃从而造成信息损失
2. **`Block Diagonal Matrix Multiplication`(块对角线矩阵乘**): 将所有 `token` 拼接成为一个大的矩阵，并且把专家计算表达转换为块对角矩阵乘法，还是要求固定长度需要 `padding`
3. **`Block Sparse Matrix Mulitplication`(块稀疏矩阵乘)**: 允许每个块大小不同，同时只计算非零块

![图片](assets/images/lecture-04-img-25.png)

#### MoE architecture modifications

- **`LatentMoE`**:   `token` 原始维度上计算门控参数和专家选择情况，传递到指定专家之前映射到低维，低维向量和专家运算之后传递回指定的硬件并且升高为原来的维度，显著减少了通信开销、专家占用显存大小，但是模型表达能力可能存在降低，**是表达能力和通信开销之间的 `trade-off`**

![图片](assets/images/lecture-04-img-26.png)

#### Other Problems

- **`MoE` 专家排队**: 之前的 `MoE` 系统可能存在，如果某一个专家需要处理的 `token` 数量太大导致 `token` 被丢弃的情况，但是这个问题目前已经得到了解决(这个问题可能导致其他人的 `query` 可能会导致你的回答 `token` 被丢弃)

![图片](assets/images/lecture-04-img-27.png)

- **`MoE` 稳定性问题**: 注意到 `MoE` 在计算专家选择概率和门控参数的时候会使用 `softmax` 操作，这一个操作可能会导致潜在的不稳定问题，解决方案是使用之前提到过的 `z-loss` 以及仅仅在专家路由的部分使用 \`Float32 ，同时之前也有一些研究做了对应的消融实验，发现如果没有 `z-loss` 确实可能出现比较严重的训练不稳定问题

![图片](assets/images/lecture-04-img-28.png)

- **`MoE` 微调过拟合**: 一些研究表示对于 `MoE` 可能产生比较严重的过拟合，解决方法可能是不微调 `MoE` 而是微调其他的模块(例如 `Attention` 、`FFN`等），同时数据量比较大的情况下甚至可以重新训练 `MoE`(DeepSeek 方案就是 SFT 阶段重新训练 MoE)

![图片](assets/images/lecture-04-img-29.png)

### Other Optimization

#### Upcycling

`Upcycling` 表示可以利用稠密的、已经训练好的模型，将对应稠密模型中的 `MLP` 的权重进行拆分复制得到 `MoE` 中每个专家的权重，然后初始化一个随机的路由器，在训练过程中这些权重就会向不同方向进行分化，从而提升模型表达能力，这是一种已有的稠密模型训练出 `MoE` 模型的方法(`MiniCPM` 模型利用这种方法实现了从已有的小参数模型训练出大一点参数的 `MoE` 模型，并且效果不错)

![图片](assets/images/lecture-04-img-30.png)

### DeepSeek MoE Development

- `DeepSeek V1`: 共享专家、`Top-k` 专家选择、负载均衡损失

![图片](assets/images/lecture-04-img-31.png)

- `DeepSeek V2`: 引入 `Top-M Device Routing`、设备负载均衡损失、通信负载均衡损失

![图片](assets/images/lecture-04-img-32.png)

- `DeepSeek V3`: `Sigmoid+Softmax TopK + TopM`、门控参数加偏置项(减少需要添加的负载均衡损失)

![图片](assets/images/lecture-04-img-33.png)
