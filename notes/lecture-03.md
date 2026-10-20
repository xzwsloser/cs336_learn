# Lecture3: Model Architecture

## 随笔

- 原始 Transformer 模型结构:

![图片展示了标准Transformer模型结构及关键组件。输入经位置编码后，进入多头注意力机制，再通过加法和归一化，接着是前馈神经网络（FFN）层，同样经历加法和归一化。此结构重复N次。图片还列出了位置嵌入公式，FFN使用ReLU激活函数，以及归一化类型为后置归一化和LayerNorm。该图与文档中对原始Transformer模型结构的介绍相呼应，直观呈现了其组成及关键部分。](assets/images/lecture-03-img-01.png)

- `Assigment1` 要求实现 `Transformer` 变种结构(归一化位置、位置编码、激活函数):

![图片展示了Transformer模型变种结构，左侧为原始Transformer模型结构，包含Token Embedding、Transformer Block等部分。右侧是变种结构，引入了LayerNorm在块前、使用Rotary position embeddings (RoPE)、FF layers使用SwiGLU而非ReLU、Linear layers和layernorm无bias等差异。右侧还列出为何选择这些差异及应如何选择的问题。该图与文档中“`Assigment1`要求实现Transformer变种结构”内容相关，直观呈现了变种结构与原始结构的区别。](assets/images/lecture-03-img-02.png)

- 各系列大模型使用不同架构，考虑公共架构(FFN、Attention、PE等)、超参数、训练技巧

## Model Architecture

### Norm

#### Pre-vs-post norm

`Transformer` 论文中 `Layer Norm` 层被放在最后，但是现代模型几乎都使用 `pre-norm` 的形式，`post-norm` 和 `pre-norm` 模型架构以及计算流程分别如下:

![这张图片是关于Transformer模型架构中Pre-vs-post norm的示意图，左侧展示了Post-LN Transformer的架构流程，右侧展示了Pre-LN Transformer的架构流程，同时还配有对应的公式推导。图片标注来源为Xiong 2020，还说明了几乎所有现代大语言模型都使用pre-norm架构，只有BERT是post-norm，OPT350M作为例外同样使用post-norm架构，另外提到需将LayerNorm设置为不影响主残差路径的形式，且该图是2024年相关研究中介绍的共识内容。](assets/images/lecture-03-img-03.png)

`pre-norm` vs `post-norm` 实验效果:

![这张图片展示了Transformer架构中pre-norm与post-norm相关的实验对比结果。左侧折线图对比了不同pre-norm变体以及post-norm模型在English-Vietnamese开发集上的BLEU值随训练轮次（epochs）的变化，横轴为训练轮次，纵轴为Dev BLEU值，可看到各模型的性能表现差异；右侧包含两个子图，(a)图为IWSLT任务的验证损失随训练轮次的变化，标注了不同配置下的模型曲线，(b)图为对应任务的BLEU值随训练轮次的变化，还有子图展示了预训练步数的验证损失，直观呈现了两种norm形式及不同配置模型的实验效果。](assets/images/lecture-03-img-04.png)

> 🤔 Q: 为什么主流大模型一般采用 `pre-norm` 而不是 `post-norm`?  
> A:  
> CS336 Teacher: `pre-norm` 架构的残差路径上保留了原始信息，使用 `post-norm` 出现复杂效应，残差路径上每一层都要经过一次 `Layer Norm`，导致反向传播过程中梯度值会发生变化，使得训练不太稳定  
> `Post-LN`: $x_{l+1} = LN(x_l + F_l(x_l))$  
> `Pre-LN`:   $x_{l+1} = x_l + F_l(LN(x_l))$  
> 另外一种视角是: 残差流可以看成一个共享总线，每一层从总线中读取信息、写入信息，最终 `logits` 静思语各层贡献的线性叠加；`Pre-LN` 中残差流本身不会被归一化或者非线性化  
> 但是 `Pre-LN` 也是存在一定代价，虽然深层网络利用 `Pre-LN` 更加稳定，但是 `Pre_LN` 相当于信息可以跳过所有层直达输出，所以我认为原始信息在最终学习到的特征中占比增大，有效深度比较小，表达能力可能比较弱；另外一些研究显示，我们甚至可以增强 `Post-LN` 中原始信号的比例来解决这个问题，例如放大残差项等也可以实现深层网络  
> DeepSeek V4 Flash:  
> Pre-LN 不是让原始信息"占的比重大"，而是**保留了它"最浅、纯线性、不被重编码"的通道地位**；Post-LN 不是"调小了原始信息的权重"，而是**让包括残差在内的所有通道都被迫变成全深度**。前者导致"有效深度小"（训练稳），后者导致"被迫深"（训练难，但每层容量略大）。  
> Q: 为什么选择添加 `pre-norm` 在计算层(`FFN`、`Attention`)前面而不是后面?  
> A:  
> `LayerNorm` 使得各个维度的特征分布更加均匀，可以提升训练的稳定性，其实目前的一些主流模型也是可以把 `LayerNorm` 层加入到计算层的后面的，甚至也有 `double norm` 的方案，也就是计算层前后都会加上 `LayerNorm`，加上 `LayerNorm` 可以提升训练稳定性，但是最好不要干扰残差路径

#### LayerNorm vs RMSNorm

- `LayerNorm`:  $y = \frac {x - E[x]}{\sqrt {Var[x] + \epsilon}} \gamma + \beta$ (`Transformer` 论文原始做法， $d_{model}$ 维度上做归一化操作，GPT3/2/1、OPT、GPT-J等模型采用)
- `RMSNorm`: $y = \frac {x}{\sqrt {||x||_2^2 + \epsilon}} \gamma$  (LLaMA、PaLM、T5等模型采用)

大部分主流模型都使用 `RMSNorm`，`LayerNorm` 的表达能力其实比 `RMSNorm` 更好(`LayerNorm` 直接在某一个维度上归一化，但是 `RMSNorm` 仅仅考虑某一个元素)，但是这个差别不太大，`RMSNorm` 的一个优点是计算效率更高(操作次数少、参数少)

同时原始 `Transformer` 论文中采用 `bias` 但是现代模型架构中一般都舍弃了 `bias` ，主要原因是:

1. `bias` 其实对于模型表达能力影响不大，有些时候甚至会引起稳定性问题
2. `bias` 需要额外的计算和显存开销，去掉 `bias` 可以节约系统资源

### Activations

> <code>Activations Function</code> 具体形式参考: [深度学习](https://feishu.cn/wiki/Kj51w8QU6iwGUSkLZE9cTTjPnJh)

门控机制使用的场景比较广泛，在激活函数中使用门控可以实现特定位置激活值的屏蔽与表达的学习，同时门控机制需要增加一个矩阵参数，所以一般来说，使用门控机制的 `FFN` 层一般会使用更小的隐藏层维度(例如一般使用 $\frac {2}{3} d_{ff}$

$$FFN_{GeGLU}(x, W, V, W_2) = (GeLU(xW) \otimes xV) W_2$$

$$FFN_{swiGLU}(x, W, V, W_2) = (Swish_1(xW) \otimes xV) W_2$$

### Transformer Block

#### Serial vs Parallel Layers

#### 传统 `Transformer Block` 串行进行 `Attention` 和 `MLP` 模块的计算，`GPT-J` 提出了并行计算 `Transformer Block` 的方式(块内计算而非块间并行)

![图片展示了Transformer Block的结构。从上至下依次为Add、Dropout、Position - Wise Feed - Forward、Norm、Add、Dropout、Causal Multi - Head Self - Attention、Norm等模块，箭头指示数据流动方向。该图与上下文紧密相关，上下文提到传统Transformer Block串行进行Attention和MLP模块，此图直观呈现了串行Transformer Block的内部结构，帮助理解其操作流程。](assets/images/lecture-03-img-05.png)

`GPT-J` 中提出了一种 `Transformer Block` 块内的并行计算算法:

- `Serial Layers`: $y = x + MLP(LayerNorm(x + Attention(LayerNorm(x))$
- `Parallel Layers`: $y = x + MLP(LayerNorm(x)) + Attention(LayerNorm(x))$

大部分现代大模型依然采用 `Serial Layers` 的方式,可能的原因是传统的 `Serial Layers` 遵循首先全局通信(`Attention`)后本地计算(`MLP`)的紧耦合迭代，但是块内并行打破了这个规则，相当于对于特征的有线耦合深度减半，虽然可以换了延迟和 `LN` 小幅度收益，但是表达上的损失比较大

### Position Encoding

位置编码(`Position Encoding`)的作用是区分输入序列中的不同位置，给输入序列中的不同位置加上不同的偏移，使用比较多的几种位置编码如下(正余弦位置编码、绝对位置编码、相对位置编码、RoPE等):

![图片展示了四种位置编码方法及其相关模型。左上角是正余弦位置编码，公式为Embed(x, i) = vx + PEpos，PEpos计算公式包含sin和cos函数。左下角是绝对位置编码，公式为Embed(x, i) = vx + ui。右下角是相对位置编码，公式为eij = (xjWQ(xjWK + aijK)T) / √dz。右上角是原Transformer模型。右下角还列出了使用相对位置编码的模型，如T5、Gopher、Chinchilla等，以及使用Rope嵌入的模型，如GPTJ、PaLM、LLaMA等，这些模型多为2024年及之后的模型。](assets/images/lecture-03-img-06.png)

#### RoPE

相对位置编码可以表示为函数 $f(x,i) $使得如下式子成立:

$$<f(x,i), f(y,j)>= g(x, y, i - j)$$

> 通俗的理解可以看成，两个词向量 $x,y$ 之间的关系仅仅取决于它们之间的相对位置 $i - j$ 而不是绝对位置，例如 `A Apple` 放在句首和句尾两个词之间的关系应该是一样的，而不是不同的；`Sine`、`Absolute`、`Relative Embeddings` 都不满足上述关系

从一个通俗的视角来理解 `RoPE`，词向量具有固定性，我们位置编码过程中最好不要改变不同词之间的内积，也就是如下两个性质:

- 词向量本身编码和绝对位置无关
- 对于任意角度的旋转，内积应该是一个不变量

例如对于句子 `we know that`，我们可以这样考虑，对于指定位置的词(这里的位置索引可以从 $0$ 开始，索引为 $n$ 的位置，旋转 $n \theta$ 角度，这样我们可以发现对于位置差为 $i - j$ 的两个词向量，相当于最后的夹角都是 $\alpha_0 - (j - i) \theta$ 所以最后的内积也是和 $j - i$ 相关的值，从而满足了上述相对位置编码的性质

![这张图展示了旋转位置编码（RoPE）的核心逻辑，围绕“选择哪种旋转方式”展开讲解，核心为将二维坐标旋转的思路应用于词的位置编码：上方虚线框内呈现了对二维向量进行旋转的示例，标注了Query/Key向量的坐标$(x_1,x_2)$，旋转后得到旋转后的坐标$X_1',X_2'$；下方示意图展示了带RoPE的Transformer结构，原始Query/Key词向量经过位置编码后得到带颜色标识的位置编码后的Query/Key向量，还补充了Gemini-4的优化示意；图底部标注了“仅配对坐标并在二维空间中旋转”的核心思路，这一设计借鉴了复数的特性，以满足相对位置编码的需求。](assets/images/lecture-03-img-07.png)

> 考虑 $D$ 维的情况，高维度下旋转的方式比较多，`RoPE` 论文中的做法是按照两个两个分组的方式进行旋转，例如对于 $D$ 维可以分为 $\frac {D}{2}$个组，对于每一个组设置一个固定的旋转角度例如 $\theta_1, \theta_2, \cdots, \theta_{\frac {D}{2}}$，同时考虑每一个词向量的位置 $m$，每一个维度组的旋转角度为 $m\theta_1, m \theta_2, \cdots, m \theta_{\frac {D}{2}}$

`RoPE` 公式: $f_{q,k} (\mathbf x_m, m) = \mathbf R^d_{\Theta, m} \mathbf W_{q,k} \mathbf x_m$，对应的旋转矩阵如下:

![图片展示了`RoPE`公式中对应的旋转矩阵$R_{\\\\Theta,m}^d$。矩阵为$D \\\\times D$的结构，其中$D$为维度数。矩阵对角线上的元素为$\\\\cos m\\\\theta_1$、$\\\\cos m\\\\theta_2$、...、$\\\\cos m\\\\theta_{d/2}$，非对角线上的元素为$\\\\sin m\\\\theta_1$、$-\\\\sin m\\\\theta_2$、...、$-\\\\sin m\\\\theta_{d/2}$。该矩阵与`RoPE`公式$f_{q,k}(\\\\mathbf x_m,m)=\\\\mathbf R^d_{\\\\Theta,m}\\\\mathbf W_{q,k}\\\\mathbf x_m$相关，用于说明不同维度上做不同角度旋转的数学表示。](assets/images/lecture-03-img-08.png)

> 🤔 Q: `RoPE` 为什么不同维度上要做不同角度的旋转？具体的含义是什么?  
> A:  
> CS336 Teacher 的观点是捕捉不同维度上距离的变化情况，这个苏神的博客中应该有提到，这个后面可以补充一下 // TODO

### Hyperparameters

`Transformer` 参数: **`FFN` 参数规模(`hidden layer` 参数层设置)、多头注意力机制中 `head` 数量；词表大小(`vocab size`)等;**

训练配置问题: 正则化参数、`Dropout` 参数设置等、`deep or wide` 模型?

#### FFN Hyperparameters

$$FFN(x) = (Activation(xW_1+b) \odot V) W_2 + b$$

我们考虑前馈神经网络的隐藏层的维度 $d_{ff}$和输入输出的模型维度 $d_{model}$ 之间存在如下关系: $d_{ff} = 4d_{model}$,同时不放假设 $d = d_{model}$，那么相当于对于 `FFN` 对应的参数量大小应该为 $2 d d_{ff}$，所以总参数量应该为 $2d \cdot 4d = 8d^2$，如果激活函数使用门控性质，那么相当于 $3d \cdot d_{ff} = 8d^2$ (保持参数量和计算量不变的情况下)，可以计算得到 $d_{ff} = \frac {8}{3} d$，各个模型对应的 $\frac {d_{ff}} {d_{model}}$ 设置如下:

![这张图为前馈神经网络的FFN超参数相关的模型与对应$d_{ff}/d_{model}$比值的对照表，清晰罗列了多款不同模型的该比值设置。其中，非GLU类的PaLM的比值为4，使用门控激活的GLU类模型的比值多在2.5至2.68附近，比如LLaMA 70B为2.68、Qwen 14B为2.67、T5 v1.1为2.5；使用GQA等其他注意力机制的模型比值多在3.5附近，例如Mistral 7B和LLaMA-2 70B的比值均为3.5，另有Yi 34B的比值为2.85，这些数值和上下文提及的不同类型模型的比值范围相匹配，直观呈现了各类模型该超参数的具体数值。](assets/images/lecture-03-img-09.png)

> 非 `GLU` 模型 -> 4
> 
> `GLU` 模型     -> 2.67 附近
> 
> 可能使用其他形式注意力机制(`GQA`等) -> 可能在 3.5 附近
> 
> 但是 `T5` 使用了 64 倍的比值，核心思想是想提升硬件利用率(提升算法算术强度值???)，但是后续出了 `T5 v1.1` 使用了正常范围的比值

`Feed-Forward` 比值和模型损失关系对比如下(使用 `50 M` 参数):

![这张图片是50M参数模型下的前馈网络比值（$d_{ff}/d_{model}$）与损失增加量的关系折线图，横轴为前馈网络比值，纵轴为损失增加百分比。图中有两条折线，分别对应$n_{head}=8$（以叉形标记）和$d_{model}/n_{head}=64$（以圆点标记）的情况。当前馈网络比值处于$10^0$至$10^1$区间时，两条折线的损失增加量相近，数值均低于2%；当比值超过$10^1$后，两者的损失增加量均快速上升，且$d_{model}/n_{head}=64$对应的损失始终略高于$n_{head}=8$的情况，在比值较高时接近8%。该图用于直观展示前馈网络比值与模型损失的关联，对应文档中关于Feed-Forward比值和模型损失关系对比的内容。](assets/images/lecture-03-img-10.png)

#### Multi-Head Hyperparameters

对于多头注意力机制，我们一般采用 $d_{head} \times n_{head} = d_{model}$，但是也有模型存在 $d_{head} > \frac {d_{model}}{n_{head}}$的情况，一些大模型对于多头注意力的参数设置如下:

![图片是一张表格，展示了不同模型的多头注意力机制参数设置情况。表格包含Num heads（头数）、Head dim（头维度）、Model dim（模型维度）和Ratio（比例）四列。其中，GPT3的头数为96，头维度为128，模型维度为12288，比例为1；T5的头数为128，头维度为128，模型维度为1024，比例为16；LaMDA的头数为128，头维度为128，模型维度为8192，比例为2；PaLM的头数为48，头维度为258，模型维度为18432，比例为1.48；LLaMA2的头数为64，头维度为128，模型维度为8192，比例为1；Qwen 3.5 (27B)的头数为24，头维度为256，模型维度为5120，比例为1.2。该表与上下文介绍的多头注意力机制参数设置相关。](assets/images/lecture-03-img-11.png)

> $$ratio = \frac {d_{head} * n_{head}}{d_{model}}$$

#### Deep vs Width Paramters

还需要考虑模型的维度和深度的比例关系，考虑模型维度和深度之间的 `trade-off`，例如大多数模型 $\frac {d_{model}}{n_{layer}}$关系如下:

![图片为表格，展示了不同模型的$d_{model}/n_{layer}$值。表格包含Model和$d_{model}/n_{layer}$两列，Model列列举了BLOOM、T5 v1.1、PaLM (540B)、GPT3/OPT/Mistral/Qwen /OLmo 3、LLaMA / LLaMA2、Gemma 3、Gemma 4、T5 (11B)等模型名称，$d_{model}/n_{layer}$列对应各模型的数值，如BLOOM为205，T5 v1.1为171等。表格右下角标注“Sweet spot?”，表明该值一般在100附近，与上下文讨论模型维度和深度比例关系的内容相关。](assets/images/lecture-03-img-12.png)

> 这个值一般在 100 附近

模型的深度和宽度之间的 `trade-off` 其实是模型表达能力和硬件利用率之间的 `trade-off`，越深月复杂的模型可以学习到更多特征，但是缺点就是难以并行，例如就算不同的 `Layer` 分布在不同的 `GPU` 上的时候，当前层计算必须等到上一层计算完成才可以，所以难以并行；但是对于模型宽度的维度，其实比较容易实现模型的并行计算

![这张图展示了模型不同层在对应GPU上的分布及计算传递逻辑，分为上下两部分。上方为模型结构，依次是Layer0到Layer3，前向传播时，数据从Layer0依次流向Layer1、Layer2、Layer3，反向传播时，梯度从Layer3依次回传到Layer2、Layer1、Layer0，呈现出串行的顺序计算关系。下方对应四个GPU，分别标注为GPU0到GPU3，每个GPU上方对应一个模型层，体现了不同层分布在不同GPU上的情况，直观解释了文章中提到的模型深度难以并行的特点，即当前层计算需等待上一层完成，无法并行处理。](assets/images/lecture-03-img-13.png)

 模型深度/宽度比值和 Loss、FLOPs 与模型表达能力实验图:

![图片展示了模型深度/宽度比值和Loss、FLOPs与模型表达能力的关系。左侧图表以Aspect Ratio (d_model / n_layer)为横轴，不同参数量（50M、274M、1.5B）的曲线呈现不同性能表现，表明多种架构可实现相似性能。右侧图表以FLOPs为横轴，展示了不同模型的Loss、Super-GLUE Accuracy及参数量，直观呈现了模型性能与资源利用情况。该图与上下文关于模型维度和深度比例关系的讨论相呼应，说明在一定范围内可选择合适超参数，真正关心系统资源利用率。](assets/images/lecture-03-img-14.png)

> 一般来说我们可以在一定范围内部选择一个合适的超参数，然后真正关心的应该是系统资源利用率

#### Vocab Size Parameters

单语种模型和多语种模型词表大小存在很大的不同，多语言模型词表更大，同时模型参数量也会更大:

![图片展示了单语种模型和多语种模型的词表大小情况。左侧为单语种模型，词表大小在30 - 50k，包括Original transformer、GPT、GPT2/3、T5/T5v1.1、LLaMA等模型，词表数量分别为37000、40257、50257、32128、32000。右侧为多语种/生产系统，词表大小在100 - 250k，包括mT5、PaLM、GPT4、Gemma 4、DeepSeek、Qwen 15B、Yi等模型，词表数量分别为250000、256000、100276、262144、100000、152064、64000。该图与上下文关于多语言模型词表大小及模型参数量更大的内容相关。](assets/images/lecture-03-img-15.png)

#### Dropout and other regularization

大模型和深度学习模型存在一些区别，大模型 `pre-train` 阶段语料库很大，同时几乎只会训练一遍，所以在计算量受限的情况下，很难出现过拟合的情况，所以是否需要对应的 `Dropout` 和正则化方案需要讨论

![各模型 Dropout、weight decay参数对比](assets/images/lecture-03-img-16.png)

> 对于新一代大模型，经常使用权重衰减(`Weight Decay`)，但是却很少使用 `Dropout`

权重衰减除了防止过拟合之外，有时可能还具有让训练优化过程更加平稳的作用(之前的分析就是约束了参数变化的范围，使得参数只能在一定的区间、形状内部产生变化，变化可能更加平滑？)

![这张图表为深度学习模型训练相关的对比图示，分为三个子图。左侧子图展示了验证损失与训练损失的关联情况，不同颜色的点对应不同的权重衰减参数设置，其中绿色、红色、蓝色分别对应权重衰减系数为0.0、0.1、0.3。中间子图标题为“10× cosine LR decay”，右侧子图标题为“Constant LR”，二者均以迭代次数为横轴、训练损失为纵轴，同样用不同颜色曲线对应不同权重衰减设置，部分曲线标注了“tiny LR”。三个子图下方的文字分别为“It’s not to control overfitting”、“Weight decay interacts with learning rates (cosine schedule)”，体现了权重衰减与学习率调整的相关规律，呼应了文档中关于大模型训练中权重衰减与正则化的讨论内容。](assets/images/lecture-03-img-17.png)

> 权重衰减和余弦学习率调整等组件作用可以使得训练优化过程更快，收敛更快

### Stability tricks

![模型训练过程中出现不稳定的情况](assets/images/lecture-03-img-18.png)

下面我们讨论可能会导致模型训练过程中出现不稳定的情况

#### Softmax

`Softmax` 在大模型架构中主要存在如下几处应用:

- 模型最后经过 `Softmax` 输出词表中各词的概率
- 注意力机制中的缩放点积注意力，归一化注意力分数 $score = softmax(\frac {QK^T}{\sqrt d})$

`Softmax` 存在的问题可能包括: 指数增长太快、分母为 0 等情况，从而可能导致潜在的训练不稳定情况

![图片展示了Transformer模型架构。左侧为模型整体流程，从输入到输出概率，依次经过Token Embedding、Transformer Block（重复多次）、Norm、Linear（Output Embedding）、Softmax等模块。右侧是Transformer Block内部结构，包含Causal Multi-Head Self - Attention w/ RoPE、Norm、Position - Wise Feed - Forward、Norm等组件，输入输出张量形状为(batch_size, seq_len, d_model)。该图与文档中介绍的Transformer模型架构内容相关，直观呈现了模型各部分组成及流程。](assets/images/lecture-03-img-19.png)

`Softmax` 计算方式:

$$\log (P(x)) = \log (\frac {e^{U_r(x)}}{Z(x)}) = U_r(x) - \log(Z(x)) \\Z(x) = \sum_{r'=1}^{|V|} e^{U_{r'}(x)}$$

我们可以考虑到这里 $\log Z(x)$ 可能存在不稳定的情况，例如这个值太大或者太小可能都不太好(`Softmax` 的作用是归一化，产出输出概率，一些极大的值和极小的值可能导致最后输出的结果不稳定，例如一个主成分很大，其他为 0)； **针对上述情况，我们可以考虑加入对于 $\log (Z(x))$ 的正则化惩罚项，让 $Z(x)$ 尽可能接近 1，这种方式可以用于解决输出端 `Softmax` 不稳定的情况**

$$L = \sum_i [\log (P(x_i)) - \alpha (\log(Z(x_i)) - 0)^2] = \sum_i [\log (P(x_i)) - \alpha \log^2 (Z(x_i))]$$

对于注意力机制中使用的 `Softmax` 操作，我们可以考虑在进行矩阵乘法之前首先分别对于 `Q、K` 进行归一化操作，可以经过 `LayerNorm` / `RMSNorm`，归一化之后的 `Q、K` 进行矩阵乘法之后可以使得后续 `Softmax` 操作更加稳定 (`QKNorm`)

![图片展示了Transformer模型中多头注意力机制（Multihead attention）和前馈网络（Feed Forward）的结构。多头注意力部分包含输入层、QKV层、Split层、BMM1层、Softmax层、BMM2层和Proj层；前馈网络部分有LN层、FC1层、Squared ReLU层、FC2层和加法层。该图与上下文紧密相关，直观呈现了文档中介绍的`Softmax`操作在注意力机制中的应用，以及前馈网络的结构，帮助理解模型架构。](assets/images/lecture-03-img-20.png)

`Logit soft-capping` 也是一种可以稳定 `Softmax` 运算的操作，核心思想是使用 `Tanh` 函数实现对于 `logits` 值的约束，对应的实现为：

$$softcap(x) = c \tanh(\frac {x}{c})$$

和硬阶段相比，使得变化更加平滑，同时硬截断对于超出指定范围的输入可能产生梯度为 0 的情况

![Softmax 稳定操作对比](assets/images/lecture-03-img-21.png)

### Attention Head

大部分模型还是使用多头注意力机制(`MHA`)，但是还是存在部分模型使用其他的注意力机制:

- `GQA / MQA`: 通过减少头的数量来实现推理资源消耗的减少
- `Sparse or sliding window attention`: 通过限制注意力机制的形式来计算来减少计算开销
- `SSM stuff (Jamba、Falcon3、Qwen3.5)`: 空间状态模型

#### GQA/MQA

首先考虑传统 `MHA`，对于整个 `MHA` 运算，我们可以计算对应的计算次数和内存占用情况，同时计算对应的算法算术强度:

![图片](assets/images/lecture-03-img-22.png)

同时考虑推理过程，用户输入一段 `token`，在注意力计算过程中我们需要使用 `kv cache` 对于 `KV` 进行存储，从而避免每次都需要把完整的序列进行矩阵计算

![图片](assets/images/lecture-03-img-23.png)

> 但是在推理过程中，我们可以发现计算次数为 $bnd^2$，**同时内存使用情况为 $bn^2d + nd^2$**(分别对应 `KV Cache` 和最后的 `projection`,对应的算术强度为 $O((\frac {n}{d} + \frac {1}{b})^{-1})$，此时提升算术强度我们需要考虑使用更大的 `batch` 或者更短的序列或者更大的模型维度等

可以发现这里存在的问题是，`K、V` 是可存储的，同时 `K、V` 数量就是头的数量，这里我们可以考虑使用很少的头，例如 `MQA` 中实现了 `K、V` 共享

![图片](assets/images/lecture-03-img-24.png)

> `MQA` 实现:
> 
> - `Q` -> 维度变化到 $h * head\_dim$
> - `K、V` -> 维度变化到 $head\_dim$
> 
> 每一个 `Q` 分别和 `K、V` 做矩阵乘法，最后结果进行堆叠即可
> 
> 推理过程中大大减少了 `KV Cache` 的大小

`MQA`/`GQA`/`MHA` 需要解决的问题主要还是 **计算效率、资源利用率等因素和模型表达能之间的 `trade-off`**

![图片](assets/images/lecture-03-img-25.png)

> `MQA` 比较极端，所有的 `Q` 共享一份 `K、V` 使得模型表达能力比较差；同时 `MHA` 虽然模型表达能力比较强，但是内存资源消耗比较严重；三者计算量几乎保持一致(从 `Q` 的角度来看，几乎都是分为了 $h$份，每一份都会经过矩阵乘法和对应的 `softmax` 操作)

![MHA、GQA、MQA计算效率、表达能力对比曲线](assets/images/lecture-03-img-26.png)

#### Sparse / sliding window attention

稀疏注意力机制核心思想是在计算过程中序列中的词个不需要和全局中的所有词计算注意力分数，而是只需要和特定范围(例如 `Astrous Self Attention` 中指定步长位置词、`Local Self Attention` 中指定窗口的词计算注意力分数)，从而极大程度上减少了计算量

![Atrous Self Attention](assets/images/lecture-03-img-27.png)

![Local Self Attention](assets/images/lecture-03-img-28.png)

![Sparse Self Attention](assets/images/lecture-03-img-29.png)

同时非常类似于卷积神经网络，可以通过多层的稀疏注意力和全局注意力的堆叠实现感受野的进一步扩大，同时稀疏注意力也具有一定的位置信息，所以对于位置编码的方案有一定的影响(例如一些最最新的模型中不再使用位置编码)

![图片](assets/images/lecture-03-img-30.png)
