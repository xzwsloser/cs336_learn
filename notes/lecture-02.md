# Lecture 2: PyTorch & Resource Accounting

## 随笔

- `FLOPS` -> 浮点运算次数，表示训练/推理过程中总共做了多少次浮点运算，一般计算方式为 $FLOPS = 6 \times N \times D$，其中 $N$ 表示模型参数量，$D$ 表示训练数据量，$6$ 可以理解为对于每一个 `token` 计算次数为前向传播 2 次(矩阵乘 = 乘+加)、反向传播 4 次 `FLOPS`，`Scaling Law` 中可以首先利用小规模实验来估计 `FLOPS` 和 loss 之间的关系
- 目标: 有限资源(例如计算资源、内存资源等)下训练出最优的模型(考虑模型能力、效率等因素)
- Q1: 70B 参数模型、15T tokens 训练语料，预训练在 1024 H100 GPU 上训练时间估计？

```
total_flops = 6 * 70e9 * 15e12
h100_flop_per_sec = 1979e12 / 2
mfu = 0.5
flops_per_day = h100_flop_per_sec * mfu * 1024 * 60 * 60 * 24  // mfu -> 模型有效算力利用率
days = total_flops / flops_per_day
```

- Q2: 8 卡 H100 上最大可以训练多大模型，优化器使用 AdamW?

```
// 考虑显存
h100_bytes = 80GB = 80e9 bytes
bytes_per_parameter = 2 + 2 + (4 + 4) // parameters、gradients、optimizer、state
num_parameters = (h100_bytes * 8) / bytes_per_parameter
```

## PyTorch

### Tensor

> 主要讨论浮点 Tensor 的几种精度。
>
> sign -> 符号位、exponent -> 指数位、fraction -> 尾数位。
>
> **value = (-1)^sign × 2^(E − bias) × (1 + M / 2^f)**

- parameters、gradients、activations、optimizer states 都是以 float32 类型的 tensor 存在的，占用 32bit (IEEE 标准)。FP32 -> 内存占用比较大 (sign 1bit + exponent 8bit + fraction 23bit) (**充足显存、小模型**)
- float16 (FP16) (sign 1bit + exponent 5bit + fraction 10bit)，运算效率高但是精度低，可以表示的范围不够
- bfloat16 (BF16) (sign 1bit + exponent 8bit + fraction 7bit)，指数位和 FP32 一样，但是精度低一点，但是在 DL 中不需要太高的精度 (**有限显存、大模型**)
- **混合精度训练**: BF16 -> 参数、激活值、梯度；FP32 -> 优化器 (PyTorch AMP Lib)
- FP8: 两个版本(精度和指数位之间的 trade-off)

![四种浮点精度格式（FP16、BF16、FP8 E4M3、FP8 E5M2）的位元组成结构，每种格式的字段分为符号位（sign）、指数位（exponent）、尾数位（mantissa）。图中清晰标注了各格式的位元分布及对应计算数值，可直观对比不同浮点精度的结构差异。](assets/images/lecture2-float-formats.png)

- `einops` -> `Tensor` 操作库，`einops` 库可以使用一套类似于爱因斯坦记号的描述性字符串实现张量形状变化，使用语法糖增加代码可读性

```python
import torch
from einops import rearrange

z = rearrange(x, 'a b c -> a b c2 c1', c1=2, c2=2)  # [2, 3, 2, 2]
```

## Resource Accounting

计算量/计算性能衡量指标:

- `FLOPs`: 浮点数计算次数(衡量计算量)
- `FLOP/s` (`FLOPS`): 每秒可进行浮点数计算的次数(描述硬件性能)，针对特定数据格式，例如 `BF16` 等

`FLOPs` for 其他操作:

> 硬件设计中，乘法和加法效率类似

- `element-wise` 运算量大概就是矩阵大小 $m \times n$
- 矩阵加法运算量也是矩阵大小 $m \times n$
- 矩阵乘法的运算量最大(每一个参数乘一次、加一次)

GPU 硬件架构简化描述:

> 计算时，从 Memory 取出数据到 Compute 计算单元。
>
> 计算完成，从 Compute 计算单元搬运数据到 Memory。

![GPU 硬件架构简化结构：上方 Compute 区域为计算单元，下方 Memory 区域为内存，两者通过连接线相连，计算单元需从内存取出数据完成计算，计算完成后再将数据搬运回内存。](assets/images/lecture2-gpu-arch.png)

计算效率决定因素:

- 计算单元计算速度 (`FLOP/s`) -> 计算效率
- 内存带宽 (`bytes/s`) -> 通信效率

计算和通信同时进行，计算过程的全部时间取决于计算和通信两个操作中耗时比较长的一个操作(耗时长的就是瓶颈):

> 针对每一种算法，瓶颈可能不同，对于某些算法可能瓶颈是计算，对于另外一些算法瓶颈可能是通信，例如对于 `ReLU` 瓶颈就是通信

```
total_time = max(communication_time, computation_time)
```

加速器强度定义: 传输一个字节过程中的计算量大小，计算方式如下。

算术强度定义: 每处理一个字节完成了多少计算量。

```
accelerator_intensity = flop_per_sec / bytes_per_sec
arithmetric_intensity = flops / bytes
```

算法瓶颈 (关注算法本身 & 处理器本身):

- `Memory-bound`: `arithmetric_intensity` < `accelerator_intensity`
- `Compute-bound`: `arithmetric_intensity` > `accelerator_intensity`

> 提高算术强度，每处理一个字节尽量做大量计算，例如 ReLU 仅仅做一次计算，但是 GeLU 做 5 次计算；带宽瓶颈的时候，GeLU、ReLU 其实计算效率类似。

激活函数、点积等操作都是 `memory-bound`，对应的算术强度比较小，但是对于矩阵乘法情况可能不太一样，计算例子如下:

> `flops` 计算方式: $y[i][j] = \sum_{k=0}^{n-1} x[i][k] \times w[k][j]$，每一个输出元素相当于 $n$ 次乘法，$n-1$ 次加法

```python
n = 1024
x = torch.ones(n, n, dtype=torch.bfloat16, device="cuda:0")
w = torch.ones(n, n, dtype=torch.bfloat16, device="cuda:0")
y = x @ w

bytes = (2 * n * n) + (2 * n * n) + (2 * n * n) # read x, read w, write y
flops = n * n * (2 * n - 1) # 对于每一个输出元素，n 次乘法，n-1次加法

arithmetric_intensity = flops / bytes # n / 3
```

- 训练和推理过程中可能瓶颈不太一样，例如对于大模型，推理过程可能就是使用一个矩阵和原来的模型中的矩阵进行相乘操作，但是训练过程还涉及到反向传播等操作

- `Roofline` 图: 可以理解为算术强度和加速器强度之间的关系，对于不同硬件、不同算法对应的瓶颈可能不同

![Roofline 图：横轴为算术强度，纵轴为 FLOPs/s（对数尺度）。粉红色曲线代表加速器峰值 FLOPs/s，蓝色曲线代表实际 FLOPs/s。图中用不同颜色区域标注了带宽瓶颈（BW1 & 2）、计算瓶颈（BW2）等情况，还标注了 Algo 1 和 Algo 2 两种算法对应的瓶颈位置。](assets/images/lecture2-roofline.png)

- 以矩阵乘法为主体的层(例如全连接、卷积)，反向传播的计算次数一般是前向传播的两倍。对于线性层，例如 $y = Wx$，前向传播的过程中，每一个参数计算一次乘法 + 一次加法；反向传播过程中需要计算两次梯度，$\frac{\partial L}{\partial W} = (\frac{\partial L}{\partial y})^T @ x$、$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} @ W$，相当于两次矩阵乘法操作。所以此时前向传播 `FLOPs` 次数为 2，反向为 4，为两倍关系；但是如果不是矩阵乘法为主体的网络情况就会不太一样了

- 训练过程中参数计算: 网络参数 + 激活值参数 + 梯度参数 + 优化器状态参数(不同优化器对应的参数不同)
- 模型训练需要考虑 `batch` 大小，`batch` 大小直接决定了参数在内存中的占用量等指标
- 激活检查点/梯度检查点: `torch.utils.checkpoint.checkpoint` **可以使用"反向时多算一遍前向"的计算开销换取"不必保存所有中间激活的显存开销"，计算换空间，同时为了更好 trade-off，可以设置保存一定层的激活值**
