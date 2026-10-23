# Lecture5: GPUs、TPUs

## GPU Architecture

### GPU Hardware Arch

- `CPU`: 主要是需要执行指令(包括复杂分支、流程等)，所以 `CPU` 中包含比较多的控制单元和少量用于计算的 `ALU` 组件等，同时 `CPU` 关注延时的优化
- `GPU`: 主要用于计算，把海量计算核心放入到芯片中，并且可以大量并行执行计算，同时 `GPU` 关注吞吐量的优化

![CPU vs GPU Architecture](assets/images/lecture-05-img-01.png)

`GPU` 核心为 `SM`，同时每一个 `SM` 内部存在多个 `SPs` (流式处理器) 可以并行执行线程，`GPU` 含有多个 `SM` (`streaming multiprocessors`)，同时这些 `SM` 可以独立执行块(也就是任务)，同时 `SM` 有共享内存

![图片展示了GPU硬件架构中SM（流式多处理器）与GA100全GPU（含128个SM）的结构。左侧是SM的结构图，每个SM包含多个SPs（流式处理器），可并行执行线程。右侧是GA100全GPU结构图，GPU含有多个独立执行块（任务）的SM。图片与上下文紧密相关，直观呈现了GPU中SM的组成及GPU整体结构，帮助理解GPU如何通过大量并行执行计算来优化吞吐量。](assets/images/lecture-05-img-02.png)

同时类似于 `CPU`， `GPU` 也存在多级缓存结构: `Shared Memory` -> `L1 Cache` -> `L2 Cache` -> `Global Memory`，从上到下，读时延增大、容量增大、成本降低，我们平常说的显存实际上就是表示 `Global Memory`，`GPU` 整体结构如下:

![这张图展示了GPU的硬件架构相关内容，配有文字说明与表格信息，明确存储部件离SM越近访问速度越快。图中包含的文字信息提及，L1与共享内存位于SM内部，L2缓存在芯片上，全局内存是紧邻GPU的存储芯片；下方表格“TABLE IV”列出了不同内存类型对应的CPI（周期数），全局内存为290、L2缓存为200、L1缓存为33、共享内存为(23/19)；右下角是英伟达GA100 7nm TSMC制成的GPU芯片示意图，标注了各分区与尺寸参数，左下角则是GPU实物图。此外文字还补充说明，SRAM（共享/缓存内存）比DRAM（全局内存）昂贵100倍，但速度快8倍。](assets/images/lecture-05-img-03.png)

### TPU Hardware Arch(Intro)

`TPU` 核心组件(`Scalar Unit`、`VPU`、`MXU`等)如下:

![图片展示了TPU TensorCore的抽象布局，包含Scalar Unit、Vector Unit（VPU）、Matrix Multiply Unit（MXU）等核心组件。Scalar Unit类似CPU调度指令，VPU执行元素操作，MXU执行矩阵乘法，是芯片FLOP/s的驱动。右侧是HBM，存储权重、激活、优化状态等数据，其带宽决定了数据传输速度。图片与上下文关系紧密，直观呈现TPU中各组件功能及与HBM的关系，辅助理解TPU硬件架构。](assets/images/lecture-05-img-04.png)

`TPU` 中每一个相关的概念在 `GPU` 中都存在对应的组件，对应关系如下:

> 注意 `Tensor Core` 的含义，`TPU` 中表示控制器、`GPU` 中表示矩阵乘法单元

![这张图片聚焦于GPU与TPU的硬件结构对比，开篇提及“Side thread – What about TPUs?”，明确对比二者的核心组件对应关系。上侧表格分别列出GPU的对应部件、TPU的对应部件及其功能，且强调Tensor Core在TPU中是控制器，在GPU中是矩阵乘法单元。下侧表格则进一步对比了GPU和TPU部分核心组件的数量差异，包括SM（流式多处理器）、Warp Scheduler、SMEM（L1缓存）等部件的数量，图片底部总结二者核心特性为“Core structure – lightweight control, fast (big) matmul unit, fast memory”。](assets/images/lecture-05-img-05.png)

### GPU Program Model

**`GPU` 编程模型**中存在三个关键概念:

- **`Thread`**: 线程，`GPU` 上的最小执行单元，有自己的寄存器、局部变量和程序计数器，负责执行计算任务
- **`Block`**: 块，可以看成线程组，同一个 `Block` 上的线程分配到一个 `SM` 上执行，同时 `block` 内的线程可以共享 `shared memory`，并且 `block` 内可以同步，一个 `SM` 中可以存在多个 `block`
- **`Wrap`**:  32 个连续编号的线程组成的一个线程组，是调度的基本单位，同一个 `Wrap` 中的线程执行相同的指令，并且收到控制器的统一调度

> 同时注意到 `Wrap` 一定来自于同一个 `Block`，并且运行在同一个 `SM` 上，不会出现跨 `Block` 的情况

![这张图展示了CUDA编程模型中GPU线程、块、线程束及调度的关联关系，左侧显示CUDA应用每个块含256个线程，如Block 0、Block 1、Block 4096这类块会被分配到SM上执行；中间部分说明每个块会划分为线程束（Warp），每个线程束包含32个连续线程，且线程束一定来自同一个块；右侧体现每个SM配有4个线程束调度器，负责调度对应线程束，同一线程束内的线程执行相同指令，且不同的线程束会由对应调度器管控，体现了GPU的SIMT调度逻辑。](assets/images/lecture-05-img-06.png)

**`GPU` 内存模型中包含 `register`、`local memory`、`shared memory`、`global memory`、`constant memory` 等组件**，并且这些存储组件是不同层次的，有些是 `Thread` 维度的（`register`、`local memory`)，有些是 `Block` 维度的(`shared memory`)，另外有些是 `grid` 维度的(`global memory`等)，**同时使用 `Host Code` 可以实现主机内存和每一个 `grid` 的 `global memory` 或者 `constant memory` 之间的数据交换**

![这张图展示了GPU的程序模型与内存模型相关内容，左侧列出了设备代码与主机代码的功能：设备代码可读写线程的寄存器、局部内存，读写块的共享内存，读写网格的全局内存，仅读取网格的常量内存；主机代码可实现数据在自身与网格的全局内存、常量内存间传输。右侧对应呈现了GPU的网格结构，一个网格包含两个块，每个块内有寄存器、共享内存，以及不同线程，而全局内存与常量内存是块、线程均可共享的上层存储，主机可与这两类存储完成数据交互，直观对应了文档中GPU内存分层与程序模型的核心概念。](assets/images/lecture-05-img-07.png)

`GPU Model` 存在如下优点:

- 扩展性好: 容易通过添加更多 `SM` 的方式来提升硬件工作负载
- **`SIMT` 模型**： 易于编程
- `Thread` 轻量化，可以随时停止和启动

> **`SIMT` 是 `GPU` 执行模型: 一条指令由硬件同时是加到一组线程(`wrap`)上，同时每个线程都有自己的寄存器和数据，但共享一条数据流 (一组指令操作一组线程，每个线程都有自己的标量数据，程序员不需要手写向量化)**

![图片是一张表格，对比了SIMD、SIMT、MIMD、SPMD四种编程模型。SIMD由程序员负责向量化，编程方式为手写向量类型/ intrinsics，典型硬件是CPU向量单元；SIMT由硬件负责，编程方式为写标量线程代码，典型硬件是GPU；MIMD由程序员/OS负责，编程方式为多线程/多进程，各跑各的，典型硬件是CPU多核；SPMD在编程模型层面，编程方式为同一份程序，不同数据，典型硬件是CUDA/MPI。该表与上下文介绍的GPU执行模型等内容相关，帮助理解不同编程模型的特点。](assets/images/lecture-05-img-08.png)

同时 `GPU` 有特定的矩阵乘法单元，使得在 `GPU` 上矩阵乘法的运算速度要远远快于其他浮点操作

**`GPU` 算力、内存、通信**发展:

> 早期算力是瓶颈，我们可能不太需要考虑显存的问题；但是随着算力提升，后期显存和通信可能成为瓶颈，我们需要重点考虑这些因素(主要是显存大小和通信效率之间的优化)

![这张图片展示了1996年至2023年间峰值硬件FLOPS、内存带宽与互联带宽的归一化缩放趋势。图中黑色散点呈现硬件算力（HW FLOPS）的增长态势，其每两年提升3.0倍，20年间增长达60000倍，且整体增速远高于内存带宽与互联带宽。绿色线条代表DRAM带宽，每两年提升1.6倍，20年间增长100倍，蓝色线条代表互联带宽，每两年提升1.4倍，20年间增长30倍，三类指标均呈现随时间线性增长的趋势。](assets/images/lecture-05-img-09.png)

## Making ML workloads fast on GPU

`rootfline model` 表示了每个字节操作次数(对于各种算法而言，就是算法强度)和硬件吞吐量之间的关系，例如:

![图片展示了GPU和CPU在不同运算强度下的吞吐量情况，即roofline模型。横轴为运算强度（flops/byte），纵轴为吞吐量（gflops）。图中用不同颜色的线分别表示GPU寄存器、共享内存、主内存以及CPU ALU的吞吐量。蓝色点代表密集矩阵乘法和稀疏矩阵乘法的运算强度和吞吐量。该图与上下文紧密相关，用于说明在不同运算强度下，算法强度和硬件吞吐量之间的关系，以及如何通过优化算法强度来提升硬件吞吐量。](assets/images/lecture-05-img-10.png)

> 斜线部分: 算力没有充分利用，提升算法强度可以提升硬件吞吐量
> 
> 平线部分:  显存瓶颈，提升算法强度也无法提升硬件吞吐量
> 
> 为了尽可能提高硬件吞吐量，我们应该尽可能让算法落在平线的部分

接下来介绍几种优化显存的几种 `trick`

### Control deivergence(not a memory issue)

考虑到 `GPU` 中的 `SIMT` 模型，**一个 `wrap` 中的 32 个线程共享一个程序计数器**，也就是每一个周期内 `warp scheduler` 发射一条指令，所有 `Thread` 同时执行，对于如下的条件分支代码:

```C
__global__ void k(float *a, float *b, int n) {
    int i = blockIdx * blockDim.x + threadIdx.x;
    if (a[i] > 0) {
        b[i] = sqrtf(a[i]);
    } else {
        b[i] = -a[i];
    }
}
```

注意到上述代码中，**由于所有的 `Thread` 共享一个程序计数器，所以两条分支都要执行，但是存在一部分 `Thread` 没有命中对应的分支，这一部分 `Thread` 就会处于等待状态**，从而增加了代码执行时延，所以我们应该避免这一种情况，这些操作尽可能使用矩阵乘法完成

同时除了条件分支还存在其他可能引起发散的情况:

> 发散: 同一个 `wrap` 内的线程执行不同的代码导致相互等待的情况

![图片展示了GPU编程中可能引起发散的几种情况。包括ifelse发散，如if(a\[i\] > 0) ... else ...；循环发散，如while (a\[i\] > 0) a\[i\]--；提前退出，如if (i >= n) return；函数调用发散，不同线程调用不同函数；switch发散，多分支。这些发散情况会导致同一wrap内的线程执行不同代码，相互等待，增加代码执行时延，因此在GPU编程中应避免这些发散情况，尽可能使用矩阵乘法等操作。](assets/images/lecture-05-img-11.png)

### Trick1: Low precision computation(低精度计算)

使用低精度的数据格式可以提高算术强度，例如对于 $ReLU = \max(0, x)$，每一次内存访问为(`1 read、1 write`)，同时进行 `1 FLOP` 的浮点运算，如果使用 `FP32` 那么算术强度为 $\frac {FLOP}{8 bytes}$，如果使用 `FP16`，那么算术强度为 $\frac {FLOP}{4 bytes}$

对于 `Tensor Core`，基本指令为 $D = A \times B + C$， $A、B$为低精度矩阵(例如 `FP16` 等)， $C、D$为累加器，通常是 `FP32`，同时对于不同的操作我们也需要考虑使用不同的精度:

- 可使用 `16-bits` 存储(`FP16 / BF16`): 矩阵乘法；元素级别操作(`relu、add、sub`等)
- 需使用高精度存储(`FP32`):  小值加到大值上，避免精度损失；`softmax`、正则化等可能造成精度损失的操作；元素级别操作(`exp`、`log`、`pow`，一般是非线性操作)

同时除了传统的数据格式(`FP16`/`BF16`等)，还存在 `MXFP8` / `MXFP4` 等新型数据格式，例如 `MX` 是一种块级的共享指数 的低精度格式: 

> **每32 个元素共享一个 8 位的指数，元素本身是 `FP8/FP6/FP4/INT8`，其中 `MXFP8` = 32 个 `E4M3/E5M2` + 1 个共享 `scale`; `MXFP4` = 32 个 `E2M1` + 1 个共享 `scale`**

![这张图展示了Blackwell架构下MXFP8的缩放机制，左侧为常规FP8数据格式，每个数据元素对应独立的缩放因子，右侧为MXFP8数据格式，32个元素共享一个缩放因子，以此说明块级共享指数的低精度计算方式。图中还呈现了前向传播与反向传播流程，前向传播中，高精度数据经Cast模块转为half精度后完成矩阵乘法；反向传播则包含两个矩阵乘法分支，权重信息参与运算，相关结果经Cast模块转回高精度，直观体现了MXFP8在深度学习计算流程中的应用逻辑。](assets/images/lecture-05-img-12.png)

> 虽然 `MXFP8` 可以有效降低矩阵乘法计算开销，但是其转置等操作可能存在一定问题

### Trick2: Operator fusion(算子融合)

`GPU` 在进行每一次计算，首先需要从显存读取数据，计算完成之后会把数据写入到显存中，所以这里如果对于一个复杂的操作，每一个子操作都需要进行显存的读取和访问可能导致内存带宽瓶颈(可能出现通信瓶颈)，所以这里一个比较自然的想法就是，**几个算子融合成为一次运算，统一进行数据读取和写入，从而减少带宽消耗，这就是算子融合；**简单的算子融合依赖编译器就可以实现、复杂的算子融合可能需要变成实现(`torch.compile`)

![图片](assets/images/lecture-05-img-13.png)

### Trick3: recomputation

**重计算的思想是 计算和显存 之间的 `trade-off`**，在计算资源非常多、显存资源比较昂贵的情况下，我们可以通过重复计算的方式来减少中间结果的存储，从而减少显存的利用； 例如下图中，输入值 $x$ 经过三个激活函数 `sigmoid` 从而得到输出值，对于这个输出值，我们反向传播的时候需要使用到两个中间值 $s_1、s_2$，此时我们可以考虑丢弃激活值，并且当需要使用这些激活值的时候，重新做一次前向传播计算，从而减少显存利用

<div class="feishu-grid" style="grid-template-columns: 0.5fr 0.5fr;">

<div class="feishu-column">

![存储激活值，前向反向过程中 8 次读写](assets/images/lecture-05-img-14.png)

</div>

<div class="feishu-column">

![丢弃激活值，前向反向过程中 5 次读写](assets/images/lecture-05-img-15.png)

</div>

</div>

### Trick4: Memory coalescing and DRAM(合并访存)

显存主要使用 `DRAM`，类似于局部性原理，也就是一次读取操作并不是读取一个元素，而是会读取一块连续内存块中的所有元素(例如 128 bytes)，所以考虑局部性原理，我们最好尽可能让一个 `wrap` 中的所有线程在读取的时候，读取的元素都在一个突发区间，此时只需要一次内存访问就可以读取到所有需要的元素

![图片](assets/images/lecture-05-img-16.png)

> 例如，我想要读取矩阵中某一列的数据，如果将线程下标映射到每一行，也就是每一个线程读取每一行，那么就会导致每一个线程操作的数据不在一次合并访存内部，所以此时每一次请求相当于都要读取整个矩阵；另外一种情况，我们可以尝试将矩阵转置存储，然后读取每一行，这样每一个线程操作的元素在一次合并访存内部，从而减少了内存读取的次数

### Trick5: tiling(分块)

分块也是一种减少内存读取次数的方式(有点类似于 csapp 里面的 malloclab)，这里我们不妨以矩阵分块为例，正常情况下，对于两个大小为 $N \times N$ 的矩阵乘法，其中一个元素我们需要读取 $N$ 次，但是此时我们如果考虑将两个矩阵都切分为大小为 $T \times T$ 大小的块，那么相当于对于每一个元素我们只需要读取 $\frac {N}{T}$ 次，从而有效降低了内存访问的次数

![图片](assets/images/lecture-05-img-17.png)

但是矩阵分块可能存在一些问题，例如可能出现**矩阵大小不规整，导致存在多种分块方案**，此时我们可能需要通过不同大小的分块实验来找到最佳的分块大小(例如 `PyTorch` 中有对应的编译选项)

![图片](assets/images/lecture-05-img-18.png)

另外一个问题是**合并访存**，例如下图中，如果我想要读取蓝色行，第一种情况下，只需要读取四次即可；但是第二种情况下，蓝色方块出现了错位的情况，由于合并访存，所以此时我们可能读取大量不需要的数据(图中阴影部分)，所以这种情况下，我们可能需要考虑使用填充等方式来优化合并访存

![图片](assets/images/lecture-05-img-19.png)

**矩阵大小和 `FLOPs/s` (吞吐量)** 之间关系区间如下:

![图片](assets/images/lecture-05-img-20.png)

> 对于上述散点图，我们可以做如下解释:
> 
> - 随着矩阵大小增大，算法算术强度(`compute intensity / memory` 增大)，未达到内存瓶颈之前，吞吐量提升
> - 不同的矩阵大小分块方案不同(例如奇数倍数和16、32倍大小矩阵分开方案不同)，特定倍数大小的矩阵，可能存在一些比较优化的方块方案，所以吞吐量高，所以可能出现起伏突变的情况

<div class="feishu-grid" style="grid-template-columns: 0.5fr 0.5fr;">

<div class="feishu-column">

![矩阵大小不同导致对齐方案不同、吞吐量不同](assets/images/lecture-05-img-21.png)

</div>

<div class="feishu-column">

![矩阵大小变化 -> 分块方案变化 -> 吞吐量变化](assets/images/lecture-05-img-22.png)

</div>

</div>

## FlashAttention Intro

> 仅仅是简单介绍 `FlashAttention`，作业时再具体学习

Attention 计算公式:

$$Attn(Q,K,V) = softmax(\frac {QK^T}{\sqrt {d_k}}) V$$

包含三次矩阵乘法(输出还需要映射一次)和一次 Softmax 操作，所以我们考虑加速矩阵乘法和Softmax操作

- **矩阵乘法**: 主要还是使用分块的方式进行优化

![图片](assets/images/lecture-05-img-23.png)

- **Softmax**: online softmax 一次遍历，遍历过程中对于最大值进行修复，同时对于指数和进行在线求解，从而减少了内存访问次数，同时更加利于分块、流式计算等

<div class="feishu-grid" style="grid-template-columns: 0.5fr 0.5fr;">

<div class="feishu-column">

![图片](assets/images/lecture-05-img-24.png)

> Normal Softmax: 第二步依赖第一步的值，需要首先遍历一行并且存储对应的最大值和指数和结果才能进行下一步

</div>

<div class="feishu-column">

![图片](assets/images/lecture-05-img-25.png)

> Online Softmax: 同步计算最大值和指数和，利于分块、流式计算

</div>

</div>

FlashAttention 整体前向传播过程如下(核心是分块、online softmax)，同时反向传播过程中还通过重计算的方式来减少显存访问次数:

![图片](assets/images/lecture-05-img-26.png)
