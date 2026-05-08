# D2L动手学深度学习 - 学习笔记问答整理

> 本文档整理了学习《动手学深度学习》过程中的所有提问和详细解答

---

## 目录

- [第一章：预备知识](#第一章预备知识)
  - [微积分 (calculus.ipynb)](#微积分-calculusipynb)
  - [自动微分 (autograd.ipynb)](#自动微分-autogradipynb)
  - [概率论 (probability.ipynb)](#概率论-probabilityipynb)
  - [查找API (lookup-api.ipynb)](#查找api-lookup-apiipynb)
- [第二章：线性神经网络](#第二章线性神经网络)
  - [引言 (index.ipynb)](#引言-indexipynb)
  - [线性回归基础 (linear-regression.ipynb)](#线性回归基础-linear-regressionipynb)
  - [线性回归从零实现 (linear-regression-scratch.ipynb)](#线性回归从零实现-linear-regression-scratchipynb)
  - [线性回归简洁实现 (linear-regression-concise.ipynb)](#线性回归简洁实现-linear-regression-conciseipynb)
  - [线性回归简洁实现 (linear-regression-concise.ipynb)](#线性回归简洁实现-linear-regression-conciseipynb)
  - [softmax回归理论 (softmax-regression.ipynb)](#softmax回归理论-softmax-regressionipynb)

---

## 第一章：预备知识

### 微积分 (calculus.ipynb)

#### Q1: 梯度的概念没看懂

**梯度** = 导数在多维空间的推广 = 最优化方向的指南针

**核心思想：** 梯度告诉你往哪个方向走，函数值增长最快

```
        损失函数曲面（像个碗）

        L(θ)
         ↑
         │    梯度 ↑（往上最快）
         │       │
         │       │  我们要走 ↓
         │       │  的方向
         │    ●──┘
         │   /│
         │  / │
         └──────────→ θ
```

**梯度公式：**
$$\nabla_{\mathbf{x}} f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_n}\bigg]^\top$$

---

#### Q2: 梯度公式中的四个规则

| 公式 | 人话解释 |
|------|---------|
| ∇x **A**x = **A**ᵀ | 矩阵乘向量 x，求 x 的梯度 → 等于矩阵转置 |
| ∇x xᵀ**A** = **A** | 向量转置乘矩阵，求 x 的梯度 → 等于矩阵本身 |
| ∇x xᵀ**A**x = (**A**+**A**ᵀ)x | 二次型（常见的损失函数形式）的梯度 |
| ∇x \|\|x\|\|² = 2x | 向量长度的平方，梯度是 2倍向量本身 |

**数值例子验证：**

```python
x = [1, 2]ᵀ
A = [[3, 4],
     [5, 6]]

# 公式4验证
||x||² = x₁² + x₂² = 1 + 4 = 5
梯度 = [2x₁, 2x₂] = [2, 4] = 2x ✓
```

---

#### Q3: 梯度的意义

| 概念 | 回答的问题 |
|------|-----------|
| 期望 | "数据平均在哪里？" |
| 方差 | "数据有多分散？" |
| 梯度方向 | "往哪走函数增加最快？" |
| 梯度大小 | "坡有多陡？" |

---

### 自动微分 (autograd.ipynb)

#### Q1: 非标量变量的反向传播没看懂

**核心问题：** `backward()` 只能对**标量**调用

```python
x = torch.arange(4.0)  # x = [0, 1, 2, 3]
y = x * x              # y = [0, 1, 4, 9]（向量，不是标量！）
```

**矛盾：** `y.backward()` 只接受标量，但 `y` 是向量。怎么办？

**两种解决方案：**
```python
# 方法1：先求和变成标量，再反向传播
y.sum().backward()

# 方法2：传入 gradient 参数
y.backward(torch.ones(len(x)))  # 传入全1向量作为权重

# 这两种方法效果相同！
```

---

#### Q2: y = x * x = [0, 1, 4, 9] 为什么？

`x * x` 是逐元素相乘：

```python
x = [0, 1, 2, 3]

# x * x 不是矩阵乘法，而是每个元素自己乘自己
位置 0:  x[0] × x[0] = 0 × 0 = 0
位置 1:  x[1] × x[1] = 1 × 1 = 1
位置 2:  x[2] × x[2] = 2 × 2 = 4
位置 3:  x[3] × x[3] = 3 × 3 = 9

结果: y = [0, 1, 4, 9]
```

| 操作 | 符号 | 含义 | 例子 |
|------|------|------|------|
| 逐元素乘 | `x * x` | 每个元素单独乘 | `[0,1,2]*[0,1,2] = [0,1,4]` |
| 点积 | `torch.dot(x,x)` | 对应位置乘后求和 | `[0,1,2]·[0,1,2] = 5` |

---

#### Q3: detach() 的作用

**detach() = 把 y 从计算图中"剪断"**

```python
x = [0, 1, 2, 3]
y = x * x        # y = [0, 1, 4, 9]
u = y.detach()   # u = [0, 1, 4, 9]，但 u 和 x "断开"了！
z = u * x        # z = [0*0, 1*1, 4*2, 9*3] = [0, 1, 8, 27]

# 计算 z 对 x 的梯度时，u 被当作常数，不追溯回 x
# 梯度 = u = [0, 1, 4, 9]
```

**计算图对比：**
```
没有 detach：
  x ──→ y=x² ──→ z=y*x=x³
  梯度流向：z → y → x
  结果：∂z/∂x = 3x²

有 detach：
  x ──→ y=x² ──→ u（剪断！）──→ z=u*x
  x ──────────────────────→ z
  梯度流向：z → x（不经过 y）
  结果：∂z/∂x = u（把 u 当常数）
```

---

#### Q4: Python控制流的梯度计算 - 为什么梯度 = d/a？

**关键点：** 无论走哪条路径，结果都是 a 的倍数

```python
a = 1.5
b = a * 2 = 3

# while循环（每次翻倍）
第1轮: b = 3 * 2 = 6      = 1.5 * 2^2
第2轮: b = 6 * 2 = 12     = 1.5 * 2^3
...
第8轮: b = 768 * 2 = 1536 = 1.5 * 2^9

# 最终
d = 1536 = 1.5 * 2^9 = a * 1024

# 求导数
对于函数 d = k * a：
∂d/∂a = k = 1024 = d/a
```

---

#### Q5: item()是什么？

**`.item()` = 把单个数的 tensor 转成 Python 的普通数字**

```python
x = torch.tensor(3.14)

print(x)        # tensor(3.14)  ← PyTorch tensor 类型
print(x.item()) # 3.14          ← Python float 类型
```

| 类型 | 写法 | 能做什么 |
|------|------|---------|
| tensor | `x = torch.tensor(3.14)` | 可以求梯度、GPU运算 |
| Python数值 | `x.item()` | 普通数字，可以打印、存列表 |

---

#### Q6: grad_vals循环代码的意思

```python
grad_vals = []
for x_val in x_vals:
    x = torch.tensor(x_val.item(), requires_grad=True)
    y = torch.sin(x)
    y.backward()
    grad_vals.append(x.grad.item())
```

**逐行解释：**
- 第1行：创建空列表，存梯度结果
- 第2行：遍历每个 x 值
- 第3行：创建可求导的 tensor（`requires_grad=True` 开启求导）
- 第4行：计算 sin(x)
- 第5行：反向传播，计算梯度 = cos(x)
- 第6行：保存梯度到列表

---

#### Q7: ?和??的使用

**只能在 Jupyter Notebook / IPython 环境中使用**

| 指令 | 作用 | 显示位置 |
|------|------|---------|
| `torch.ones?` | 查看文档 | 弹出窗口 |
| `torch.ones??` | 查看源代码 | 弹出窗口 |

**替代方案：** 在普通Python终端用 `help(torch.ones)` 代替

---

### 概率论 (probability.ipynb)

#### Q1: fair_probs和multinomial代码没看懂

```python
fair_probs = torch.ones([6]) / 6
multinomial.Multinomial(1, fair_probs).sample()
```

**解析：**
```python
fair_probs = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]  # 骰子6个面，每面概率相等

# multinomial.Multinomial(1, fair_probs).sample()
# 意思：掷骰子1次
# 输出类似：tensor([0., 0., 1., 0., 0., 0.])
#         表示第3面出现了

# 掷10次
multinomial.Multinomial(10, fair_probs).sample()
# 结果：tensor([2., 3., 1., 1., 2., 1.])
#       面1出现2次，面2出现3次...
```

---

#### Q2: counts和cumsum代码

```python
counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
```

**解析：**
```python
# 第1行：500组实验，每组掷10次骰子
# counts 是500×6的矩阵

# 第2行：cumsum(dim=0) 沿着行方向累积求和
# 第1行累积 = 第1组
# 第2行累积 = 第1组 + 第2组
# ...

# 第3行：计算累积概率估计
# cum_counts / 总次数 = 相对频率
```

---

#### Q3: dim=1, keepdims=True是什么意思

```python
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

# dim=1 → 沿着列方向（横着）操作，每行求和
x.sum(dim=1)           # tensor([6, 15])    shape: (2,)
x.sum(dim=1, keepdims=True)  # tensor([[6], [15]])  shape: (2, 1)
```

| 参数 | 含义 |
|------|------|
| `dim=1` | 沿着第1维（列方向）操作 |
| `keepdims=True` | 保持输出维度，方便后续广播运算 |

---

#### Q4: 概率论章节核心概念总结

| 标题 | 核心内容 |
|------|---------|
| 基本概率论 | 大数定律：实验越多，频率→概率 |
| 概率论公理 | 概率的三条基本规则 |
| 随机变量 | 离散 vs 连续 |
| 联合概率 | 多事件同时发生 |
| 条件概率 | 已知A发生后B的概率 |
| 贝叶斯定理 | 条件概率反转 |
| 边际化 | 消掉某些变量 |
| 独立性 | 事件互不影响 |
| HIV应用 | 贝叶斯实战案例 |
| 期望和方差 | 描述分布的中心和离散程度 |

---

### 查找API (lookup-api.ipynb)

#### Q1: dir()函数的作用

```python
import torch
print(dir(torch.distributions))
```

**输出所有属性名：**
```
['Bernoulli', 'Beta', 'Binomial', 'Categorical', 'Normal', ...]
```

| 函数 | 作用 |
|------|------|
| `dir(module)` | 列出所有属性 |
| `help(func)` | 查看函数参数和用法 |
| `func?` | Jupyter中查看文档 |
| `func??` | Jupyter中查看源代码 |

---

## 第二章：线性神经网络

### 引言 (index.ipynb)

#### 章节结构

| 标题 | 核心内容 |
|------|---------|
| 引言开头 | 传统编程 vs 机器学习 |
| 日常生活ML | Siri、唤醒词识别 |
| 关键组件 | 数据、模型、目标函数、优化算法 |
| 各种ML问题 | 监督学习、无监督学习、强化学习 |
| 起源 | 统计学+信息论+神经科学 |
| 深度学习发展 | 数据爆炸+GPU算力 |
| 成功案例 | 语音识别、图像识别、游戏AI、自动驾驶 |
| 特点 | 多层表示学习、端到端训练 |

---

### 线性回归基础 (linear-regression.ipynb)

#### Q1: 线性回归的基本元素

| 概念 | 公式 | 含义 |
|------|------|------|
| 线性模型 | ŷ = wᵀx + b | 特征的加权和 |
| 权重w | 决定特征影响程度 | 面积权重大 → 面积更重要 |
| 偏置b | 所有特征为0时的预测 | 增强模型表达能力 |
| 损失函数 | ½(ŷ - y)² | 衡量预测误差 |

---

#### Q2: 解析解公式

$$\mathbf{w}^* = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}$$

**局限性：**
| 问题 | 说明 |
|------|------|
| 矩阵求逆计算量大 | 大矩阵求逆非常慢，O(d³)复杂度 |
| 矩阵可能不可逆 | XᵀX 可能奇异，无法求解 |
| 只适用于线性模型 | 深度神经网络没有解析解 |
| 内存问题 | 需要存储整个数据集矩阵 |

---

#### Q3: 随机梯度下降更新公式

$$\mathbf{w} \leftarrow \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)$$

$$b \leftarrow b - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)$$

**推导过程：**

```
损失函数：
  l⁽ⁱ⁾ = ½(wᵀx⁽ⁱ⁾ + b - y⁽ⁱ⁾)²

对w求偏导：
  ∂l⁽ⁱ⁾/∂w = x⁽ⁱ⁾ × (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾) = x⁽ⁱ⁾ × 误差

对b求偏导：
  ∂l⁽ⁱ⁾/∂b = (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾) = 误差

小批量平均后更新参数：
  w ← w - η × 平均梯度
  b ← b - η × 平均梯度
```

---

#### Q4: Timer类详解

```python
class Timer:
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()  # 记录起点时刻

    def stop(self):
        elapsed = time.time() - self.tik
        self.times.append(elapsed)  # 存储elapsed时间
        return elapsed

    def avg(self):
        return sum(self.times) / len(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()
```

| 方法 | 作用 |
|------|------|
| `start()` | 开始计时，记录起点 |
| `stop()` | 停止计时，返回elapsed时间 |
| `avg()` | 返回平均时间 |
| `cumsum()` | 返回累计时间列表 |

---

#### Q5: 正态分布可视化代码

```python
x = np.arange(-7, 7, 0.01)  # 生成1400个点（横轴）
params = [(0, 1), (0, 2), (3, 1)]  # 三种正态分布参数

[normal(x, mu, sigma) for mu, sigma in params]
# 列表推导式，计算三条曲线的y值
```

---

#### Q6: 似然函数公式推导

$$P(y|x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (y - \mathbf{w}^\top \mathbf{x} - b)^2\right)$$

**推导步骤：**
```
1. 噪声假设：ε ~ N(0, σ²)
2. 模型关系：y = wᵀx + b + ε
3. y的分布：y ~ N(wᵀx + b, σ²)
4. 正态分布公式代入均值μ = wᵀx + b
5. 得到似然函数
```

---

#### Q7: 线性回归作为神经网络

**层数计数规则：** 只计算有参数/计算的层（输入层不计）

```
线性回归：
  输入层 → 输出层（有参数w和b）
  层数 = 1（单层神经网络）
```

---

#### 练习解答

| 练习 | 核心结论 |
|------|---------|
| 练习1 | 最小化∑(x_i-b)² → b*=均值（正态分布MLE） |
| 练习2 | 解析解w=(XᵀX)⁻¹Xᵀy，适用于小规模、可逆情况 |
| 练习3 | 指数分布噪声 → L1损失，零点不连续 → 需要Huber损失解决 |

---

#### 练习详细解析

---

### 练习1：找到常数b使∑(x_i - b)²最小

#### 问题1.1：找到最优值b的解析解

**第一步：明确优化目标**

```
目标函数：
  L(b) = Σᵢ (x_i - b)²

含义：
  找一个常数b，使得所有数据点x_i偏离b的平方和最小

例子：
  数据：x = [1, 2, 3, 4, 5]

  如果选 b = 3：
    L(3) = (1-3)² + (2-3)² + (3-3)² + (4-3)² + (5-3)²
         = 4 + 1 + 0 + 1 + 4 = 10

  如果选 b = 2：
    L(2) = (1-2)² + (2-2)² + (3-2)² + (4-2)² + (5-2)²
         = 1 + 0 + 1 + 4 + 9 = 15（更大）
```

**第二步：求导**

```
L(b) = Σᵢ (x_i - b)²

展开求导：
  dL/db = Σᵢ d/db [(x_i - b)²]

利用链式法则：
  d/db [(x_i - b)²] = 2(x_i - b) × d/db(x_i - b)
                    = 2(x_i - b) × (-1)
                    = -2(x_i - b)

所以：
  dL/db = Σᵢ -2(x_i - b)
        = -2 (Σᵢ x_i - nb)

其中 n = 样本数量
```

**第三步：设导数为0求解**

```
最优解的条件：导数 = 0

-2 (Σᵢ x_i - nb) = 0

两边除以 -2：
  Σᵢ x_i - nb = 0

移项：
  nb = Σᵢ x_i

除以 n：
  b = Σᵢ x_i / n

这就是均值！
```

**第四步：验证**

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5])

# 解析解
b_optimal = np.mean(x)  # = 3.0

# 计算损失
L_optimal = np.sum((x - b_optimal)**2)  # = 10

# 尝试其他值验证
for b_test in [2.0, 2.5, 3.0, 3.5, 4.0]:
    L_test = np.sum((x - b_test)**2)
    print(f"b={b_test}, L={L_test}")

输出：
b=2.0, L=15
b=2.5, L=12.5
b=3.0, L=10    ← 最小！
b=3.5, L=12.5
b=4.0, L=15

确认 b=3（均值）是最优解
```

---

#### 问题1.2：与正态分布的关系

**第一步：理解正态分布的均值估计**

```
正态分布：
  X ~ N(μ, σ²)

概率密度函数：
  p(x) = 1/√(2πσ²) × exp(-1/2σ² × (x-μ)²)

问题：
  给定数据 x_1, x_2, ..., x_n
  假设它们来自正态分布 N(μ, σ²)
  如何估计均值 μ？
```

**第二步：写出似然函数**

```
假设：数据x_i来自 N(μ, σ²)

单个样本的概率密度：
  p(x_i|μ) = 1/√(2πσ²) × exp(-1/2σ² × (x_i - μ)²)

整体似然（所有样本的联合概率）：
  P(x|μ) = ∏ᵢ p(x_i|μ)
         = ∏ᵢ 1/√(2πσ²) × exp(-1/2σ² × (x_i - μ)²)
```

**第三步：取对数简化**

```
对数似然：
  log P(x|μ) = Σᵢ [log(1/√(2πσ²)) - 1/2σ² × (x_i - μ)²]

  = n × log(1/√(2πσ²)) - 1/2σ² × Σᵢ (x_i - μ)²

极大似然估计：
  找μ使 log P(x|μ) 最大

  等价于：
  找μ使 Σᵢ (x_i - μ)² 最小（因为第一项是常数）
```

**第四步：得出结论**

```
结论：

问题"最小化 Σ(x_i - b)²"等价于：
  用极大似然估计正态分布的均值

最优解：
  b* = x̄（均值）= μ̂（均值估计）

这就是为什么：
  → 均值是正态分布数据中心的最优估计
  → 平方损失来自正态分布的极大似然估计
```

---

### 练习2：线性回归解析解推导

#### 问题2.1：矩阵表示优化问题

**第一步：定义数据矩阵**

```
数据：
  X ∈ ℝ^(n×d)  → 特征矩阵（n个样本，d个特征）
  y ∈ ℝ^n      → 标签向量

示例：
  n = 3（3个样本）
  d = 2（2个特征）

  X = [[x₁₁, x₁₂],   ← 样本1的两个特征
       [x₂₁, x₂₂],   ← 样本2
       [x₃₁, x₃₂]]   ← 样本3

  y = [y₁, y₂, y₃]   ← 三个标签
```

**第二步：忽略偏置b**

```
技巧：
  把偏置b合并到权重向量中

方法：
  1. 在X中增加一列全为1
     X' = [X | 1] = [[x₁₁, x₁₂, 1],
                    [x₂₁, x₂₂, 1],
                    [x₃₁, x₃₂, 1]]

  2. 把b放到w末尾
     w' = [w₁, w₂, b]

  3. 现在：
     ŷ = X'w'
     = x₁₁×w₁ + x₁₂×w₂ + b

这样就包含了偏置！
```

**第三步：写出损失函数**

```
预测：
  ŷ = Xw（已合并偏置）

损失函数（平方误差）：
  L(w) = ½||y - Xw||²

展开：
  L(w) = ½(yᵀy - 2wᵀXᵀy + wᵀXᵀXw)
```

---

#### 问题2.2：计算梯度

```
L(w) = ½[yᵀy - 2wᵀXᵀy + wᵀXᵀXw]

逐项求导：

第1项：yᵀy → 不含w，导数为0

第2项：-2wᵀXᵀy → 导数 = -2Xᵀy

第3项：wᵀXᵀXw → 导数 = 2XᵀXw（二次型）

合并：
  ∂L/∂w = ½[0 - 2Xᵀy + 2XᵀXw]
        = -Xᵀy + XᵀXw
        = Xᵀ(Xw - y)
        = Xᵀ(ŷ - y)

这就是梯度！
```

---

#### 问题2.3：设梯度为0求解

```
设梯度为0：
  XᵀXw - Xᵀy = 0

  XᵀXw = Xᵀy

如果 XᵀX 可逆：
  w = (XᵀX)⁻¹Xᵀy

这就是解析解！

验证（简单例子）：
  X = [[1, 1],
       [1, 2]]
  y = [3, 5]

  w = (XᵀX)⁻¹Xᵀy = [1, 2]

  验证：
    Xw = [3, 5] = y ✓
```

---

#### 问题2.4：解析解 vs 梯度下降

**解析解更好：**
```
情况1：数据规模小（n和d较小）
情况2：XᵀX可逆且计算快
情况3：需要精确解（理论最优）
情况4：单次计算（不需要迭代）
```

**解析解失效：**
```
问题1：XᵀX不可逆
  → 特征之间有完全线性相关
  → 解决：加正则化（岭回归）

问题2：数据规模太大
  → 矩阵求逆O(d³)太慢
  → 解决：用梯度下降

问题3：非线性模型
  → 深度神经网络没有解析解
  → 必须用梯度下降

问题4：内存不够
  → 存储XᵀX需要O(d²)内存
  → 解决：用小批量梯度下降
```

---

### 练习3：指数分布噪声模型

#### 问题3.1：写出负对数似然

**第一步：理解指数噪声分布**

```
噪声模型：
  p(ε) = ½ exp(-|ε|)

这是拉普拉斯分布（双指数分布）

特点：
  → 在ε=0处峰值最高
  → 比正态分布有更重的尾部（对异常值更鲁棒）
```

**第二步：写出似然函数**

```
线性模型：
  y = wᵀx + b + ε

重写：
  ε = y - ŷ

单个样本的似然：
  p(y|x) = ½ exp(-|y - ŷ|)
         = ½ exp(-|误差|)

整体似然：
  P(y|X) = (½)^n × exp(-Σᵢ |y_i - ŷ_i|)
```

**第三步：取负对数**

```
负对数似然：
  -log P(y|X) = n log(2) + Σᵢ |y_i - ŷ_i|

去掉常数项：

最小化：
  Σᵢ |y_i - ŷ_i|

这就是绝对误差损失（L1损失）！
```

---

#### 问题3.2：解析解

**第一步：分析问题**

```
优化目标：
  最小化 L(w) = Σᵢ |y_i - wᵀx_i|

问题：
  绝对值函数 |z| 在 z=0 处不可导！
```

**第二步：单变量解析解**

```
简化问题：只估计常数b

L(b) = Σᵢ |x_i - b|

解析解：
  b* = 中位数（median）

数值例子：
  数据：x = [1, 2, 3, 4, 100]

  均值 = 22
  中位数 = 3

  均值的绝对误差：156
  中位数的绝对误差：101 ← 更小！

结论：
  → 中位数比均值更鲁棒（对异常值不敏感）
```

**第三步：多变量情况**

```
多变量线性回归：
  没有简单的解析解公式！

原因：
  → 绝对值函数不光滑
  → 在多个点不可导

解决方法：
  1. 转化为线性规划问题
  2. 使用迭代优化算法
```

---

#### 问题3.3：随机梯度下降及问题

**第一步：写出梯度**

```
损失函数：
  L(w) = Σᵢ |y_i - wᵀx_i|

梯度（在不为零的点）：
  ∂L/∂w = -Σᵢ x_i × sign(y_i - wᵀx_i)

  其中 sign(z) = +1 if z > 0
                = -1 if z < 0
```

**第二步：问题分析**

```
问题1：梯度在零点不连续
  → sign(error) 从 +1 跳到 -1
  → 更新方向不稳定

问题2：驻点附近震荡
  → sign(error) 来回跳变
  → w在最优解附近来回震荡
  → 无法精确收敛

问题3：学习率选择困难
  → 太小收敛慢
  → 太大震荡加剧
```

**第三步：解决方法**

```
方法1：Huber损失（平滑过渡）

定义：
  Lδ(a) = ½a²            if |a| ≤ δ
        = δ|a| - ½δ²     if |a| > δ

特点：
  → 误差小时用平方（光滑）
  → 误差大时用绝对值（鲁棒）

---

方法2：使用次梯度

技巧：
  设 sign(0) = 0
  → 梯度在零点为0
  → 参数不更新（停在最优解）

---

方法3：衰减学习率

公式：
  η_t = η₀ / (1 + t)

效果：
  → 学习率逐渐减小
  → 震荡幅度减小

---

方法4：批量梯度平均

技巧：
  用小批量的平均梯度
  → 梯度噪声减小
  → 更新更稳定
```

---

### 线性回归从零实现 (linear-regression-scratch.ipynb)

#### Q1: synthetic_data函数详解

```python
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))  # 生成正态分布特征
    y = torch.matmul(X, w) + b                      # 计算真实线性关系
    y += torch.normal(0, 0.01, y.shape)             # 添加噪声
    return X, y.reshape((-1, 1))
```

| 函数/变量 | 作用 |
|----------|------|
| `torch.normal()` | 生成正态分布随机数 |
| `torch.matmul()` | 矩阵乘法 |
| `reshape((-1,1))` | 变成列向量 |

---

#### Q2: data_iter函数详解

```python
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
```

**逐行解析：**

```python
# 第1行：获取样本数量
num_examples = len(features)  # 如1000

# 第2行：创建索引列表
indices = list(range(num_examples))  # [0, 1, 2, ..., 999]

# 第3行：随机打乱索引
random.shuffle(indices)  # [47, 23, 891, 12, 5, ...] 随机顺序

# 第4行：循环遍历，每次跳batch_size步
for i in range(0, num_examples, batch_size):  # i = 0, 10, 20, ...

# 第5-6行：取出当前批的索引
batch_indices = indices[i: i + batch_size]

# 第7行：yield返回一个批次
yield features[batch_indices], labels[batch_indices]
```

**yield vs return的区别：**

```python
# return：一次性返回所有结果
def get_all(): return [1, 2, 3, 4, 5]

# yield：每次返回一个结果（生成器）
def get_one_by_one():
    for i in [1, 2, 3, 4, 5]:
        yield i  # 每次只返回一个

# 使用yield的优势：
# → 按需生成，内存友好
# → 不需要一次性生成所有数据
```

---

#### Q3: 初始化模型参数详解

```python
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

**为什么权重用小随机数？**

```
如果权重全为0 → 对称性问题：
  → 每个输入对输出的影响相同
  → 梯度更新完全相同
  → 所有权重永远相同
  → 相当于只有1个有效参数！

随机初始化打破对称性：
  → 每个权重可以独立学习不同的值

为什么用小数值（0.01）？
  → 避免梯度爆炸
  → 数值范围合理
  → 训练初期稳定
```

**为什么偏置可以初始化为0？**

```
偏置不影响对称性：
  → 偏置只加到所有输出上
  → 不涉及权重之间的竞争
  → 训练过程会自动调整到合适值
```

**requires_grad=True的含义：**

```python
requires_grad = True → 需要计算梯度

含义：
  → PyTorch会跟踪这个张量的操作
  → 记录计算图
  → 可以调用.backward()计算梯度
  → 梯度会存储在.grad属性中
```

---

#### Q4: linreg模型函数详解

```python
def linreg(X, w, b):
    return torch.matmul(X, w) + b
```

**torch.matmul详解：**

```python
torch.matmul(X, w) → 矩阵乘法

假设：
  X.shape = (10, 2)  → 10个样本，每个2个特征
  w.shape = (2, 1)   → 2个权重

矩阵乘法：
  X × w = (10, 2) × (2, 1) = (10, 1)

  结果：10个预测值

数学展开：
  [x₁₁, x₁₂]   [w₁]   [x₁₁×w₁ + x₁₂×w₂]
  [x₂₁, x₂₂] × [w₂] = [x₂₁×w₁ + x₂₂×w₂]
```

**广播机制：**

```python
torch.matmul(X, w) + b

形状分析：
  matmul结果 → shape (10, 1)
  b → shape (1)

广播：
  b从(1)扩展成(10, 1)
  然后逐元素相加

例子：
  matmul结果 = [[3.5], [4.2], [5.1]]
  b = [0.5]

  广播后b变成 [[0.5], [0.5], [0.5]]

  相加结果 = [[4.0], [4.7], [5.6]]
```

---

#### Q5: squared_loss损失函数详解

```python
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
```

**为什么要reshape？**

```python
问题：y和y_hat形状可能不同！

例子：
  y_hat.shape = (10, 1)  → 模型输出，二维
  y.shape = (10,)        → 标签，一维

直接相减可能广播错误！

解决：
  y.reshape(y_hat.shape)

  把y从(10,) reshape成(10, 1)

  现在形状匹配：(10, 1) - (10, 1) ✓
```

**返回每个样本的损失（不是平均）：**

```python
squared_loss返回每个样本的损失：

y_hat.shape = (10, 1)
返回：loss.shape = (10, 1)

含义：
  → 每个元素是单个样本的损失
  → 后续需要手动计算平均

用途1：计算梯度（需要每个样本的误差）
用途2：调试和监控
用途3：加权损失
```

---

#### Q6: sgd优化算法详解

```python
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
```

**逐行解析：**

```python
# 第1行：with torch.no_grad()
# → 不跟踪梯度（更新操作不记录到计算图）
# → 防止计算图越来越大

# 第2行：for param in params
# → 遍历所有参数（如[w, b]）
# → 第1次循环：param = w
# → 第2次循环：param = b

# 第3行：param -= lr * param.grad / batch_size
# → SGD更新公式
# → lr → 学习率η
# → param.grad → 累加梯度
# → / batch_size → 平均梯度

# 第4行：param.grad.zero_()
# → 清零梯度（防止累加）
# → PyTorch默认累加梯度
# → 不清零下次梯度会叠加
```

**为什么除以batch_size？**

```
PyTorch的backward()默认累加梯度：
  param.grad = Σ ∂l⁽ⁱ⁾/∂param

数学上需要平均梯度：
  1/|B| × Σ ∂l⁽ⁱ⁾/∂param

所以代码中：
  param.grad / batch_size

实现平均梯度更新！

两种方式等价：
  方式1：loss.sum() + 除batch_size
  方式2：loss.mean()（已平均）
```

**参数遍历机制：**

```python
params = [w, b]

Python引用机制：
  params存储的是引用（指针）

执行：
  param = params[0]  → param指向w的内存地址

修改param：
  param -= update → 直接修改内存中的张量
  → w也被修改！（因为指向同一地址）

关键：-= 是原地操作
  → 不创建新张量
  → 直接修改原张量
```

---

#### Q7: 训练循环详解

```python
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```

**训练流程：**

```
外层循环（epoch）：
  → 每轮遍历整个数据集一次
  → epoch 0: 第1轮
  → epoch 1: 第2轮
  → epoch 2: 第3轮

内层循环（batch）：
  → data_iter每次返回一个batch
  → 每个batch更新一次参数

训练步骤：
  1. 前向传播：net(X, w, b) → 预测值y_hat
  2. 计算损失：loss(y_hat, y) → 每个样本损失
  3. 反向传播：l.sum().backward() → 计算梯度
  4. 更新参数：sgd([w, b], lr, batch_size)
  5. 清零梯度：在sgd中完成

epoch结束后评估：
  → 计算整体损失
  → 监控训练进展
```

---

#### Q8: backward()内部机制详解

**核心原理：计算图 + 链式法则**

```
PyTorch自动微分：

前向传播 → 构建计算图（记录每个运算）
反向传播 → 沿计算图反向遍历，用链式法则计算梯度
```

**前向传播构建计算图：**

```python
每个运算都创建新节点：

w → 节点w（requires_grad=True）
b → 节点b（requires_grad=True）

y_hat = matmul(X, w) + b → 创建节点y_hat，记录运算
error = y_hat - y → 创建节点error
l = error² / 2 → 创建节点l
L = l.sum() → 创建节点L（标量）

计算图结构：
  X, w, b → matmul+add → y_hat → subtract → error → pow/div → l → sum → L
```

**反向传播过程：**

```
从L开始反向遍历：

节点L = l.sum():
  上游梯度 = 1（初始）
  局部梯度 = dL/dl_i = 1
  下游梯度 = grad_l = 1

节点l = error² / 2:
  上游梯度 = 1
  局部梯度 = dl/derror = error
  下游梯度 = grad_error = error

节点error = y_hat - y:
  上游梯度 = error
  局部梯度 = derror/dy_hat = 1
  下游梯度 = grad_y_hat = error

节点y_hat = matmul(X, w) + b:
  上游梯度 = error
  局部梯度：
    dy_hat/dw = X
    dy_hat/db = 1
  下游梯度：
    grad_w = error × X
    grad_b = error

最终存储：
  w.grad = Σ error × X（累加）
  b.grad = Σ error（累加）
```

**grad_fn的作用：**

```python
每个节点的grad_fn定义局部梯度如何计算：

y_hat.grad_fn → <AddBackward0>
  → 定义：如何计算matmul和add的梯度

l.grad_fn → <MulBackward0>
  → 定义：如何计算乘法和幂的梯度

这些函数是PyTorch内置的，自动应用
```

**批量梯度累加：**

```
当batch_size = 10时：

每个样本产生梯度贡献：
  样本1：grad_w = error¹ × X¹
  样本2：grad_w = error² × X²
  ...

累加结果：
  w.grad = Σ error⁽ⁱ⁾ × X⁽ⁱ⁾

这就是Σ ∂l/∂w！
```

---

#### Q8: backward()内部机制再解释（通俗版）

**核心问题：梯度是怎么算出来的？**

```
答案：PyTorch帮你自动算，你只需要调用backward()。
```

**最简单例子演示：**

```python
# 最简单的例子
x = torch.tensor([2.0], requires_grad=True)  # x=2，需要求梯度
y = x ** 3                                    # y = x³ = 8

y.backward()  # 调用backward

print(x.grad)  # 输出：tensor([12.])
              # 12 = 3 × 2² = 3x² ✓ 数学验证
```

**backward()做了什么？** 它自动帮你算出了 3x² = 12

---

**PyTorch是怎么做到的？**

```
秘密：记录你做了什么运算

你写代码：
  y = x ** 3

PyTorch偷偷记录：
  "y是通过把x做3次方得到的"

  存储在：y.grad_fn = PowBackward0

当你调用backward()时：
  PyTorch问grad_fn："这个运算的梯度是多少？"

  grad_fn回答："x³的梯度是3x²，把x=2代入，得到12"

  PyTorch把12存到x.grad里
```

---

**再复杂一点的例子：**

```python
x = torch.tensor([2.0], requires_grad=True)

# 多步运算
a = x + 1     # a = 3
b = a * 2     # b = 6
y = b ** 2    # y = 36

y.backward()

print(x.grad)  # 输出：tensor([24.])
```

**PyTorch怎么算出来的？**

```
它记录了整个运算链：

x → x+1 → a → a×2 → b → b² → y

每一步都记录了：

x → a:  "a = x + 1"
a → b:  "b = a × 2"
b → y:  "y = b²"

反向时，从y往回走：

第1步：y = b²
  问："b²对b的导数是什么？"
  答："是2b"

  现在知道：需要传给b的梯度 = 2b = 2×6 = 12

第2步：b = a × 2
  问："a×2对a的导数是什么？"
  答："是2"

  链式法则：传给a的梯度 = 12 × 2 = 24

第3步：a = x + 1
  问："x+1对x的导数是什么？"
  答："是1"

  链式法则：传给x的梯度 = 24 × 1 = 24

最终：x.grad = 24 ✓
```

---

**链式法则就是"接力"：**

```
想象接力跑：

y → b → a → x

梯度从y开始，每次乘一个"局部导数"，传给下一个

y传给b：梯度 = 2b = 12
b传给a：梯度 = 12 × 2 = 24
a传给x：梯度 = 24 × 1 = 24

每一步只关心自己那一段：
  "y = b²" → 只知道导数是2b
  "b = a×2" → 只知道导数是2
  "a = x+1" → 只知道导数是1

然后把梯度传下去！
```

---

**线性回归的例子：**

```python
w = torch.tensor([2.0], requires_grad=True)
X = torch.tensor([[1.0]])  # 一个样本
y_true = torch.tensor([5.0])

y_hat = X * w              # y_hat = 1×2 = 2
error = y_hat - y_true     # error = 2-5 = -3
l = error ** 2 / 2         # l = (-3)²/2 = 4.5

l.backward()

print(w.grad)  # 输出：tensor([-3.])
```

**backward()怎么算出来的？**

```
记录的运算链：

w → X×w → y_hat → y_hat-y → error → error²/2 → l

反向传播：

从l开始（梯度=1）：

l = error²/2
  → 导数 = error
  → 传给error：梯度 = 1 × error = -3

error = y_hat - y_true
  → 导数（对y_hat） = 1
  → 传给y_hat：梯度 = -3 × 1 = -3

y_hat = X × w
  → 导数 = X = 1
  → 传给w：梯度 = -3 × 1 = -3

最终：w.grad = -3
```

---

**总结一句话：**

```
backward()的工作：

1. 你写前向运算代码时，PyTorch偷偷记录每个运算

2. 每个运算都知道自己的"局部导数"

3. backward()从终点开始，往回走

4. 每走到一个运算，就用链式法则：
   传下去的梯度 = 收到的梯度 × 局部导数

5. 最终到达w、b，把梯度存到.grad属性

你不需要管细节！调用backward()，梯度自动算好！
```

---

**比喻理解：**

```
backward()就像快递追踪系统

你发货（前向运算）：
  北京 → 上海 → 广州 → 深圳

系统记录每一步（计算图）

现在要回溯（backward）：
  问深圳："货从哪来的？" → "广州"
  问广州："货从哪来的？" → "上海"
  问上海："货从哪来的？" → "北京"

梯度就是沿着这条链反向传回去！
```

---

**常见运算的导数（PyTorch内置）：**

| 运算 | 导数 |
|------|------|
| `y = x + c` | `dy/dx = 1` |
| `y = x × c` | `dy/dx = c` |
| `y = x²` | `dy/dx = 2x` |
| `y = x³` | `dy/dx = 3x²` |
| `y = sin(x)` | `dy/dx = cos(x)` |
| `y = exp(x)` | `dy/dx = exp(x)` |

PyTorch知道这些，你调用backward()它自动应用！

---

#### 练习解答

| 练习 | 核心答案 |
|------|---------|
| 1. 权重初始化为0 | 线性回归有效，多层网络无效（对称性问题） |
| 2. 电压电流模型 | 可以，欧姆定律是线性模型，自动微分可学习 |
| 3. 普朗克定律 | 理论可行，实际需注意数值稳定性 |
| 4. 二阶导数问题 | 内存大、数值不稳定、开销大 |
| 5. reshape原因 | 形状对齐，避免广播错误 |
| 6. 不同学习率 | 太小慢，适中快，太大发散 |
| 7. 不能整除时 | 最后batch样本少，影响小 |

**练习1详解：权重初始化为零**

```
线性回归情况：有效
  → 只有一个输出层
  → 即使权重全为0，偏置b也会被更新
  → 梯度传播正常，权重会逐渐区分不同特征

多层神经网络：无效
  → 多个隐藏层神经元权重全为0
  → 所有神经元收到相同的梯度
  → 所有神经元更新后权重相同
  → 相当于只有一个神经元！

原因：对称性问题
解决：随机初始化权重，打破对称性
```

---

### 线性回归简洁实现 (linear-regression-concise.ipynb)

#### Q1: PyTorch高级API的核心组件

| 组件 | 作用 | 替代的从零实现代码 |
|------|------|------------------|
| `nn.Sequential` | 定义网络结构 | 手写linreg函数 |
| `nn.Linear` | 定义线性层 | 手写matmul+广播 |
| `nn.MSELoss` | 定义损失函数 | 手写squared_loss |
| `torch.optim.SGD` | 定义优化器 | 手写sgd函数 |
| `DataLoader` | 批量数据迭代 | 手写data_iter |

---

#### Q2: nn.Sequential详解

```python
net = nn.Sequential(nn.Linear(2, 1))

解释：
  → nn.Sequential：按顺序组合多个层
  → nn.Linear(2, 1)：输入2维，输出1维的线性层
  → 自动包含权重w和偏置b

内部结构：
  net[0] → 第0层（nn.Linear）
    net[0].weight → 权重矩阵，shape=(1, 2)
    net[0].bias → 偏置向量，shape=(1,)
```

---

#### Q3: DataLoader和TensorDataset

```python
from torch.utils import data

# TensorDataset：将特征和标签打包
dataset = data.TensorDataset(features, labels)

# DataLoader：批量迭代器
data_iter = data.DataLoader(dataset, batch_size, shuffle=True)

# 使用
for X, y in data_iter:
    # 每次返回一个batch的数据
```

**对比从零实现的data_iter：**

| 方式 | 优点 |
|------|------|
| 从零实现 | 理解底层原理，灵活定制 |
| DataLoader | 高效、GPU支持、多线程加载 |

---

#### Q4: trainer.zero_grad()为什么放在l.backward()之前？

**核心原因：PyTorch默认累加梯度**

```python
# 错误顺序（梯度累加）
l.backward()          # 计算梯度，累加到.grad
trainer.step()        # 用累加梯度更新参数
trainer.zero_grad()   # 清零梯度（晚了！）

问题：下次backward会累加到上次的梯度上！
```

```python
# 正确顺序（梯度独立）
trainer.zero_grad()   # 先清零上次的梯度
l.backward()          # 计算本次梯度（从0开始）
trainer.step()        # 用本次梯度更新参数
```

**梯度累加演示：**

```python
# PyTorch默认行为
x = torch.tensor([1.0], requires_grad=True)

# 第1次backward
y1 = x * 2
y1.backward()
print(x.grad)  # tensor([2.])  ← 梯度=2

# 第2次backward（不清零）
y2 = x * 3
y2.backward()
print(x.grad)  # tensor([5.])  ← 梯度=2+3=5（累加！）

# 正确做法：每次backward前zero_grad
x.grad.zero_()
y3 = x * 4
y3.backward()
print(x.grad)  # tensor([4.])  ← 梯度=4（正确）
```

**为什么PyTorch默认累加？**

```
好处：
  → 支持梯度累积（batch太小时，累积多个batch再更新）
  → 支持多任务学习（多个损失函数的梯度累加）
  → 灵活性更高

代价：
  → 需要手动清零
  → 不清零会导致梯度错误
```

---

#### Q5: 训练循环对比

**从零实现：**

```python
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
```

**简洁实现：**

```python
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()  # 清零梯度
        l.backward()         # 反向传播
        trainer.step()       # 更新参数
```

| 差异点 | 从零实现 | 简洁实现 |
|--------|---------|---------|
| 调用net | `net(X, w, b)` | `net(X)` |
| 梯度计算 | `l.sum().backward()` | `l.backward()` |
| 参数更新 | 手写sgd函数 | `trainer.step()` |
| 梯度清零 | 在sgd中完成 | 显式调用`zero_grad()` |

---

#### 练习解答

**练习1：如果用损失总和而非平均值**

```
问题1：有什么影响？

损失总和：
  l = Σᵢ (ŷ_i - y_i)² / 2

梯度：
  ∂l/∂w = Σᵢ x_i × (ŷ_i - y_i)  ← 累加，不是平均

影响：
  → 梯度放大了batch_size倍
  → 参数更新幅度变大
  → 需要调整学习率：η' = η / batch_size

nn.MSELoss(reduction='sum')用法：
  → 损失求和而非平均
  → 需要手动调整学习率
```

**练习2：PyTorch损失函数和初始化**

```
问题2.1：其他损失函数

| 损失函数 | 用途 | 公式 |
|---------|------|------|
| nn.L1Loss | 回归，对异常值鲁棒 | Σ|ŷ-y| |
| nn.SmoothL1Loss | Huber损失 | 小误差平方，大误差绝对值 |
| nn.CrossEntropyLoss | 分类 | -Σy×log(ŷ) |

nn.SmoothL1Loss（Huber损失）：
  当|ŷ-y| < δ：l = ½(ŷ-y)²
  当|ŷ-y| > δ：l = δ|ŷ-y| - ½δ²

  特点：
    → 误差小时光滑（像MSE）
    → 误差大时鲁棒（像L1）
    → 默认δ=1.0
```

```
问题2.2：权重初始化方法

| 初始化方法 | 适用场景 |
|-----------|---------|
| 随机初始化 | 一般情况 |
| Xavier初始化 | tanh/sigmoid激活 |
| Kaiming初始化 | ReLU激活 |
| 预训练权重 | 迁移学习 |

PyTorch内置：
  nn.init.normal_(net[0].weight, mean=0, std=0.01)
  nn.init.zeros_(net[0].bias)
  nn.init.xavier_uniform_(net[0].weight)
  nn.init.kaiming_normal_(net[0].weight)
```

**练习3：访问梯度**

```python
# 访问权重梯度
print(net[0].weight.grad)  # shape=(1, 2)

# 访问偏置梯度
print(net[0].bias.grad)    # shape=(1,)

# 训练后查看梯度
for epoch in range(num_epochs):
    for X, y in data_iter:
        trainer.zero_grad()
        l = loss(net(X), y)
        l.backward()
        trainer.step()

    print(f'weight.grad: {net[0].weight.grad}')
    print(f'bias.grad: {net[0].bias.grad}')
```

---

### softmax回归理论 (softmax-regression.ipynb)

#### softmax回归核心概念

**回归 vs 分类：**

| 问题类型 | 回答的问题 | 例子 |
|---------|-----------|------|
| 回归 | "多少？" | 房价、胜场数、住院天数 |
| 分类 | "哪一个？" | 垃圾邮件判断、图像分类 |

---

#### Q1: 独热编码是什么？

**定义：** 用向量表示类别，正确类别位置为1，其余为0

```
例子：猫、鸡、狗三个类别

猫 → (1, 0, 0)
鸡 → (0, 1, 0)
狗 → (0, 0, 1)

特点：
  → 向量长度 = 类别数量
  → 只有一个位置是1
  → 其余位置都是0
```

**为什么不用整数编码（1, 2, 3）？**

```
整数编码的问题：
  → 类别之间没有自然顺序
  → 猫=1, 鸡=2, 狗=3 没有大小关系
  → 模型可能误解为"猫 < 鸡 < 狗"

独热编码优势：
  → 每个类别独立表示
  → 不暗示任何顺序关系
  → 适合无序分类问题
```

---

#### Q2: 为什么需要softmax运算？

**问题：线性输出不能直接作为概率**

线性输出 $o$ 存在两个问题：

1. **没有约束总和为1**：输出可能和为2、3或其他值
2. **可能为负值**：概率不能为负

这违反了概率的基本公理！

**softmax函数解决方案：**

$$\hat{y}_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$$

```
作用：
  → exp(o_j)：确保非负（指数函数值>0）
  → 除以总和：确保总和为1
  → 保持可导性：方便反向传播
```

**数值例子：**

```
未规范化输出：o = (2.0, 1.0, 0.1)

计算过程：
  exp(o₁) = e²·⁰ ≈ 7.389
  exp(o₂) = e¹·⁰ ≈ 2.718
  exp(o₃) = e⁰·¹ ≈ 1.105
  总和 = 7.389 + 2.718 + 1.105 ≈ 11.212

softmax输出：
  ŷ₁ = 7.389/11.212 ≈ 0.659
  ŷ₂ = 2.718/11.212 ≈ 0.242
  ŷ₃ = 1.105/11.212 ≈ 0.099

验证：总和 = 0.659 + 0.242 + 0.099 = 1.0 ✓
```

---

#### Q3: softmax不改变大小顺序

**重要性质：**

$$\operatorname*{argmax}_j \hat{y}_j = \operatorname*{argmax}_j o_j$$

```
含义：
  → softmax前最大的o，softmax后对应的ŷ也最大
  → 预测类别时，直接看最大的o即可

原因：
  → 指数函数是单调递增的
  → o₁ > o₂ → exp(o₁) > exp(o₂)
  → 除以同一个总和，顺序不变
```

---

#### Q4: 交叉熵损失公式推导

**从极大似然估计出发：**

```
假设：模型输出ŷ是每个类别的条件概率

似然函数：
  P(Y|X) = ∏ᵢ P(y⁽ⁱ⁾|x⁽ⁱ⁾)

负对数似然（最小化目标）：
  -log P(Y|X) = Σᵢ -log P(y⁽ⁱ⁾|x⁽ⁱ⁾)
              = Σᵢ l(y⁽ⁱ⁾, ŷ⁽ⁱ⁾)
```

**交叉熵损失定义：**

$$l(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{j=1}^q y_j \log \hat{y}_j$$

```
独热编码的特殊性：
  → y = (0, 0, 1) 表示真实类别是第3类
  → 只有y₃=1，其他y₁=y₂=0
  → 求和时只有一项不为零

简化公式：
  l(y, ŷ) = -log ŷ_correct
  其中ŷ_correct是真实类别的预测概率
```

**数值例子：**

```
真实标签：y = (0, 0, 1)（狗）
预测概率：ŷ = (0.1, 0.2, 0.7)

交叉熵损失：
  l = -[0×log(0.1) + 0×log(0.2) + 1×log(0.7)]
    = -log(0.7)
    ≈ 0.357

解释：
  → 预测概率越高（接近1），损失越低（接近0）
  → 预测概率越低（接近0），损失越高（很大）
```

---

#### Q5: softmax导数 = 预测 - 真实

**将softmax代入交叉熵损失：**

$$l(\mathbf{y}, \hat{\mathbf{y}}) = \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j$$

**对$o_j$的导数：**

$$\frac{\partial l}{\partial o_j} = \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} - y_j = \mathrm{softmax}(\mathbf{o})_j - y_j$$

```
直观理解：
  → 导数 = 预测概率 - 真实标签
  → 与线性回归中 梯度 = 预测值 - 真实值 类似！

为什么这么简洁？
  → 这是指数族分布的普遍性质
  → softmax属于指数族分布
  → 所有指数族分布的对数似然梯度都有这种形式
```

**数值例子：**

```
真实标签：y = (0, 0, 1)
softmax输出：ŷ = (0.1, 0.2, 0.7)
未规范化输出：o = (2.0, 1.0, 0.1)

梯度计算：
  ∂l/∂o₁ = ŷ₁ - y₁ = 0.1 - 0 = 0.1
  ∂l/∂o₂ = ŷ₂ - y₂ = 0.2 - 0 = 0.2
  ∂l/∂o₃ = ŷ₃ - y₃ = 0.7 - 1 = -0.3

含义：
  → 正确类别的梯度为负 → 需要增加o₃
  → 错误类别的梯度为正 → 需要减少o₁、o₂
```

---

#### Q6: 信息论基础概念

**熵（Entropy）：**

$$H[P] = \sum_j - P(j) \log P(j)$$

```
含义：
  → 量化分布P的不确定性
  → 越不确定，熵越大
  → 完全确定（只有一个事件概率为1），熵=0

例子：
  → 确定性事件：P = (1, 0, 0)，H = 0
  → 完全不确定：P = (1/3, 1/3, 1/3)，H最大

单位：
  → 底为e：纳特（nat）
  → 底为2：比特（bit）
  → 1纳特 ≈ 1.44比特
```

**信息量：**

$$\text{信息量} = -\log P(j)$$

```
含义：
  → 事件发生的"惊异程度"
  → 概率低的事件 → 信息量大（更惊异）
  → 概率高的事件 → 信息量小（不惊异）

例子：
  → 确定事件P=1：信息量=0（不惊异）
  → 稀有事件P=0.01：信息量≈4.6（很惊异）
```

**交叉熵：**

$$H(P, Q) = -\sum_j P(j) \log Q(j)$$

```
含义：
  → 用概率分布Q编码来自分布P的数据所需平均比特数
  → "主观概率为Q的观察者看到根据P生成的数据时的预期惊异"

重要性质：
  → H(P, Q) ≥ H(P)
  → 当P=Q时，交叉熵最小，等于熵H(P)
```

**交叉熵损失的理解：**

```
两种理解方式：

1. 统计角度：极大似然估计
   → 最小化负对数似然
   → 让模型预测概率与真实分布匹配

2. 信息论角度：最小化惊异
   → 用模型预测Q编码真实标签P
   → 让预测越准确，惊异越小
```

---

#### Q7: 小批量样本的矢量化

**维度说明：**

```
样本数：n
特征维度：d
类别数：q

矩阵形状：
  X ∈ ℝ^(n×d)  → 特征矩阵
  W ∈ ℝ^(d×q)  → 权重矩阵
  b ∈ ℝ^(q)    → 偏置向量
```

**矩阵运算：**

$$
\begin{aligned}
\mathbf{O} &= \mathbf{X} \mathbf{W} + \mathbf{b} \\
\hat{\mathbf{Y}} &= \mathrm{softmax}(\mathbf{O})
\end{aligned}
$$

```
运算过程：
  → O = X×W + b（形状：n×q）
  → 每行是一个样本的未规范化输出
  → softmax按行执行（每行独立归一化）
```

---

#### 练习解答

| 练习 | 核心答案 |
|------|---------|
| 练习1 | softmax交叉熵二阶导数 = softmax方差 |
| 练习2 | 三等概率编码问题 → 联合编码效率更高 |
| 练习3 | RealSoftMax证明 → soft-min类似定义 |

**练习1详解：softmax交叉熵的二阶导数**

```
问题1.1：计算二阶导数

已知一阶导数：
  ∂l/∂o_j = softmax(o)_j - y_j

二阶导数（Hessian矩阵）：
  ∂²l/∂o_i∂o_j = ∂/∂o_i [softmax(o)_j - y_j]

  = ∂softmax(o)_j/∂o_i

  因为y_j是常数，导数为0

softmax的导数：
  当i = j时：
    ∂softmax_j/∂o_j = softmax_j × (1 - softmax_j)

  当i ≠ j时：
    ∂softmax_j/∂o_i = -softmax_j × softmax_i

验证：
  softmax_j = exp(o_j)/Σexp(o_k)

  对o_j求导（i=j）：
    = exp(o_j)/Σexp × [Σexp - exp(o_j)]/Σexp
    = softmax_j × (1 - softmax_j)

  对o_i求导（i≠j）：
    = -exp(o_j)×exp(o_i)/(Σexp)²
    = -softmax_j × softmax_i
```

```
问题1.2：验证方差关系

softmax分布的方差：
  Var(X) = E[X²] - E[X]²

  对于softmax输出：
    E[X_j] = softmax_j（期望）
    E[X_j²] = softmax_j（因为是伯努利）

  方差：
    Var_j = softmax_j - softmax_j²
          = softmax_j × (1 - softmax_j)

这与二阶导数中i=j的情况相同！

结论：
  → Hessian矩阵的对角元素 = softmax分布的方差
  → 不对角元素 = -softmax_j × softmax_i（协方差）
```

---

**练习2详解：三等概率编码问题**

```
问题2.1：二进制编码的问题

三个类别概率相等：P = (1/3, 1/3, 1/3)

尝试二进制编码：
  → 每个类别用固定长度二进制码
  → 3个类别至少需要2位（能表示4个状态）
  → 类别编码：00, 01, 10（剩11未用）

熵计算：
  H(P) = -3 × (1/3) × log(1/3) = log(3) ≈ 1.585纳特

二进制编码效率：
  → 每个事件用2位 = 2比特 ≈ 1.386纳特
  → 1.386 < 1.585
  → 这违反了信息论基本定理！

问题：
  → 平均编码长度 < 熵是不可能的理论上
  → 但实际我们用了2位固定长度
  → 效率浪费：2 - 1.585 = 0.415纳特/事件
```

```
问题2.2：更好的编码方法

方法1：联合编码两个观察

考虑两个独立观察：
  → P(x₁, x₂) = P(x₁) × P(x₂)
  → 共有9种组合，概率都是1/9

熵：
  H(P₂) = 2 × log(3) ≈ 3.17纳特

编码9种状态：
  → 需要4位二进制（能表示16种状态）
  → 平均编码长度 = 4/2 = 2位/观察 = 1.386纳特/观察

改进：
  → 从1.386纳特提升到接近熵1.585纳特
  → 但仍有浪费

方法2：联合编码n个观察

编码n个观察：
  → 共有3ⁿ种组合
  → 需要⌈log₂(3ⁿ)⌉位

平均编码长度：
  bits/观察 = ⌈n × log₂(3)⌉ / n

当n→∞：
  → ⌈n × log₂(3)⌉ / n → log₂(3) ≈ 1.585比特
  → 逼近熵的极限！

结论：
  → 联合编码越多观察，效率越高
  → 大数定律保证组合分布趋于均匀
  → 可以设计接近最优的编码
```

---

**练习3详解：RealSoftMax证明**

```
RealSoftMax定义：
  RealSoftMax(a, b) = log(exp(a) + exp(b))

问题3.1：证明 > max(a, b)

证明：
  exp(a) > 0 且 exp(b) > 0（指数函数性质）

  exp(a) + exp(b) > exp(a) 且 > exp(b)

  取log：
    log(exp(a) + exp(b)) > log(exp(a)) = a
    log(exp(a) + exp(b)) > log(exp(b)) = b

  所以：RealSoftMax(a, b) > max(a, b) ✓
```

```
问题3.2：证明 λ⁻¹ RealSoftMax(λa, λb) > max(a, b)

设λ > 0：

λ⁻¹ RealSoftMax(λa, λb)
  = λ⁻¹ × log(exp(λa) + exp(λb))

由问题3.1：
  RealSoftMax(λa, λb) > max(λa, λb)
  = λ × max(a, b)（因为λ > 0）

所以：
  λ⁻¹ × RealSoftMax(λa, λb) > λ⁻¹ × λ × max(a, b)
  = max(a, b) ✓
```

```
问题3.3：证明λ→∞时趋于max(a, b)

设a > b（不失一般性）：

λ⁻¹ RealSoftMax(λa, λb)
  = λ⁻¹ × log(exp(λa) + exp(λb))
  = λ⁻¹ × log(exp(λa) × (1 + exp(λ(b-a))))
  = λ⁻¹ × [λa + log(1 + exp(λ(b-a)))]
  = a + λ⁻¹ × log(1 + exp(λ(b-a)))

当λ→∞：
  → b - a < 0（因为a > b）
  → λ(b-a) → -∞
  → exp(λ(b-a)) → 0
  → log(1 + exp(λ(b-a))) → log(1) = 0
  → λ⁻¹ × log(1 + ...) → 0

所以：
  λ⁻¹ RealSoftMax(λa, λb) → a = max(a, b) ✓

解释：
  → λ越大，softmax越"硬"
  → 最大值主导，其他值被抑制
  → 极限情况下就是真正的max函数
```

```
问题3.4：soft-min定义

类比soft-max：
  softmax是"平滑的max"

soft-min应该是"平滑的min"

定义：
  soft-min(a, b) = -soft-max(-a, -b)
                 = -log(exp(-a) + exp(-b))
                 = log(1/(exp(-a) + exp(-b)))

或者：
  soft-min(a, b) = log(exp(-a) + exp(-b))（负号在外）

验证：
  soft-min(a, b) < min(a, b)

  因为：soft-max(-a, -b) > max(-a, -b) = -min(a, b)

  所以：-soft-max(-a, -b) < min(a, b) ✓
```

```
问题3.5：扩展到多个数

多个数的soft-max：
  softmax(a₁, a₂, ..., aₙ) = log(Σᵢ exp(a_i))

性质：
  → softmax > max(a₁, ..., aₙ)
  → λ⁻¹ softmax(λa₁, ..., λaₙ) → max(a₁, ..., aₙ) 当λ→∞

多个数的soft-min：
  soft-min(a₁, ..., aₙ) = log(Σᵢ exp(-a_i))

性质：
  → soft-min < min(a₁, ..., aₙ)
  → λ⁻¹ soft-min(λa₁, ..., λaₙ) → min(a₁, ..., aₙ) 当λ→∞
```

---

## 附录：核心概念速查表

### 梯度相关

| 概念 | 公式/说明 |
|------|---------|
| 梯度定义 | ∇f = [∂f/∂x₁, ∂f/∂x₂, ...]ᵀ |
| 梯度下降 | w ← w - η × 梯度 |
| 矩阵乘梯度 | ∇(Ax) = Aᵀ |

### 概率相关

| 概念 | 公式 |
|------|------|
| 条件概率 | P(B|A) = P(A,B)/P(A) |
| 贝叶斯定理 | P(A|B) = P(B|A)P(A)/P(B) |
| 边际化 | P(B) = Σ P(A,B) |
| 独立性 | P(A,B) = P(A)×P(B) |

### 线性回归相关

| 概念 | 公式 |
|------|------|
| 线性模型 | ŷ = wᵀx + b |
| 平方损失 | L = ½(ŷ-y)² |
| 解析解 | w = (XᵀX)⁻¹Xᵀy |
| SGD更新 | w ← w - η/|B| Σ x(ŷ-y) |

---

*文档生成时间：2026-05-07*
*最后更新时间：2026-05-07（添加线性回归从零实现章节完整问答）*