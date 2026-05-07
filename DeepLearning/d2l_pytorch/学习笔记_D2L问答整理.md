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