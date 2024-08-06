# Numpy

Numpy库包含两个基本的数据类型数组(array)和矩阵(matrix)

## 数组(array)

### 创建方法

#### 列表

用列表和嵌套列表来创建

```python
a = array([1, 2, 3])  # 一维数组
b = array([[1, 2, 3], [4, 5, 6]])  # 二维数组
```

#### arange

使用`range`可以创建一个数字序列的一维数组，和python中的range函数一样，设置序列范围和步进，范围同样是**左闭右开**

```python
# array([start], stop[, step,], dtype=None)
a = arange(10)
b = arange(2, 10, 2)
print(a) # [0 1 2 3 4 5 6 7 8 9]
print(b) # [2 4 6 8]
```

#### ones/one_like

使用`ones`创建全为1的数组

```python
# one(shape, dtype=None, order='C')
a = ones(10) # [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
b = ones((2, 3)) # [[1. 1. 1.] [1. 1. 1.]]
```

`shape`参数是以元组的形式描述数组的维度，例如ones((2, 3))，就是希望生成的数组是一个2x3的数组

对于`one_like`则是创建一个和传入参数形状相同的全1数组

```python
a = ones_like((1, 2))
b = ones_like([1, 1])
print(a) # [1 1]
print(b) # [1 1]
```

#### zeros/zeros_like

使用`zeros`创建一个全为0的数组，用法同ones/ones_like

#### empty/empty_like

使用`empty`创建一个全为0的数组，用法同ones/ones_like

**注意**：本身的意思是创建一个未初始化的数组，因此里面的数据可能是随机值

#### full/full_like

使用`full`创建一个全是指定值的数组，用法同ones/ones_like，只需在形状后加上指定值

```python
a = full(10, 'a')
b = full((2,2), 12)

print(a) # ['a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a' 'a']
print(b) # [[12 12] [12 12]]
```

#### random.randn

使用`random`模块下的`randn`创建一个全是随机数的数组，直接传入数组的维度即可（不需要以元组的形式，这是与上面不同的）

```python
a = random.randn()
b = random.randn(2)
c = random.randn(2, 2)

print(a) # -1.5996600128513558
print(b) # [-1.24893925 -0.63994095]
print(c) # [[ 0.73468967  1.45951709] [-0.11737052  0.52854982]]
```

#### tile

使用`tile`可以按指定的重复方式对数组进行复制，以创建更大的数组

```python
"""
np.tile(arr, repetitions)
arr是用来复制的数组，repetitions是在各维度上复制的次数
"""

arr = np.array([1, 2, 3])
repeated_arr = np.tile(arr, 3)
print(repeated_arr)
# 输出：[1 2 3 1 2 3 1 2 3]

arr2 = np.array([[1, 2], [3, 4]])
repeated_arr2 = np.tile(arr2, (2, 3))
print(repeated_arr2)
# 输出：
# [[1 2 1 2 1 2]
#  [3 4 3 4 3 4]
#  [1 2 1 2 1 2]
#  [3 4 3 4 3 4]]
```

### 本身属性

#### 维度 shape

#### 维数 ndim

```python
a = array([1, 2, 3])  # 一维数组
b = array([[1, 2, 3], [4, 5, 6]])  # 二维数组
c = array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # 三维数组
d = array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
             [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]])  # 四维数组

print(f"维度：{a.shape},维数：{a.ndim}") 
print(f"维度：{b.shape},维数：{b.ndim}")
print(f"维度：{c.shape},维数：{c.ndim}")
print(f"维度：{d.shape},维数：{d.ndim}")

# 维度：(3,),维数：1
# 维度：(2, 3),维数：2
# 维度：(2, 2, 2),维数：3
# 维度：(2, 2, 2, 2),维数：4
```

数括号，几维数组就有几个括号开头；

例如，对于d

​	d开头有四个括号是四维数组

​	这个四维数组由2个三维数组组成；

​	每个三维数组包含2个二位数组；

​	每个二维数组包含2个一维数组；

​	每个一维数组包含2个元素。

#### 元素个数 size

```python
a = array([1, 2, 3])  # 一维数组
b = array([[1, 2, 3], [4, 5, 6]])  # 二维数组

print(a.size) # 3
print(b.size) # 6
```

#### 元素类型 dtype

```python
a = array([1, 2, 3])  # 一维数组
b = array([['a', 'v', 'a'], ['s', 's', 'v']])  # 二维数组


print(a.dtype) # int32
print(b.dtype) # U1
```

### 操作方法

#### reshape

使用`reshape`，可以转换数组的维度

```python
a = arange(10).reshape(2, 5)

print(a)
```

#### astype

使用`astype`，可以转换数组的元素的类型

#### 运算

数组的运算都是逐一对每个元素运算的，两个数组之间进行运算需要维度一致

```python
a = array([0, 3.1415/2])
b = array([0, 1])
c = arange(1, 7).reshape(2, 3)
d = c
print(sin(a)) # [0. 1.]
print(exp(b)) # [1.         2.71828183]
print(b + 1)	　# [1 2]
print(d + c) # [[ 2  4  6] [ 8 10 12]]
print(c - d) # [[0 0 0] [0 0 0]]
```

### 索引方式

#### 基础索引

**一维数组**

```python
a = arange(10)

print(a[0], a[9], a[-1])   # 0 9 9
print(a[0:5:1])     # [0 1 2 3 4]
print(a[:5:2])      # [0 2 4]
print(a[1:-1])      # [1 2 3 4 5 6 7 8]
```

**二维数组**

```python
a = arange(1, 10, 1).reshape(3, 3)

print(a)
print(a[0, 1])      # 2				
print(a[0, :-1:])   # [1 2]			
print(a[:2, 1])     # [2 5]
print(a[:2, :2])    # [[1 2] [4 5]]
```

#### 数组索引

**一维数组**

```python
a = arange(0, 10, 1)
print(a[[3, 4, 7]])	# [3 4 7]

index = array([[0, 1], [2, 3]])
print(a[index])		# [[0 1] [2 3]]
```

**二维数组**

```python
a = arange(1, 10, 1).reshape(3, 3)

print(a[[0, 1],])    		# [[1 2 3] [4 5 6]]
print(a[:, [0, 1]]) 		# [[1 2] [4 5] [7 8]]
print(a[[0, 1], [0, 1]])	# [1 5] 取得是(0,0), (1,1)的值
```

#### 布尔索引

```python
# 布尔索引,判断本身会是一个与原数组同形的bool数组
a = arange(10)
a > 7
print(a > 7)	#[False False False False False False False False  True  True]

```

**一维数组**

```python
a = arange(10)

print(a[a > 7])	#[8 9]
```

**二维数组**

```python
a = arange(10).reshape(2, 5)
a[:, 3] > 7				#对每一行第三列=是否大于7判断
print(a[a[:, 3] > 7])	#筛选的是第三列＞7的行
```

**组合操作**

```python
"""
condition = (条件1) 与或非 (条件2) ......

只是将单一的条件a>7变成了(a>7) & (a<10) 这样形式的组合条件用来索引
"""
a = arange(10)
b = arange(1, 13).reshape(4, 3)
condition = (a > 5) | (a < 3)
print(a[condition])		# [0 1 2 6 7 8 9]

condition = (b[:, 2] > 9) | (b[:, 2] < 6)
print(b[condition])		#[[ 1  2  3] [10 11 12]]
```

### random随机函数

#### 常见函数

| 函数名                        | 说明                                   |
| ----------------------------- | -------------------------------------- |
| seed([seed])                  | 设定随机种子，使每次生成的随机数相同   |
| rand(d0, d1, ... ,dn)         | 返回数据在[0,1)，满足正态分布          |
| randn(d0, d1, ... ,dn)        | 返回数据具有标准正态分布               |
| randint(d0, d1, ... ,dn)      | 生成随机整数，左闭右开                 |
| random(d0, d1, ... ,dn)       | 生成[0.0, 1.0)的随机数                 |
| choice(a[, size, replace, p]) | 从一维数组a中生成                      |
| shuffle(x)                    | 对x随机排序                            |
| permutation(x)                | 对x随机排序，或者数字的全排序          |
| normal([loc, scale, size])    | 均值为loc，方差为scale的高斯分布的数字 |
| uniform([low, high, size])    | 在[low，high）之间生成均匀分布的数字   |

d表示维度，size生成的数量

### 数学统计函数

#### 常用函数

| 函数名                   | 说明                                    |
| ------------------------ | --------------------------------------- |
| np.sum(arr)              | 所有元素之和                            |
| np.prod(arr)             | 所有元素乘积                            |
| np.cumsum(arr)           | 元素的累加的和                          |
| np.cumsum(arr)           | 元素的累乘的积                          |
| np.min(arr)              | 最小值                                  |
| np.max(arr)              | 最大值                                  |
| np.median(arr)           | 中位数                                  |
| np.mean(arr)             | 平均数                                  |
| np.std(arr)              | 标准差                                  |
| np.var(arr)              | 方差                                    |
| np.average(arr, weight=) | 加权平均<br />weight需要和array形状一致 |
| np.percentile(arr)       | 0-100百分位数                           |
| np.quantile(arr)         | 0-1分位数                               |

#### axis参数

`axis = 0` 表示行

`axis = 1` 表示列

可以理解为，将数组按照第`axis`个维度进行计算：

​	当axis=0，表示按照第0个维度也就是行进行计算，那么就是每一行第一个元素进行这个操作，第二个元素进行这个操作......

​	当axis=1，表示按照第1个维度也就是列进行计算，那么就是每一列第一个元素进行这个操作，第二个元素进行这个操作......

​	以此类推

### 增加维度

#### newaxis关键字

`newaxis`本身就是`None`，使用时可以直接使用None

```python
arr = arange(5)

print(f"arr的维度：{arr.shape}")				# (5,)

print(f"添加行维度：{arr[None, None, :].shape}") # (1, 1, 5)

print(f"增加列维度：{arr[:, None, None].shape}") # (5, 1, 1)

```

理解为用`newaxis`或者是`None`在数组的维度描述中占位：

​	例如原来数组维度是 (5,) ，通过`arr[None, None, :]`在前面占位，最终结果符合前面关于shape的理解

#### expand_dims方法

```python
arr = random.randn(3, 3, 3)
print(f"arr的维度：{arr.shape}")						# (3, 3, 3)
print(f"添加行维度：{expand_dims(arr,axis=0).shape}")	   # (1, 3, 3, 3)
print(f"增加列维度：{expand_dims(arr,axis=1).shape}")	   # (3, 1, 3, 3)
```

理解为在`axis`位置处增加一个维度：

​	例如`expand_dims(arr,axis=1)`就是在原来的第1个维度处增加一个维度所以是(3, 1, 3, 3)

#### reshape方法

```python
arr = random.randn(3, 3, 3)
print(f"arr的维度：{arr.shape}")
print(f"添加行维度：{numpy.reshape(arr,(1, -1, 3, 3)).shape}")
print(f"增加列维度：{numpy.reshape(arr, (3, 3, 1, 3)).shape}")
print(f"增加列维度：{numpy.reshape(arr, (3, 1, -1, 1)).shape}")
```

将数组的维度改为如描述的那样，允许使用一个`-1`去指定一个未知的长度，未知的长度会等于元素个数/各维度长度累乘，如第4个例子中，共有27个元素，-1处等于27/3

### 数组合并

使用时会面临两个场景：

给已有的数据添加多行，即添加新的样本数据

给已有的数据添加多列，即添加新的特征

#### concatenate方法

```python
# 添加行
a = arange(12).reshape(3, 4)
b = random.randint(10, 20, size=(3, 2))
concatenate([a, b])

# 添加列
concatenate([a, b], axis=1)
```

#### vstack/row_stack方法

`vstack` 垂直方向添加

```python
a = arange(6).reshape(2, 3)
b = random.randint(10, 20, size=(4, 3))
vstack([a, b])
```

`row_stack`行添加

```python
a = arange(6).reshape(2, 3)
b = random.randint(10, 20, size=(4, 3))

print(row_stack([a, b]))
```

#### hstack/column_stack方法

`hstack`水平方向添加

```python
a = arange(6).reshape(2, 3)
b = random.randint(10, 20, size=(4, 3))
hstack([a, b])
```

`column_stack`列添加

```python
a = arange(6).reshape(2, 3)
b = random.randint(10, 20, size=(4, 3))
column_stack([a, b])
```