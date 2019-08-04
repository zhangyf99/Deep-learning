# 学习笔记+资料整理

[TOC]

## 1. NumPy学习简记

NumPy——一个功能强大的python库，主要用于对多维数组执行计算。

### 数组属性

NumPy的主要对象是同类型的多维数组。在NumPy中，维度称为轴，轴的数目为rank。

例如，对于数组：

```python
[[ 1., 0., 0.],
[ 0., 1., 2.]]
```

该数组有两个轴，第一个轴的长度2，第二个轴的长度为3。

NumPy的数组类被称为`ndarray`。`ndarray` 对象提供一些关键的属性：

- `ndarray.ndim`：数组的轴（维度）的个数。在Python中，维度的数量被称为rank。
- `ndarray.shape`：数组的维度。这是一个整数的元组，表示每个维度中数组的大小。对于有n行和m列的矩阵，`shape`将是(n,m)。因此，`shape`元组的长度就是rank或维度的个数 `ndim`。
- `ndarray.size`：数组元素的总数。这等于 `shape` 的元素的乘积。
- `ndarray.dtype`：一个描述数组中元素类型的对象。可以使用标准的Python类型创建或指定 `dtype`。另外NumPy提供它自己的类型，例如`numpy.int32`、`numpy.int16` 和 `numpy.float64`。
- `ndarray.itemsize`：数组中每个元素的字节大小。例如，元素为 `float64` 类型的数组的 `itemsize` 为8（=64/8），而 `complex32` 类型的数组的 `itemsize` 为4（=32/8）。它等于 `ndarray.dtype.itemsize` 。
- `ndarray.data`：该缓冲区包含数组的实际元素。通常，我们不需要使用此属性，因为我们将使用索引访问数组中的元素。

### 创建数组

```python
>>> import numpy as np
>>> a = np.array([2,3,4])
>>> a
array([2, 3, 4])
>>> a.dtype
dtype('int64')
>>> b = np.array([1.2, 3.5, 5.1])
>>> b.dtype
dtype('float64')

# 一个常见的错误在于使用多个数值参数调用array函数，而不是提供一个数字列表作为参数
>>> a = np.array(1,2,3,4)    # WRONG
>>> a = np.array([1,2,3,4])  # RIGHT

# 二维数组
>>> b = np.array([(1.5,2,3), (4,5,6)])
>>> b
array([[ 1.5,  2. ,  3. ],
        [ 4. ,  5. ,  6. ]])

# 数组的类型也可以在创建时明确指定
>>> c = np.array( [ [1,2], [3,4] ], dtype=complex )
>>> c
array([[ 1.+0.j,  2.+0.j],
        [ 3.+0.j,  4.+0.j]])

# NumPy提供了一个类似range的函数，该函数返回数组而不是列表
>>> np.arange( 10, 30, 5 )
array([10, 15, 20, 25])
>>> np.arange( 0, 2, 0.3 )
array([ 0. ,  0.3,  0.6,  0.9,  1.2,  1.5,  1.8])

# 函数linspace接受我们想要的元素数量而不是步长作为参数
>>> np.linspace( 0, 2, 9 )   # 9 numbers from 0 to 2
array([ 0.  ,  0.25,  0.5 ,  0.75,  1.  ,  1.25,  1.5 ,  1.75,  2.  ])
```

函数 `zeros` 创建一个由0组成的数组；函数 `ones` 创建一个由1数组的数组；函数 `empty` 内容是随机的并且取决于存储器的状态。默认情况下，创建的数组的 `dtype` 是 `float64`。

### 多维数组切片

```python
a = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28 ,29, 30],
              [31, 32, 33, 34, 35]])

print(a[0, 1:4]) # >>>[12 13 14]
print(a[1:4, 0]) # >>>[16 21 26]
print(a[::2,::2]) # >>>[[11 13 15]
                  #     [21 23 25]
                  #     [31 33 35]]
print(a[:, 1]) # >>>[12 17 22 27 32]
```

### 基本操作符

完全可以使用四则运算符+、-、/、*来完成运算操作。

除了 dot() 之外，这些操作符都是对数组进行逐元素运算。比如 (a, b, c) + (d, e, f) 的结果就是 (a+d, b+e, c+f)。

dot() 函数计算两个数组的点积。它返回的是一个标量（只有大小没有方向的一个值）而不是数组。

### 广播 Broadcasting

广播是一种强大的机制，它允许numpy在执行算术运算时使用不同形状的数组。通常，我们有一个较小的数组和一个较大的数组，我们希望多次使用较小的数组来对较大的数组执行一些操作。

例如，假设我们要向矩阵的每一行添加一个常数向量。我们可以这样做：

```python
import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v  # Add v to each row of x using broadcasting
print(y)  # Prints "[[ 2  2  4]
          #          [ 5  5  7]
          #          [ 8  8 10]
          #          [11 11 13]]"
```

将两个数组一起广播遵循以下规则：

1. 如果数组不具有相同的rank，则将较低等级数组的形状添加1，直到两个形状具有相同的长度。
2. 如果两个数组在维度上具有相同的大小，或者如果其中一个数组在该维度中的大小为1，则称这两个数组在维度上是兼容的。
3. 如果数组在所有维度上兼容，则可以一起广播。
4. 广播之后，每个阵列的行为就好像它的形状等于两个输入数组的形状的元素最大值。
5. 在一个数组的大小为1且另一个数组的大小大于1的任何维度中，第一个数组的行为就像沿着该维度复制一样。



## 2. Pandas学习简记

首先，导入：

```python
In [1]: import numpy as np

In [2]: import pandas as pd
```

### 创建对象

#### Series字典对象

通过传递一个值列表来创建一个Series对象，让pandas创建一个默认的整数索引：

```python
In [3]: s = pd.Series([1, 3, 5, np.nan, 6, 8])

In [4]: s
Out[4]: 
0    1.0
1    3.0
2    5.0
3    NaN
4    6.0
5    8.0
dtype: float64
```

```python
>>> s = pd.Series(data=[1,2,3,4],index = ['a','b','c','d'])  #传入键和值方式
>>> s
a    1
b    2
c    3
d    4
dtype: int64
>>> s.index    #获取键列表
Index(['a', 'b', 'c', 'd'], dtype='object')
>>> s.values    #获取值列表
array([1, 2, 3, 4], dtype=int64)
```

#### DataFrame表格对象

通过传递带有日期时间索引和列标记的NumPy数组来创建DataFrame对象：

```python
In [5]: dates = pd.date_range('20130101', periods=6)

In [6]: dates
Out[6]: 
DatetimeIndex(['2013-01-01', '2013-01-02', '2013-01-03', '2013-01-04',
               '2013-01-05', '2013-01-06'],
              dtype='datetime64[ns]', freq='D')

In [7]: df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))

In [8]: df
Out[8]: 
                   A         B         C         D
2013-01-01  0.469112 -0.282863 -1.509059 -1.135632
2013-01-02  1.212112 -0.173215  0.119209 -1.044236
2013-01-03 -0.861849 -2.104569 -0.494929  1.071804
2013-01-04  0.721555 -0.706771 -1.039575  0.271860
2013-01-05 -0.424972  0.567020  0.276232 -1.087401
2013-01-06 -0.673690  0.113648 -1.478427  0.524988
```

还可以传递带有Series的对象：

```python
In [9]: df2 = pd.DataFrame({'A': 1.,
   ...:                     'B': pd.Timestamp('20130102'),
   ...:                     'C': pd.Series(1, index=list(range(4)), dtype='float32'),
   ...:                     'D': np.array([3] * 4, dtype='int32'),
   ...:                     'E': pd.Categorical(["test", "train", "test", "train"]),
   ...:                     'F': 'foo'})
   ...: 

In [10]: df2
Out[10]: 
     A          B    C  D      E    F
0  1.0 2013-01-02  1.0  3   test  foo
1  1.0 2013-01-02  1.0  3  train  foo
2  1.0 2013-01-02  1.0  3   test  foo
3  1.0 2013-01-02  1.0  3  train  foo

# 每列的类型不同
In [11]: df2.dtypes
Out[11]: 
A           float64
B    datetime64[ns]
C           float32
D             int32
E          category
F            object
dtype: object
```

### 查看数据

查看frame对象的头尾数据：

```python
In [13]: df.head()   # 默认值5
Out[13]: 
                   A         B         C         D
2013-01-01  0.469112 -0.282863 -1.509059 -1.135632
2013-01-02  1.212112 -0.173215  0.119209 -1.044236
2013-01-03 -0.861849 -2.104569 -0.494929  1.071804
2013-01-04  0.721555 -0.706771 -1.039575  0.271860
2013-01-05 -0.424972  0.567020  0.276232 -1.087401

In [14]: df.tail(3)   # 默认值5
Out[14]: 
                   A         B         C         D
2013-01-04  0.721555 -0.706771 -1.039575  0.271860
2013-01-05 -0.424972  0.567020  0.276232 -1.087401
2013-01-06 -0.673690  0.113648 -1.478427  0.524988
```

展示索引和列标记：

```python
In [15]: df.index
Out[15]: 
DatetimeIndex(['2013-01-01', '2013-01-02', '2013-01-03', '2013-01-04',
               '2013-01-05', '2013-01-06'],
              dtype='datetime64[ns]', freq='D')

In [16]: df.columns
Out[16]: Index(['A', 'B', 'C', 'D'], dtype='object')
```

展示基础数据的NumPy表示：

```python
In [17]: df.to_numpy()
Out[17]: 
array([[ 0.4691, -0.2829, -1.5091, -1.1356],
       [ 1.2121, -0.1732,  0.1192, -1.0442],
       [-0.8618, -2.1046, -0.4949,  1.0718],
       [ 0.7216, -0.7068, -1.0396,  0.2719],
       [-0.425 ,  0.567 ,  0.2762, -1.0874],
       [-0.6737,  0.1136, -1.4784,  0.525 ]])

In [18]: df2.to_numpy()
Out[18]: 
array([[1.0, Timestamp('2013-01-02 00:00:00'), 1.0, 3, 'test', 'foo'],
       [1.0, Timestamp('2013-01-02 00:00:00'), 1.0, 3, 'train', 'foo'],
       [1.0, Timestamp('2013-01-02 00:00:00'), 1.0, 3, 'test', 'foo'],
       [1.0, Timestamp('2013-01-02 00:00:00'), 1.0, 3, 'train', 'foo']],
      dtype=object)
```

### 选择数据

#### 选择单列

```python
In [23]: df['A']
Out[23]: 
2013-01-01    0.469112
2013-01-02    1.212112
2013-01-03   -0.861849
2013-01-04    0.721555
2013-01-05   -0.424972
2013-01-06   -0.673690
Freq: D, Name: A, dtype: float64
```

#### 选择局部

```python
In [24]: df[0:3]
Out[24]: 
                   A         B         C         D
2013-01-01  0.469112 -0.282863 -1.509059 -1.135632
2013-01-02  1.212112 -0.173215  0.119209 -1.044236
2013-01-03 -0.861849 -2.104569 -0.494929  1.071804

In [25]: df['20130102':'20130104']
Out[25]: 
                   A         B         C         D
2013-01-02  1.212112 -0.173215  0.119209 -1.044236
2013-01-03 -0.861849 -2.104569 -0.494929  1.071804
2013-01-04  0.721555 -0.706771 -1.039575  0.271860
```

1. 如果数组不具有相同的rank，则将较低等级数组的形状添加1，直到两个形状具有相同的长度。

2. 如果两个数组在维度上具有相同的大小，或者如果其中一个数组在该维度中的大小为1，则称这两个数组在维度上是兼容的。

3. 如果数组在所有维度上兼容，则可以一起广播。

4. 广播之后，每个阵列的行为就好像它的形状等于两个输入数组的形状的元素最大值。

5. 在一个数组的大小为1且另一个数组的大小大于1的任何维度中，第一个数组的行为就像沿着该维度复制一样

   

## 3. TensorFlow学习简记

### 基本分类

#### 导入MNIST数据集

使用MNIST数据集（60000张灰度图像来训练网络和10000张灰度图像）来评估网络模型学习图像分类任务的准确程度。数据集可以直接从TensorFlow使用，只需导入并加载数据。

```python
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
```

加载数据集并返回四个NumPy数组：`train_images`和`train_labels`是训练集，`test_images`和`test_labels`是测试集。

图像是28×28NumPy数组，像素值介于0~255。labels是一个整数数组，数值介于0~9，代表图片上的数值。每个图像都映射到一个标签。

#### 数据预处理

在训练网络之前必须对数据进行预处理。在馈送到神经网络模型之前，需要将图片的像素值缩放到0到1的范围，为此，需要将像素值除以255，并且要对训练集和测试集以同样的方式进行预处理。

#### 构建模型

构建神经网络需要配置模型的层，然后编译模型。

#### 设置网络层

```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
```

网络中的`keras.layers.Flatten`将图像格式从二维数组（包含着28×28个像素）转换成一个包含28*28=274个像素的一维数组。可以将这个网络层视为其将图像中未堆叠的像素排列在一起。这个网络层没有需要学习的参数，它仅仅对数据进行格式化。

在像素被展平之后，网络由一个包含有两个`keras.layers.Dense`的网络层的序列组成，其被称为稠密连接层或全连接层。第一个全连接层包括128个节点或被称为神经元，第二个全连接层是一个包含10个节点的softmax层，其将返回包含10个概率分数的数组，总和为1，每一个节点包含一个分数，表示当前图像属于10个类别之一的概率。

#### 编译模型

在模型准备好进行训练之前还需要一些配置，这些是在模型的编译步骤中添加的，包括：

- 损失函数：可以衡量模型在训练过程中的准确程度。应该使损失函数最小化以"驱使"模型朝正确的方向拟合。

- 优化器：是指模型根据它看到的数据及其损失函数进行更新的方式。

- 评价方式：用于监控训练和测试步骤。示例中使用*准确率(accuracy)*，即正确分类的图像的分数。

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

#### 训练模型

训练神经网络模型需要以下步骤：

1. 将训练数据（`train_images` 和 `train_labels` 数组）提供给模型。
2. 模型学习如何将图像与其标签关联。我们使用模型对测试集`test_images`进行预测。
3. 我们验证预测结果是否匹配`test_labels`数组中保存的标签。

```python
model.fit(train_images, train_labels, epochs=5)
```

#### 评估准确率

可以得到模型在测试集上的一些指标数值：

```python
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
```

#### 进行预测

模型训练后，可以使用其来预测某些图像。

```python
predictions = model.predict(test_images)
```

这样模型就预测了测试集中每个图像的标签。其中的第一个预测`predictions[0]`可以是：

```python
array([6.6858855e-05, 2.5964803e-07, 5.3627105e-06, 4.5019146e-06,
       2.7420206e-06, 4.7881842e-02, 2.3233067e-04, 5.4705784e-02,
       8.5581087e-05, 8.9701480e-01], dtype=float32)
```

预测是10个数字的数组，对应于图片的10个分类，从中可以看出模型对该图像的预测为9，可以通过检查`test_labels[0]`以得知预测是否正确。

模型可以对单个图像进行预测：

```python
# 从测试数据集中获取图像
img = test_images[0]
```

keras模型经过优化，可以一次性对批量或者一个集合的数据进行预测。因此，即使使用单个图像，也需要将其添加到列表中：

```python
# 将图像添加到批次中，即使它是唯一的成员
img = (np.expand_dims(img,0))
predictions_single = model.predict(img)
print(predictions_single)
```

### 过拟合

在训练周期达到一定次数后，模型在验证数据上的准确率会达到峰值，然后便开始下降，也就是说，模型会过拟合训练数据。尽管通常可以在训练集上实现很高的准确率，但真正需要的是开发出能够很好地泛化到测试数据（或之前未见过的数据）的模型。过拟合的指标是训练准确性高于测试准确性。

与过拟合相对的是欠拟合。当测试数据仍存在改进空间时，便会发生欠拟合。出现这种情况的原因有很多，例如模型不够强大、过于正则化，或者根本没有训练足够长的时间。这意味着网络未学习训练数据中的相关模式。

如果训练时间过长，模型将开始过拟合，并从训练数据中学习无法泛化到测试数据的模式。所以，需要在这两者之间实现平衡。

为了防止发生过拟合，最好的解决方案是使用更多的训练数据。用更多数据进行训练的模型自然能够更好地泛化。如无法采用这种解决方案，则次优解决方案是使用正则化等技术。这些技术会限制模型可以存储的信息的数量和类型。如果网络只能记住少量模式，那么优化过程将迫使它专注于最突出的模式，因为这些模式更有机会更好地泛化。

防止神经网络出现过拟合的最常见方法有获取更多训练数据、降低网络容量、添加权重正则化、添加丢弃层等。

#### 添加权重正则化      

奥卡姆剃刀定律用于神经网络学习的模型表达为：给定一些训练数据和一个网络架构，有多组权重值（多个模型）可以解释数据，而简单模型比复杂模型更不容易过拟合。

在这种情况下，“简单模型”是一种参数值分布的熵较低的模型（或者具有较少参数的模型）。因此，要缓解过拟合，一种常见方法是限制网络的复杂性，具体方法是强制要求其权重仅采用较小的值，使权重值的分布更“规则”。这称为“权重正则化”，通过向网络的损失函数添加与权重较大相关的代价来实现。这个代价分为两种类型：

- L1 正则化，其中所添加的代价与权重系数的绝对值（即所谓的权重“L1 范数”）成正比。
- L2 正则化，其中所添加的代价与权重系数值的平方（即所谓的权重“L2 范数”）成正比。L2 正则化在神经网络领域也称为权重衰减。

在keras中，权重正则化的添加方法是：将权重正则化项实例作为关键字参数传递给层。

如下是L2权重正则化：

```python
l2_model = keras.models.Sequential([
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                       activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                       activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
```

#### 添加丢弃层

丢弃（应用于某个层）是指在训练期间随机“丢弃”（即设置为 0）该层的多个输出特征。假设某个指定的层通常会在训练期间针对给定的输入样本返回一个向量 [0.2, 0.5, 1.3, 0.8, 1.1]，在应用丢弃后，此向量将随机分布几个 0 条目，例如 [0, 0.5, 1.3, 0, 1.1]。

“丢弃率”指变为 0 的特征所占的比例，通常设置在 0.2 和 0.5 之间。在测试时，网络不会丢弃任何单元，而是将层的输出值按等同于丢弃率的比例进行缩减，以便平衡以下事实：测试时的活跃单元数大于训练时的活跃单元数。

在keras中，可以通过丢弃层将丢弃引入网络中，以便事先将其应用于层的输出：

```python
dpt_model = keras.models.Sequential([
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
```



## 4. mnist_cnn.py运行

### 代码浅解

```python
'''Trains a simple convnet on the MNIST dataset.
MNIST:手写数字数据集，包含了60000张的训练灰度图像和10000张的测试灰度图像，其中每一张图片包含28X28个像素点。
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

#引入模块及包
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K   #此处选用的是TensorFlow

#每次喂给神经网络的样本数
#即每次将一部分样本喂给神经网络，让神经网络一部分样本一部分样本地迭代（batch梯度下降法）
batch_size = 128

#数据集有10个分类，即数字0~9
num_classes = 10

#周期，即所有训练数据的循环次数
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
#加载训练集和测试集数据，包括图像样本和标签
(x_train, y_train), (x_test, y_test) = mnist.load_data()

''' "channels_first"，即通道维靠前，会将图片表示为（样本维，通道维，图片高，图片宽）
如把100张RGB三通道的16×32彩色图表示为如形式（100,3,16,32）
而"channels_last"，会将图片表示为如形式（100,16,32,3），TensorFlow是这种
'''
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

#将数据转为float32类型
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#将数据从[0-255]缩小到[0-1]的范围
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
#将标签转换为one-hot格式，即只在对应标签那一列为1、其余为0的布尔矩阵
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#搭建网络结构，Sequential模型是多个网络层的线性堆叠，通过add()逐个将layer加入模型
model = Sequential()
#二维卷积层，对二维输入进行滑动窗卷积
'''当使用该层作为第一层时，需要提供关键词参数input_shape
此处后端使用TensorFlow，所以input_shape是"channels_last"模式
卷积核（即输出的维数）为32，卷积核的宽度和长度均为3，激活函数为ReLU
'''
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
#卷积核（即输出的维数）为64，卷积核的宽度和长度均为3，激活函数为ReLU
model.add(Conv2D(64, (3, 3), activation='relu'))
#Pooling对于输入的Feature Map，选择某种方式对其进行压缩
#Pooling能够减少参数，并且能在当像素在邻域发生微小位移时不受影响，增加鲁棒性
model.add(MaxPooling2D(pool_size=(2, 2)))
#Dropout能够方式CNN过拟合
#按一定的概率p对weight参数进行随机采样，将这个子网络作为此次更细的目标网络
#如果整个网络有n个参数，那么可用的子网络个数为2^n
model.add(Dropout(0.25))
#Flatten把多维的输入一维化，常用在从卷积层到全连接层的过渡
model.add(Flatten())
#Dense是常用的全连接层，所实现的运算是output=activation(dot(input, kernel)+bias)
#此处输出维度为128，激活函数为ReLU
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
#将Densen作为输出层，激活函数为softmax,输出维度与标签的one-hot维度一致
#softmax激活函数能够输出是有效的概率分布
model.add(Dense(num_classes, activation='softmax'))

'''Theano和TensorFlow都允许在python中定义计算图，然后在CPU或GPU上有效地编译和运行
在编译模型时需要提供损失函数和优化器
此处的损失函数是categorical crossentropy，是一种非常适合比较两个概率分布的损失函数
metrics是指标列表，对分类问题，我们一般将该列表设置为metrics=['accuracy']
'''
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

'''训练模型
fit函数返回一个History的对象，其History.history属性记录了损失函数和其他指标的数值随epoch变化的情况
verbose为日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
validation_data为测试集数据
'''
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

#evaluate函数按batch计算在某些输入数据上模型的误差,返回一个测试误差的标量值
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

### 运行结果

Using TensorFlow backend.

Downloading data from https://s3.amazonaws.com/img-datasets/mnist.npz

11493376/11490434 [==============================] - 942s 82us/step

x_train shape: (60000, 28, 28, 1)

60000 train samples

10000 test samples

Train on 60000 samples, validate on 10000 samples

Epoch 1/12

60000/60000 [==============================] - 246s 4ms/step - loss: 0.2644 - acc: 0.9180 - val_loss: 0.0599 - val_acc: 0.9807

Epoch 2/12

60000/60000 [==============================] - 228s 4ms/step - loss: 0.0882 - acc: 0.9740 - val_loss: 0.0414 - val_acc: 0.9854

Epoch 3/12

60000/60000 [==============================] - 232s 4ms/step - loss: 0.0666 - acc: 0.9804 - val_loss: 0.0358 - val_acc: 0.9872

Epoch 4/12

60000/60000 [==============================] - 226s 4ms/step - loss: 0.0555 - acc: 0.9833 - val_loss: 0.0294 - val_acc: 0.9904

Epoch 5/12

60000/60000 [==============================] - 224s 4ms/step - loss: 0.0481 - acc: 0.9850 - val_loss: 0.0354 - val_acc: 0.9888

Epoch 6/12

60000/60000 [==============================] - 219s 4ms/step - loss: 0.0424 - acc: 0.9871 - val_loss: 0.0301 - val_acc: 0.9901

Epoch 7/12

60000/60000 [==============================] - 212s 4ms/step - loss: 0.0390 - acc: 0.9885 - val_loss: 0.0276 - val_acc: 0.9910

Epoch 8/12

60000/60000 [==============================] - 212s 4ms/step - loss: 0.0348 - acc: 0.9895 - val_loss: 0.0284 - val_acc: 0.9903

Epoch 9/12

60000/60000 [==============================] - 214s 4ms/step - loss: 0.0336 - acc: 0.9897 - val_loss: 0.0286 - val_acc: 0.9912

Epoch 10/12

60000/60000 [==============================] - 216s 4ms/step - loss: 0.0314 - acc: 0.9900 - val_loss: 0.0262 - val_acc: 0.9911

Epoch 11/12

60000/60000 [==============================] - 213s 4ms/step - loss: 0.0290 - acc: 0.9910 - val_loss: 0.0273 - val_acc: 0.9924

Epoch 12/12

60000/60000 [==============================] - 213s 4ms/step - loss: 0.0273 - acc: 0.9917 - val_loss: 0.0248 - val_acc: 0.9934

Test loss: 0.024847858635666308

Test accuracy: 0.9934



## 5. MNIST in Keras.ipynb 运行

```python
%matplotlib inline

# 代码与上类似，故注释略

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7,7) # Make the figures a bit bigger

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

nb_classes = 10

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("X_train original shape", X_train.shape)
print("y_train original shape", y_train.shape)

for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_train[i], cmap='gray', interpolation='none')
    plt.title("Class {}".format(y_train[i]))

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu')) # An "activation" is just a non-linear function applied to the output
                              # of the layer above. Here, with a "rectified linear unit",
                              # we clamp all values below 0 to 0.
                           
model.add(Dropout(0.2))   # Dropout helps protect the model from memorizing or "overfitting" the training data
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax')) # This special "softmax" activation among other things,
                                 # ensures the output is a valid probaility distribution, that is
                                 # that its values are all non-negative and sum to 1.
        
model.compile(loss='categorical_crossentropy', optimizer='adam')

# 此处有修改
model.fit(X_train, Y_train,
          batch_size=128, nb_epoch=4,
          verbose=1,
          validation_data=(X_test, Y_test))

# 此处有修改
score = model.evaluate(X_test, Y_test, verbose=0)
# 此处有修改
#print('Test score:', score[0])
#print('Test accuracy:', score[1])
print('Test score:', score)

# The predict_classes function outputs the highest probability class
# according to the trained classifier for each input example.
predicted_classes = model.predict_classes(X_test)

# Check which items we got right / wrong
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]

plt.figure()
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]))
    
plt.figure()
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test[incorrect]))
```

### 运行结果

X_train original shape (60000, 28, 28)

y_train original shape (60000,)

Training matrix shape (60000, 784)

Testing matrix shape (10000, 784)

Train on 60000 samples, validate on 10000 samples

Epoch 1/4

60000/60000 [==============================] - 17s 282us/step - loss: 0.2529 - val_loss: 0.1070

Epoch 2/4

60000/60000 [==============================] - 12s 201us/step - loss: 0.0996 - val_loss: 0.0795

Epoch 3/4

60000/60000 [==============================] - 12s 205us/step - loss: 0.0712 - val_loss: 0.0773

Epoch 4/4

60000/60000 [==============================] - 11s 189us/step - loss: 0.0593 - val_loss: 0.0637

Test score: 0.06372718122184742

![img](data:image/png;base64,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Dh4t2U38SECr542U+kSyoT6RGuXzdV9MO3iUeKlPJBvqE6lRvdekzGwjyUuQ7MSwCYBRZvZ2wSqTBkF9ItlQn0htSrpbJK2eV7w3zaxbsReiPql46hOpUyl+3SciIlJUGqRERCRaGqRERCRaGqRERCRaGqRERCRaGqRERCRaGqRERCRaGqRERCRaGqRERCRaGqRERCRaGqRERCRaGqRERCRa+RxPSgrshhtucPmmm25yeYst/GeKo48+2uWXXnqpKHWJSPm1aNHC5ebNm7v8gx/8wOUdd9zR5bvuusvlDRs2FLC64tGalIiIREuDlIiIREuDlIiIREtzUmU0cOBAl6+99lqXv/7664y3L+UBK0WkuDp16uRy+H5w1FFHubz//vvndP/t27d3+bLLLsvp9uWiNSkREYmWBikREYmWBikREYmW5qTKqGPHji5vs802ZapECumII45w+eyzz3a5V69eLu+3334Z7+/qq692eenSpS736NHD5bFjx7o8ffr0jPcvpbHPPvu4fMUVV7h81llnubztttu6TNLlxYsXu7x27VqX9913X5dPP/10l++9916X582bV1PZZac1KRERiZYGKRERiZYGKRERiZbmpEromGOOcfnSSy/NeP3wO+J+/fq5vGzZssIUJnn58Y9/7PLw4cNd3mGHHVwO5xZefPFFl8N9rt15550Zlx/eX3j7M844I+PtpTBatWrl8u233+5y2CfhvvjqMn/+fJePO+44l5s2bepy+P4R9mGYY6U1KRERiZYGKRERiZYGKRERiZbmpIoo3H7loYcecjn8DjsUzkUsWrSoMIVJTrbc0v836datm8t//OMfXW7WrJnL06ZNc/nmm292+ZVXXnF56623dvnxxx93+dhjj81Y78yZMzNeLsVxyimnuPyf//mfed3fwoULXe7Tp4/L4XZSnTt3zmt5sdKalIiIREuDlIiIRKvOQYrkKJLLSc6udl5bklNIzk//bVPcMiV26hPJhvpEcpXNnNRoACMAPFztvCEAnjezYSSHpPnaGm7bqA0YMMDl73znOxmvH24v8/DDD9d8xTiNRgPtk3Dfew888EDG60+ZMsXlcPuYNWvWZLx9eP265qCWLFni8pgxYzJev8xGo4H2yWmnnZbT9d9//32XZ8yY4XJ4PKlwDioU7quvoahzTcrMpgFYGZzdH8Dm/wljAJxc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)

![img](data:image/png;base64,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0pGUiIhES42UiIhES42UiIhES31SVXTIIYcEcevWrYM4/TyqF154oeI5SWFHHHFEEO+88855x588eXIQX3zxxeVOSZqhb3/720HsXPhMx/Hjx1cznWjoSEpERKKlRkpERKKlRkpERKKlPqkKWn/99YP4oIMOCuLPP/88iNN9F1988UVlEpO80tc9XXDBBUGc7ktMe+2114JY9+aT+my22WZBvPfeewdx+l6dDzzwQMVzipGOpEREJFpqpEREJFpqpEREJFrqk6qgc889N4h32mmnIE7fi+v555+veE5S2DnnnBPEu+66a97xH3zwwSDWdVGSxcknnxzEXbt2DeLHHnusitnES0dSIiISLTVSIiISLTVSIiISLfVJldGhhx4axL/85S+D+OOPPw7iyy67rOI5SfF+9rOfFTX+mWeeGcS6Lkqy6N27d97hy5Ytq1ImcdORlIiIREuNlIiIREuNlIiIREt9UiVI3+Ptf//3f4O4VatWQfzoo48G8YsvvliZxKSqOnXqFMSl3nNxxYoVeeeXvnfgRhttlHd+G2+8cRAX2+e2du3aID7//POD+NNPPy1qfpI47LDD8g7/y1/+UqVM4qYjKRERiZYaKRERiVbBRsrMeprZJDObaWYzzGyEf72TmU00s3f8746VT1dipTqRLFQnUqwsfVJrgHOcc6+Y2YbANDObCJwMPOWcu9rMRgIjgfPzzKfmpfuY0vfe22KLLYL43XffDeL0dVPNTIutkzfeeKOs87v33nuDeMGCBUG86aabBvFxxx1X1uUXsnDhwiC+4ooripm8xdbJgAEDgjj9PCmpX8EjKefcAufcK/7vlcBMYHPgSGCsH20sMLhSSUr8VCeShepEilXUt/vMrA+wE/ASsKlzbgEkhWdmXRuYZhgwrLQ0pZaoTiQL1YlkkbmRMrP2wH3A2c65j80s03TOudHAaD8P15gkpXaoTiQL1YlklamRMrPWJAV1l3Pufv/yh2bWze/1dAMWVSrJWPTt2zeId9lll7zjp69HSfdRNTfNpU7S17MdeeSRVV3+McccU9L0a9asCeIvv/wy7/gTJkwI4qlTp+Yd/5lnnmlcYl5zqZNiHXXUUUGc7uN+9dVXg3jKlCkVz6kWZPl2nwG3AjOdc9flDJoADPV/DwUeKn96UitUJ5KF6kSKleVIai/gJODvZvaaf+0C4GrgHjM7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mpv11lsviDfffPMgHjduXDXTKahv3755h0+fPj3vcKkN/fv3zzs8fQ9G9UmJiIg0MTVSIiISLTVSIiISrWbdJzV8+PAgTj93JzaHH354EO+0005BnM4/Haf7pKR+K1euDOL0Pc922GGHIO7UqVMQL12afvp5eXXtGj6YNv28q7Rnn322kulIhQwYMCCITzjhhLzjr1ixIojnz59f9pxipCMpERGJlhopERGJlhopERGJVrPuk0r38TS1Ll26BPG2224bxBdccEFR81u8eHEQ67lB2axevTqI08/dOvroo4P4kUceCeL0c76Ktf322wfxlltuGcTp50c55/LOL/a+Vqlf586dg7jQdZATJ06sZDrR0pGUiIhES42UiIhES42UiIhEq1n3ScXmwgsvDOIzzjijqOnnzJkTxEOHDg3i999/v1F5tXQXX3xxEKfvoXjooYcGcan39ks/lyzd55R+XlQhY8aMKSkfaRqFrn9L36vvd7/7XSXTiZaOpEREJFpqpEREJFpqpEREJFrqk6qgRx99NIi32Wabkub35ptvBrHu2VYes2bNCuJjjz02iHfccccg7tevX0nLGz9+fN7hY8eODeITTzwx7/jp674kTj169AjiQvfqS9+bb+rUqWXPqRboSEpERKKlRkpERKKlRkpERKLVrPuk0te7FLo31sEHH5x3+OjRo4O4e/fuecdPL6/Ue6zFdi/CliL9vKl0XG7vvfdeUeOn7wU4ffr0cqYjZbLnnnsGcaHt0YMPPljJdGqGjqRERCRaaqRERCRaBRspM+tpZpPMbKaZzTCzEf71TmY20cze8b87Vj5diZXqRLJQnUixsvRJrQHOcc69YmYbAtPMbCJwMvCUc+5qMxsJjATOr1yqxbv55puD+Jprrsk7/sMPPxzEhfqQiu1jKnb8W265pajxm1jN1kls0n2p6TitxvqgWmydpJ8flZa+p+P1119fyXRqRsEjKefcAufcK/7vlcBMYHPgSKDuqsOxwOBKJSnxU51IFqoTKVZRfVJm1gfYCXgJ2NQ5twCSwgO6ljs5qU2qE8lCdSJZZP4Kupm1B+4DznbOfVzoFETOdMOAYY1LT2qN6kSyUJ1IVpkaKTNrTVJQdznn7vcvf2hm3ZxzC8ysG7Covmmdc6OB0X4+rr5xKuX+++8P4nPPPTeIu3TpUs10WLx4cRDPnDkziIcN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7wD7+1haMNWJZKE6kWIVPN3nnDu+gUH7lTkXqWGqE8lCdSLFalb37ivk0EMPDeL0vbeWL18exDfffHNJy0v3VQwcODCI08+TSRs/fnxJy5fGWW+99YI43Tf429/+Nu/0n332WRDffvvtQXzMMccE8ZZbbpl3fum+Dd27rzIGDw6/+T5y5Mggfv/994N47733DuIVK1ZUJrEWTrdFEhGRaKmREhGRaKmREhGRaDWrPqnrr78+iL/73e8Gcffu3YN4n332CWKz8FZSRxxxREn5pOdX6D6J7733XhCnr6eR6khf15SW7tt88MEHi5p///713tauQennDq1ataqo6SWbPffcM+/wV199NYjTz3OSytCRlIiIREuNlIiIREuNlIiIRKvg86TKurAq3xCyY8fw2Wnpe7IddNBBQXzuuecGcfq5PmPHjqUY6edbvf7663nHv/POO4N46NChDYzZZBp8TlA5NfWNQ4899tggHjduXBCnnx81ZMiQIP7Wt74VxEcddVQQp6+T+vjjj4M4Xbfpe/Ol+1LffPNNIlOTdZL+vHfu3DmI//WvfwXxr3/96yBO32PxtddeK2N2zU/ZniclIiLSVNRIiYhItNRIiYhItJp1n1RTS9+Tbfbs2UGcPmd94IEHBvHixYsrk1jj1WRfQ7E6deoUxOn/20YbbRTExV4P99e//jWIzzjjjCB++OGHg3irrbYK4t///vdBPHz48LzLawI1WSeFnkdXSHr8W265JYjT17v16tUriNN1NmPGjLzL22677YL4hRdeCOLYr+NSn5SIiNQ8NVIiIhItNVIiIhIt9UlV0JgxY4L4pJNOCuL0dVoTJ06sdEqlqsm+hlINGjQoiNPP+Ur3UaU/UzfccEMQn3/++UGcfv7UlVdeGcTp5xrNnTs3b37vvvsuTawm6+Taa68N4p/97GflnH3FpfuwJ0+eHMTp6/mamvqkRESk5qmREhGRaKmREhGRaKlPqozS92S7++67g3jlypVBnH7e1SuvvFKZxMqnJvsayi3dB3TCCScE8fLly4P4v/7rv4K40POg1l9//SD+05/+FMTp55xFeM/HmqyTVq1aBfFOO+0UxOn/w7rrho/j69mzZxCvs07THgOkt+2XXHJJEF9++eVVzObr1CclIiI1T42UiIhES42UiIhEa93Co0hWBx98cN7h6Xuy1UAflNQjfe+9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)

![img](data:image/png;base64,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kfo06mWRdAy56k1zzhV72f+VpjqpeqoTKcg5V8zXFzXogxMiUgbt2rWL8u233x7l2bNnR/nkk0+ueJtEskKXRRIRkcxSJyUiIpmlTkpERDJLH5yQUmhAvAI22mijKL/66qtRXrJkSZR79+4d5U8++aQyDVt5qhMpqNgPTmhPSkREMkudlIiIZJY6KRERySydJyWSce+//36Uly5d2kQtEWl82pMSEZHMUiclIiKZpU5KREQyS2NSOQ499NAoDxoUn+pR6jXTVlkl3gZ47LHHovzggw9GefTo0VH+6KOPSlqeNE8PPfRQlD/77LMmaolI49OelIiIZJY6KRERySx1UiIiklktakzq3HPPjfLPf/7zKK+++upRbtWqVZRLvc7h8uXLo7ztttvmzZtvvnmUDz/88JKWJ9Xppz/9aZTT86Auv/zyxmyOlMmaa64Z5f79+0e50P/3iBEjolzq+8/8+fOj/J3vfCfKc+bMKWl+TUV7UiIiklnqpEREJLPUSYmISGY16zGp888/P8qnnnpqlFu3zr/6CxcujPL9998f5QceeCDK6VjC+PHji2pnrfSYddeuXaP84YcfljQ/yaZ11103ykOHDo1yeh7U66+/XvE2ScOlY0xnnnlmlDfeeOOS5peOQT3//PNRbtOmTZQ32WSTKHfv3j3K66yzTpQ1JiUiItJA6qRERCSz1EmJiEhmNasxqfXXXz/Kw4YNi3L6vTy33nprlG+44YYof/nll1F+66238i4/HQNLzZ49O8qffPJJlLfaaqso9+vXL8oak2oedttttyin59OcccYZjdkcWUnptT6vueaaKKfnXab/7/fcc0+Up0+fHuV///vfUU7HkNIx9bfffjvv8g877LAoP/nkk1QD7UmJiEhmqZMSEZHMUiclIiKZZaVeD6pBCzOr6MJefPHFKKfnDdx3331RPvDAA8u6/N69e0c5PYacLj+9dmB6DHrKlClRPuaYYxraxIaa5pwbVHiyhql0nTS2bt26RXnq1KlRXmONNaI8YMCAKL/33nuVaVjlNMs6adeuXZTT8yTTa32ed955Uf7vf/8b5SVLljSoPemYUzrmnt6/00475W1PY3POWTHTaU9KREQyS52UiIhkVsFOyszGmNn7ZvZSzm1dzGyimc0MvztXtpmSdaoTKYbqREpVzHlSY4G/ADfm3DYSeNQ5N8rMRoZ8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IpJZDTlPSgrYaKONonzttddG+dZbb43yddddV/E2iYhUE+1JiYhIZqmTEhGRzFInJSIimaUxqTJKx6D+/ve/R3m99daLcr9+/aKsMSkRkZj2pEREJLPUSYmISGapkxIRkczSmFQDjBgxIm9ed9118z5+zpyV+QJTEZGv69GjR5TvuOOOKO+4445RPvPMM6N84YUXVqZhDaQ9KRERySx1UiIiklnqpEREJLM0JlWC1q3jp2vTTTeNct++faOcfqHk66+/HuUjjjiijK0Tkeaka9euUV5ttdWi3K1btyi/8cYbUb733nujPGhQ/D2UJ598cpQ1JiUiIlIidVIiIpJZ6qRERCSzNCZVguOPPz7Kxx57bEmP/+ijj6I8b968BrdJClt77bWjfP7550d5u+22i/Irr7wS5T/96U9RNrMoz549O8rvvvvuSrVTWpZDDjkkygcffHCUhwwZEuV0jDt15ZVXRvmkk06K8mabbRblAw44IMrrrLNOlOfPn593eY1Fe1IiIpJZ6qRERCSz1EmJiEhmaUwqj549e0b5uOOOi3I6NrHKKnGfv3z58ij/6le/KmPrWq599tknyqNGjYpyer5aen5ber7Je++9F+XNN988yj/4wQ+inL7OS5cujfKyZcuifM8990T51ltvJZ+nn346yp988kne6SUbvvWtb0X56KOPjvJhhx0W5bSOOnfunHf+CxYsiPI777wT5eeffz7v47feeusop+dVdejQIe/jm4r2pEREJLPUSYmISGapkxIRkczSmFQe6fdBffOb34xyet5COgb1wAMPRPnZZ58tY+tarl//+tdRTs//mDFjRpTT85bSY/cTJ06McjqGlY4JrbrqqlHedttto7znnntGuU+fPlG+6667otyxY8cop+fTXXzxxVFO6+rVV19FKm/NNdeM8umnnx7lo446KsrptfVS6XlK6RjWRRddFOU11lgjym+++WaU0zrYd999o5y+f6X/R7Nmzcrb3qaiPSkREcksdVIiIpJZBTspMxtjZu+b2Us5t3Uxs4lmNjP8zv/ZSWn2VCdSDNWJlMoKXQ/KzL4LLAZudM5tHm67GPjYOTfKzEYCnZ1zp+ebT3hc/oVlTHq+zKOPPhrltdZaK8rpeVMffvhhlHfdddcop2MnVWCac25QXXc0Zp2k3+M1duzYKL/88stRTscKmlp6Pkw6pnbggQdGeejQoVFOr/l44oknRnnKlCkNbWJDZaJOyu2GG26Icvq6pP//c+fOjfKYMWOinJ7f98UXXzSofel5Wg8//HCUf/Ob30T57rvvjvLChQsbtPxSOees8FRF7Ek556YCHyc3DwHGhb/HAfuX1DppdlQnUgzViZRqZcekujvnagDC7/wfY5GWSnUixVCdSL0q/hF0MxsGDKv0cqS6qU6kGKqTlqfgmBSAmfUDHsw5hvwasLNzrsbMegCTnXMbFzGfqhqTSl177bVRTr9PKj0mnT636eNPOOGEMrauUdQ71gBNVyfpGM+XX34Z5c8//7yU2TW6tm3bRrlfv35RTs+XSb9naMKECVH+5z//GeUbb7wxyul5WBWQyTopVf/+/aP83HPPRbldu3ZRHj16dJR/+9vfRvmDDz4oY+u+3r5JkyblzUceeWRZl99QZRuTqsd4oHbUcChw/0rOR5o31YkUQ3Ui9SrmI+i3AY8DG5vZPDM7FhgFDDazmcDgkKUFU51IMVQnUqqCY1LOuUPruWu3MrdFqpjqRIqhOpFSFTUmVbaFVfmYVHoNtvTaWYXGpGpqaqKcXlur0PfBZEDesYZyqfY6KST9Pqxrrrkmyuk14tq3b593fum1BdNrvKXXEkzP96uAZlEn6dhf+r1gjzzySJTT57nSfv7zn0f5l7/8ZZQHDhwY5Y8/Tj/537QqPSYlIiJSceqkREQks9RJiYhIZun7pEqQXovriiuuiPIpp5wS5fT7pXr27Bnl8ePHR7lv374NbaJUgbQOevXqlXf6F198Me/0Z511VpQXLFgQ5UYYg2qRunfvHuX0/3fOnDllXV56rdD0mpS///3vo5y1MaiVpT0pERHJLHVSIiKSWeqkREQks3SeVAOk56PsvffeUU6v5bX66qtH+auvvopyem2/9Ptnpk+fvlLtLKNmcf5LU2vTpk2U09c5/T6pH/7wh1FOryE3f/78MrauLJpFnXTp0iXK//jHP6K89dZbR/mNN96I8mGHHRblZ555pkHt6dq1a5TTazZuv/32UW7o91NVms6TEhGRqqdOSkREMkudlIiIZJbGpCro3nvvjfLOO+8c5Y4dO+Z9/HvvvRflAQMGRLnc309ThGYx1pA16fkv6fku6bX+nn322Sin3xO0ePHiMrZupbSIOrnwwgujfPTRR0e5W7f4C4avv/76KJ9zzjlRfuedd6LcqVOnKN99991RTq/Nl36/1MKFC+todXZoTEpERKqeOikREcksdVIiIpJZGpNqRMcff3yUr7zyyrzTp99Pte6660Y5PYbdCFrEWENTa906vqTm4MGDo5yOTTzxxBNRPvXUU6PcBOfXtYg6Sc+T7NevX5T/9re/RXm99daLcjqm/OSTT0Z53rx5UR4+fHiU77zzzigfemh93yeZTRqTEhGRqqdOSkREMkudlIiIZJa+T6oRPf/8803dBKkCy5Yti/JDDz0U5fSacQ888ECUf/vb30Y5HQttgvPrmqX0PKT0/3vTTTeN8iWXXBLlYcOGRXnfffeNcjomnX5+YJVV4n2Mtm3bRnnp0qV1NbvqaE9KREQyS52UiIhkljopERHJrBY1JrXTTjvlvX/KlCllXd5PfvKTKJ9xxhlRTo85p9JjztIytW/fPsqDBsWnIKXfe/SDH/wgyukYVp8+fcrYOqlPOiY0YsSIKF911VVRvuOOO6K8xRZb5J1/+r1jy5cvj/IxxxwT5SVLluSdX1bpXVBERDJLnZSIiGSWOikREcmsZj0m1bNnzyjff//9UZ46dWqU0+9/KeT73/9+lNMxr+7du0e5VatWUU7Pe0ivsTZkyJAoz58/v6T2SXVI6/Swww6Lcnqe0wYbbJB3fl988UWU//GPfzSgdVIp6fdPffOb34zyCy+8EOVf/OIXUR45cmSUf/SjH0X5z3/+c5Qfe+yxlWpnU9OelIiIZJY6KRERyayCnZSZ9TGzSWb2ipnNMLMR4fYuZjbRzGaG350r31zJKtWJFEN1IqUq+H1SZtYD6OGce9bMOgLTgP2Bo4CPnXOjzGwk0Nk5d3qBeTXq97+k54O8+eabaXui3NDv1io0v0WLFkX59NPjpyu9BltNTU2D2lMB9X5PUDXXSaVtt912UT744IOjnJ7P0rFjx5Lmn15D7pRTTonyDTfcUNL8ykB1Uoe11loryrNmzYpy+v1U3/jGN6L8+uuvR3nAgAFRTsecdt9997z3N7WyfZ+Uc67GOfds+HsR8ArQCxgCjAuTjcMXmrRQqhMphupESlXSmJSZ9QMGAk8C3Z1zNeALDyjto3HSbKlOpBiqEylG0R9BN7MOwN+Ak51znxa6pE/O44YBwwpOKM2C6kSKoTqRYhXVSZlZG3xB3eKcuyfc/J6Z9XDO1YTjzO/X9Vjn3GhgdJhPox5D/uqrr6Kcjgl16tSprMubN29elJ977rkoX3HFFVGeNGlSWZff1Kq1Thrq8MMPj/Iuu+wS5fT8lQ4dOuSd34wZM6L8xBNPRHn27NlRvu6666L84Ycf5p1/U2updXLBBRdEOX3/ufvuu6OcjkH16NEjyuPHj4/yww8/HOX0/adaFfPpPgOuB15xzl2Wc9d4YGj4eyhwf/pYaTlUJ1IM1YmUqpg9qe2BHwMvmlntJRHOBEYBd5rZscDbwEGVaaJUCdWJFEN1IiUp2Ek55/4D1HfAeLfyNkeqlepEiqE6kVI162v3vfvuu1Hef//4U60DBw7M+/if//znUZ48eXKUX3zxxShffvnlJbZQmoN0bGCbbbaJ8tNPPx3lV199NcoPPvhglOfMmRPll19+uaFNlAyYOXNm3vvT893S86ruvPPOKPfq1SvK6bVEq/X7o1K6LJKIiGSWOikREcksdVIiIpJZBa/dV9aFVdl5DfI19V6TrZxUJ1VPdVKHtm3bRjkda0zPg1q6dGmU27dvH+X0mow//elPo7x8+fKVamdjKdu1+0RERJqKOikREcksdVIiIpJZzfo8KRGRrEjHmA444IAoP/TQQ1FOL7p75plnRvnqq68uY+uyS3tSIiKSWeqkREQks9RJiYhIZmlMSkSkCTz//PNR7tmzZxO1JNu0JyUiIpmlTkpERDJLnZSIiGSWOikREcksdVIiIpJZ6qRERCSz1EmJiEhmqZMSEZHMUiclIiKZpU5KREQyS52UiIhkVmNfu+9DYA7QNfydVWpf3fo20nJUJ+WhOsmGLLcv8zVizrlKNqTuhZo945wb1OgLLpLalw1ZX0+1Lxuyvp5Zbl+W21ZLh/tERCSz1EmJiEhmNVUnNbqJllsstS8bsr6eal82ZH09s9y+LLcNaKIxKRERkWLocJ+IiGSWOikREcmsRu2kzGwvM3vNzGaZ2cgyzK+fmTkzax3yQ2Y2tMR5jDGz983spZzbupjZRDObGX53Th5zjpnd3ND2F9G2PmY2ycxeMbMZZjbCzCab2Un52lftslYnK1MjYZqmrJNzzOxO1UlJ82uJdTLWzC7Ncp1UvJMys7fMbImZLQYeBF4AtgYONbNNy7ks59z3nHPjimzT7iGOBfZKJhkJPOqc2xB4NOSimdluZvaqmX0eiqL4E9fM2oainQm8BmwMPA78CDgBaAfs3ZD2lcLM2pnZVWb2oZktNLOpFVpOlutkLGWukTD/H4U3jEVm9rKZ7V/i4w8zs2eAV4HNgbeAk/F1sjawWUPbWEJbvmNmT4V1ecHMdqjQclpUnZjZdqHj+NjMPjCzu8ysRwmPt7BR+xL+/WRT4CXgGHydrAF8pyFtLIWZ7Wpmz5rZp2Y228yGFXpMY+1J7QcMBv4DbAKcDtwODKmdIDyZjX740Tk3Ffg4uXkIUFuc44Ci3zzMrCtwD3AW0AV4BrijhCbdDXwfOAzohH++pgHbAa8AqwIDVrZ9K2E0fj02Cb9/UcFlZbJOyl0jAGbWC7gZOAX/Ov8KuNXMuhX5+FOAy4ELgG5AT+AqYA98nXQEejekjcUysy7AeOASYE3gYuCBCm6Rt5g6ATrj/wf74a/SsAi4oYTHXwGMAE4K81oPuA/YBV8n7fAbwo1RJ22Ae4Fr8Z3jwcBlZvatvA90zlX0B791tztwIPBXfCE/CPwYeAc4H/gvsAToHxp/PVAT7j8PaBXm1Qq4FH8Zj9n4LQEHtA73TwaOy1n2T/AvxCLgZWBL4CZgeVjeYuA0fAG8ATwGLAC+AnbOmc9CYEqYz0TgL8DN9azvMOCxnNw+LOsbRTxXu4dp+9RxXz/gbeDfwOfhtg2Af4X1+RC4BVgz5zGnh+dwEX4rardw+zb4zvNT4D3gsnras3GYppPqhAvxW6DbhTpxwPO1dQJ8gn8DKLZOtgXeT277APh2Ec/VGqFNB+WpkwuBpTm33xXWZyEwFdgs5769w3ovCs/lL8PtXcNrsAD/5vtvYJU6lrkvMCO57XXgWNVJw+qkjvXfElhU5LQb4t/LtslTJ7cAX4TbOofncnlo54NA75zHHBWep0XAm8Dh4fb+YX0Whufyjnra0z08H+1ybnsaODTvejTim89BwG3ADODcUFTzwhO1Gf46gm3wvfy1+Df3bsBTwPFhXsPxhzb64LfqJ9VXVGF57+APBVh4Ivvmtil5w1iG/2ddJRTbR8Da4f5lwGX4vZjvhhepvjefK4Crk9teAg4o4rkaBUyp4/YO+L2pH4Z1rO2k+uO3KD/BH96ZClwe7tsYmAv0zCnKDcLfjwM/zpn3dvW050jgReCPofheLGY9mmOdhOfv1VAXe+PfuAfX1kl4DR4voU5a4f+xvx/+3j+sZ/sinqu9Qk22zlMn5xB3UseENq6K3wObnnNfDbBjzhvVluHvC4FrwvPdBtiRcNpKstz9gJeT22YCf1SdNKxO6lj/k4Enipx2ODCnwPvJWFZ0UmsBB4Q2dsRv2NwX7muP32DdOOQehA2d8Dr8Gv/euRqwQ5423YrfGGgFfBt4nzo2yqPHVOINp46iWhxeiCX4QxKrA2fge+Xf50zbHfgSWD3ntkOBSeHvfwHDc+7bI09RTQBG5Cv0nDwKWJCTXwsFOxQYFJbRPuf+W+srKvxW26jktv8CRxXxXF0H3J7c1iasyyk56zgf6JFTLK+Fv/cHngt/9w8FsDvQJpnnVOB3QNcC7TkzrPs5QFtgp/BabtLS6gT/5jMfuCmnRnqEx5+E3xNfVmydhPuPDeu8DPgc2KfI5+pwYH6BOjkHv2VbV52sGZ6PNUJ+GzieZI8Z+D1wP9C/QHvWwr8ZHxraMRS/NX6t6qThdZIz3Rb4Pdodi3yufk3SodVRJ2PxHWhddTIA+CT83T68xgfkPp/hvhvxhyR7F9Gm/fBHb5aFn58UekxjHbPdH7+FVoPfPf8KOAS/dT43Z7q++CexxswWmNkC/FZQ7XH6nsn0c/Issw++IIrRC+iUs8y+wPb4F+xoYIlz7rMil7sYP8aQqxP+n6qQj8IyAX9cHd/pveKcuyxnuunA0DB+8TDQ08w+xY9xdAVwzs3Cb3WdA7xvZrebWc/w+GOBjYBXzexpM9u3nvYsAf4POM85t9Q5NwXfee9RxLqsjKzXSRvgoJwamQ3sAHwPf2jnk2LrJAy0XwzszIoNgL+a2YAi2vER0DXnU2j11ck8fJ20wm/tdgt18la4v2v4fQB+q3+OmU0xs2+H2y8BZgH/DIPcdQ6oO+c+wo+9nIJ/A9oLeCQsvxJaTJ3UMrP+wEP4jvLfRbaj2PeT1/F10g6/57lOqJOpwJpm1iq092D83lmNmf3dzL4RHn8afu/yKfOfGjymnnX4Bn58/kh8zW8GnGZm++RbiUYbWHTOLQNOxPfirwB34rceXc5kc/FbPl2dc2uGn07Ouc3C/TX4Yqm1bp5FzsWP2dTZnNo/zOw2Vgy4LgZOxXda/8a/mQ8A2ppZ+yKXOwP430BgeNwG4fZCHgG2MbPeIW+PP4yxq5lNN7Pp+MMSD+EPIczEby0OcM51Ao7AF4tfSedudc7tgP9HccBF4faZzrlD8f+sFwF3J+tX64Ui2lxWWayTUCOP4/dAlrGiRh4D3sX/w10CdC6hTgYAU51zzzjnljvnngaexO/5FvI48AUrBrjrqpMN8TU3mBWHqXbBj9H0C48zAOfc0865Ifh6uA//nOOcW+ScO9U5tz5+C/gUM9utrgY556Y457Z2znUJbdkYf2itIlpQnWD+08GPAOc6527KN23iUaC3mdVe5byuOqlt3+CwjhsB24b3k+/WNgHAOTfBOTcY3/G9ij/yg3NuvnPuJ865nvg98qtCp5raHL+XNiHU/GvA3/Gdd/1WZpe7lB+SQ2vJfZPJGZgMt92PH9fphO9ENwB2Cvf9FD9g2Ru/JfUo+Y8hzwW24uvHkJ8AhuUssw9+F31P/LHS1fBbuL1zpr8UX2Q74I/N1ne4b238YZYDwnwuImeXG79nMznP8zUeP5i4Ff64ekf81ssxdazjnfhCaYUvtv8C88J9GwO74o97twXGAGPDfUewYrxtd/wb3mp1tKUNfkv6rNCW7fF7hAU/BKI6KVgnO+G3/AeEPBC/5btHyDsDLs/zVbvXsj/+E1pt8P/sF+fU2c3h75/h97474Q/bXBWej/6hrYez4tDfscBb4e99wzQW1r2GnA8UJe0ZGNrQCT/m9V+9n5SlTnrh9+B+Vc/9R9W+XvXc/2f8xuzOYXmr4fc6R4b7x+KPlIDfs38oTNMF/0k8h//f744fP20fnsffEd7HwnNTu26b4Y/ArFdHWzbA7wjsGp7DDfDvL3kP+ZW9iMpQVGsAV+MPFSwEngMOCfe1xg/if4T/dEmhT+MMxx8TXoz/8MLAcPsQ/HH4Baz4JNO2+IHsj/Gfsvo7sG64b338ntViivg0Dv6N/9XwYk0G+uXcdz1wfp7Htg0FMAv4DH8o4K85bfnfOoaCmBbaNR2/5VbbSW2B35JdFNbpQVZ8iOJm/HjVYvzW9v552rMZfuvwM/w/9A9UJ2WrkxPD67wIf0jo1Jz7fkzOp0Trefzh+E9pfoZ/U/w78J1w3zms6KQ64N+sF4V6OpK4k3oYP1j+KX4DaYfwuF+E1+Wz8Dyflactt4XXYSH+kE431UnD6wQ4O7Rpce5Pzv1nAbfkeb4M/xH0Gfg9zXfC61P7oYexrOikeoZ1Xow/BHg8KzqpHqz4BN+CMN2m4XEXh/kuxneow/K050fhuVsUXpOLqOMTo7k/usBsIwu72Ls5fxxfpE5m9lfgLufchKZui2SXmf0TP071SlO3pVLUSYmISGbpArMiIpJZ6qRERCSzGtRJWZmvQizNk+pEiqE6kTo14FM2rfCf5Fgf/wmh5wmf9sjzGKefqv75QHWiH9WJfsrxU2xtNGRPahtglnNutnNuKclViKVZKnhmfB1UJy2P6kTKpiGdVC/iS4rMC7dFzGyYmT1j/ntvpOVRnUgxVCdSp9YNeKzVcZv72g3OjcZffBAz+9r90uypTqQYqhOpU0P2pOYRX/eqN/76VCK5VCdSDNWJ1KkhndTTwIZmtp6ZtcVfD2p8eZolzYjqRIqhOpE6rfThPufcMjOrvQpxK2CMc66YK31LC6I6kWKoTqQ+jXpZJB1DrnrTnHODCk/WMKqTqqc6kYKcc3WNQ36NrjghIiKZpU5KREQyS52UiIhkljopERHJrIaczCsiIo1kq622ivLEiROjvGDBgijvtddeUX799dcr07AK056UiIhkljopERHJLHVSIiKSWRqTEhHJgHbt2kX52muvjfI+++wT5U6dOuXNd911V5S/9a1vNbSJTUJ7UiIiklnqpEREJLPUSYmISGZpTCqPk046Kcp/+tOfmqgl0pz95je/ifLvf//7KJvF1+H84IMPorzrrrtG+aWXXipj66RSNt988yjfcMMNUR44cGCU0zoodHHwyZMnr3zjMkR7UiIiklnqpEREJLPUSYmISGa16DGp9u3bR3nUqFFR7tevX5Q1JiXF6N+/f5RPO+20KB955JFRbt06/jdMxxrSvNZaa0X5nnvuifJGG21UfGOl0fTs2TPKJ598cpTTMaiGOvroo6P81FNPRfmWW24p6/IqRXtSIiKSWeqkREQks9RJiYhIZrXoMan11lsvyj/72c+ivO222zZmc6RKpWNAv/rVr6J8zDHHlDS/9957L8qfffZZlNdff/28+corr4zyCSecUNLypTJGjhwZ5aOOOqqiy0uvBThu3Lgob7nlllF+/vnno3zjjTdWpmEl0p6UiIhkljopERHJLHVSIiKSWVbo+k9lXZhZ4y2sCI888kiU11577SgfdthhUZ4xY0bF25Rx05xzgyq9kKzVSSHPPvtslAt9b8+9994b5aeffjrK11xzTZTTsYT99tsv7/zffffdKPfp0yfv9BWgOgG22mqrKE+cODHKa6yxRknzW2WVeJ9i+fLlK9eweuZ3xx13RPmQ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)



### 代码修改

1. 更新后的keras中关键词show_accuracy不再存在，所以原代码中的`show_accuracy=True`要删除。

2. `model.evaluate`返回一个测试误差的标量值（如果模型没有其他评价指标），或一个标量的list（如果模型还有其他的评价指标）。原代码返回的是一个标量值Test Loss（暂时不知模型有没有其他评价指标是从何设置的），所以 `score[0]` 和 `score[1]` 报错，将其改为`print('Test score:', score)`。

   

## 6. 资料整理

### 深度学习模型优化

------

#### 过拟合问题

##### 模型大小

模型大小，即单位数。如果使用的模型远远大于手头上的问题，那么很有可能会学习与问题无关的特征或模式，因而过拟合训练数据。较大的模型无法很好地概括，而较小的模型将会欠拟合数据。 

可以评估缩小和增加模型大小对模型性能产生的影响，并且根据训练回合数绘制验证数据的损失函数来比较不同模型大小对模型性能产生的影响。

以下为一个例子：

```python
baseline_model = keras.models.Sequential([
        keras.layers.Flatten(input_shape = ( maxsize_w, maxsize_h , 1)),
                keras.layers.Dense(128, activation=tf.nn.sigmoid),
                keras.layers.Dense(16, activation=tf.nn.sigmoid),
        keras.layers.Dense(2, activation=tf.nn.softmax)
        ])
 
bigger_model2 = keras.models.Sequential([
                keras.layers.Flatten(input_shape = ( maxsize_w, maxsize_h , 1)),
                keras.layers.Dense(1024, activation=tf.nn.relu),
                keras.layers.Dense(512, activation=tf.nn.relu),
                keras.layers.Dense(64, activation=tf.nn.relu),
                keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(2, activation=tf.nn.softmax)
        ])
 
bigger_model1 = keras.models.Sequential([
        keras.layers.Flatten(input_shape = ( maxsize_w, maxsize_h , 1)),
                keras.layers.Dense(512, activation=tf.nn.relu),
                keras.layers.Dense(128, activation=tf.nn.relu),
                keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(2, activation=tf.nn.softmax)
        ])
 
smaller_model1 = keras.models.Sequential([
        keras.layers.Flatten(input_shape = ( maxsize_w, maxsize_h , 1)),
                keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(2, activation=tf.nn.softmax)
        ])
```

Smaller、Bigger 和 Baseline 模型的比较。

![img](https://www.ibm.com/developerworks/cn/cognitive/library/image-recognition-challenge-with-tensorflow-and-keras-pt2/model_size_comparision_plot2.png)

Bigger、Bigger2 和 Baseline 模型的比较：

![img](https://www.ibm.com/developerworks/cn/cognitive/library/image-recognition-challenge-with-tensorflow-and-keras-pt2/model_size_comparision_plot3.png)

##### 训练回合数

根据上面的模型比较图可知，随着训练回合数（epoch）的增加，验证数据的损失函数达到最小值，并且在进一步训练时再次增加，而训练数据的损失函数则进一步减小，这便是过拟合了。所以需要把控好回合数，防止过拟合。

##### L1和L2正则化

应用L2正则化的效果是向层添加一些随机噪音。下图显示了将此应用于模型的效果：

![img](https://www.ibm.com/developerworks/cn/cognitive/library/image-recognition-challenge-with-tensorflow-and-keras-pt2/L2_regularization.png)

##### 使用Dropout

keras库提供Dropout层，Dropout是一种防止神经网络过拟合的简单方法。添加Dropout层的结果是训练时间增加，但如果Dropout较高，就会欠拟合。

应用Dropout层后的模型：

```python
bigger_model1 = keras.models.Sequential([
                keras.layers.Flatten(input_shape = ( maxsize_w, maxsize_h , 1)),
                keras.layers.Dense(512, activation=tf.nn.relu),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(128, activation=tf.nn.relu),
                keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(2, activation=tf.nn.softmax)
        ])
 
bigger_model2 = keras.models.Sequential([
                keras.layers.Flatten(input_shape = ( maxsize_w, maxsize_h , 1)),
                keras.layers.Dense(1024, activation=tf.nn.relu),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(512, activation=tf.nn.relu),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(64, activation=tf.nn.relu),
                keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(2, activation=tf.nn.softmax)
        ])
```

下图显示了应用Dropout正则化的效果：

![img](https://www.ibm.com/developerworks/cn/cognitive/library/image-recognition-challenge-with-tensorflow-and-keras-pt2/dropout_regularization.png)

在一次运行期间，Bigger 模型根本没有收敛，即使在 250 个训练回合之后也仍然如此。这是应用 dropout 正则化的副作用之一。

![img](https://www.ibm.com/developerworks/cn/cognitive/library/image-recognition-challenge-with-tensorflow-and-keras-pt2/Model_No_converge.png)

#### 超参数

##### 贝叶斯优化

贝叶斯方法试图建立一个函数（更准确地说，是关于可能函数的概率分布），用于估计模型对于某个超参数选择的好坏程度。使用这种近似函数（在文献中称为代理函数）可以不必在设置、训练、评估的循环上花费太多时间，因为你可以优化代理函数的超参数。

例如，假设我们想要最小化此函数（将其视为模型损失函数的代理）：

![新知图谱, 如何优化深度学习模型](https://img.shangyexinzhi.com/xztest-image/article/ee6d1f1191ddeabf16a58ae7d18587be.png?x-oss-process=image/sharpen,200)

代理函数来自于高斯过程（注意：还有其他方法来模拟代理函数，但此处将使用高斯过程）。此处不做任何数学上的重要推导，但是所有关于贝叶斯和高斯的讨论归结为：    

![新知图谱, 如何优化深度学习模型](https://img.shangyexinzhi.com/xztest-image/article/6e2d69052a79af0c3adc0dc5aabd9f0d.png?x-oss-process=image/sharpen,200)

左侧告诉你涉及概率分布（假设存在P）。在括号内看，我们可以看到它是P的概率分布，这是一个任意的函数。为什么？请记住，我们正在定义所有可能函数的概率分布，而不仅仅是特定函数。本质上，左侧表示将超参数映射到模型的度量的真实函数（如验证准确性，对数似然，测试错误率等）的概率为Fn(X)，给定一些样本数据Xn等于右侧的式子。

现在我们有了优化函数，就开始进行优化吧。

以下是在开始优化过程之前高斯过程的样子（在利用两个数据点迭代之前的高斯过程）：

![img](https://img.shangyexinzhi.com/xztest-image/article/c8ea8d26f4232ede0dbf5a683ca13fb0.jpeg?x-oss-process=image/sharpen,200)

经过几次迭代后，高斯过程在近似目标函数方面变得更好（在利用两个数据点迭代三次之后的高斯过程）：

![img](https://img.shangyexinzhi.com/xztest-image/article/3f88b152f4f2cb98c790af93245438b1.jpeg?x-oss-process=image/sharpen,200)

最终结果应如下所示（在利用两个数据点迭代七次之后的高斯过程）：

![img](https://img.shangyexinzhi.com/xztest-image/article/4778c9e1bfa9d87fe46dc45c5936b4ac.jpeg?x-oss-process=image/sharpen,200)

优点：贝叶斯优化比网格搜索和随机搜索提供更好的结果。

缺点：并行化并不容易。    

#### 优化算法

##### SGD

此处的SGD指mini-batch gradient descent。

SGD就是每一次迭代计算mini-batch的梯度，然后对参数进行更新，是最常见的优化方法了。即：

![[公式]](https://www.zhihu.com/equation?tex=g_t%3D%5Cnabla_%7B%5Ctheta_%7Bt-1%7D%7D%7Bf%28%5Ctheta_%7Bt-1%7D%29%7D)

![[公式]](https://www.zhihu.com/equation?tex=%5CDelta%7B%5Ctheta_t%7D%3D-%5Ceta%2Ag_t)

其中，![[公式]](https://www.zhihu.com/equation?tex=%5Ceta)是学习率，![[公式]](https://www.zhihu.com/equation?tex=g_t)是梯度 SGD完全依赖于当前batch的梯度，所以![[公式]](https://www.zhihu.com/equation?tex=%5Ceta)可理解为允许当前batch的梯度多大程度影响参数更新。

缺点：

- 选择合适的learning rate比较困难 - 对所有的参数更新使用同样的learning rate。对于稀疏数据或者特征，有时我们可能想更新快一些对于不经常出现的特征，对于常出现的特征更新慢一些，这时候SGD就不太能满足要求了。

- SGD容易收敛到局部最优，并且在某些情况下可能被困在鞍点。

  

##### Momentum

momentum是模拟物理里动量的概念，积累之前的动量来替代真正的梯度。公式如下：

![[公式]](https://www.zhihu.com/equation?tex=m_t%3D%5Cmu%2Am_%7Bt-1%7D%2Bg_t)

![[公式]](https://www.zhihu.com/equation?tex=%5CDelta%7B%5Ctheta_t%7D%3D-%5Ceta%2Am_t)

其中，![[公式]](https://www.zhihu.com/equation?tex=%5Cmu)是动量因子。

特点：

- 下降初期时，使用上一次参数更新，下降方向一致，乘上较大的![[公式]](https://www.zhihu.com/equation?tex=%5Cmu)能够进行很好的加速 。

- 下降中后期时，在局部最小值来回震荡的时候，![[公式]](https://www.zhihu.com/equation?tex=gradient%5Cto0)，![[公式]](https://www.zhihu.com/equation?tex=%5Cmu)使得更新幅度增大，跳出陷阱 。

- 在梯度改变方向的时候，![[公式]](https://www.zhihu.com/equation?tex=%5Cmu)能够减少更新，总而言之，momentum项能够在相关方向加速SGD，抑制振荡，从而加快收敛。

  

##### Nesterov

nesterov项在梯度更新时做一个校正，避免前进太快，同时提高灵敏度。 将上一节中的公式展开可得：

![[公式]](https://www.zhihu.com/equation?tex=%5CDelta%7B%5Ctheta_t%7D%3D-%5Ceta%2A%5Cmu%2Am_%7Bt-1%7D-%5Ceta%2Ag_t)

可以看出，![[公式]](https://www.zhihu.com/equation?tex=m_%7Bt-1%7D%0A)并没有直接改变当前梯度![[公式]](https://www.zhihu.com/equation?tex=g_t)，所以Nesterov的改进就是让之前的动量直接影响当前的动量。即： 

![[公式]](https://www.zhihu.com/equation?tex=g_t%3D%5Cnabla_%7B%5Ctheta_%7Bt-1%7D%7D%7Bf%28%5Ctheta_%7Bt-1%7D-%5Ceta%2A%5Cmu%2Am_%7Bt-1%7D%29%7D)

![[公式]](https://www.zhihu.com/equation?tex=m_t%3D%5Cmu%2Am_%7Bt-1%7D%2Bg_t)

![[公式]](https://www.zhihu.com/equation?tex=%5CDelta%7B%5Ctheta_t%7D%3D-%5Ceta%2Am_t)

所以，加上nesterov项后，梯度在大的跳跃后，进行计算对当前梯度进行校正。如下图： 

![preview](https://pic4.zhimg.com/fecd469405501ad82788f068985b25cb_r.jpg)

momentum首先计算一个梯度（短的蓝色向量），然后在加速更新梯度的方向进行一个大的跳跃（长的蓝色向量），nesterov项首先在之前加速的梯度方向进行一个大的跳跃（棕色向量），计算梯度然后进行校正（绿色梯向量）。

 其实，momentum项和nesterov项都是为了使梯度更新更加灵活，对不同情况有针对性。但是，人工设置一些学习率总还是有些生硬，接下来介绍几种自适应学习率的方法。



##### Adagrad 

Adagrad其实是对学习率进行了一个约束。即： 

![[公式]](https://www.zhihu.com/equation?tex=n_t%3Dn_%7Bt-1%7D%2Bg_t%5E2)

![[公式]](https://www.zhihu.com/equation?tex=%5CDelta%7B%5Ctheta_t%7D%3D-%5Cfrac%7B%5Ceta%7D%7B%5Csqrt%7Bn_t%2B%5Cepsilon%7D%7D%2Ag_t)

此处，对![[公式]](https://www.zhihu.com/equation?tex=g_t)从1到![[公式]](https://www.zhihu.com/equation?tex=t)进行一个递推形成一个约束项regularizer，![[公式]](https://www.zhihu.com/equation?tex=-%5Cfrac%7B1%7D%7B%5Csqrt%7B%5Csum_%7Br%3D1%7D%5Et%28g_r%29%5E2%2B%5Cepsilon%7D%7D)，![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon)用来保证分母非0 。

特点：

- 前期![[公式]](https://www.zhihu.com/equation?tex=g_t)较小的时候， regularizer较大，能够放大梯度。
- 后期![[公式]](https://www.zhihu.com/equation?tex=g_t)较大的时候，regularizer较小，能够约束梯度。
- 适合处理稀疏梯度。

缺点：

- 由公式可以看出，仍依赖于人工设置一个全局学习率。

- ![[公式]](https://www.zhihu.com/equation?tex=%5Ceta)设置过大的话，会使regularizer过于敏感，对梯度的调节太大。

- 中后期，分母上梯度平方的累加将会越来越大，使![[公式]](https://www.zhihu.com/equation?tex=gradient%5Cto0)，使得训练提前结束。

  

##### Adadelta 

Adadelta是对Adagrad的扩展，最初方案依然是对学习率进行自适应约束，但是进行了计算上的简化。Adagrad会累加之前所有的梯度平方，而Adadelta只累加固定大小的项，并且也不直接存储这些项，仅仅是近似计算对应的平均值。即：

![[公式]](https://www.zhihu.com/equation?tex=n_t%3D%5Cnu%2An_%7Bt-1%7D%2B%281-%5Cnu%29%2Ag_t%5E2)

![[公式]](https://www.zhihu.com/equation?tex=%5CDelta%7B%5Ctheta_t%7D+%3D+-%5Cfrac%7B%5Ceta%7D%7B%5Csqrt%7Bn_t%2B%5Cepsilon%7D%7D%2Ag_t)

在此处Adadelta其实还是依赖于全局学习率的，但是作者做了一定处理，经过近似牛顿迭代法之后：

![[公式]](https://www.zhihu.com/equation?tex=E%7Cg%5E2%7C_t%3D%5Crho%2AE%7Cg%5E2%7C_%7Bt-1%7D%2B%281-%5Crho%29%2Ag_t%5E2)

![[公式]](https://www.zhihu.com/equation?tex=%5CDelta%7Bx_t%7D%3D-%5Cfrac%7B%5Csqrt%7B%5Csum_%7Br%3D1%7D%5E%7Bt-1%7D%5CDelta%7Bx_r%7D%7D%7D%7B%5Csqrt%7BE%7Cg%5E2%7C_t%2B%5Cepsilon%7D%7D)

其中，![[公式]](https://www.zhihu.com/equation?tex=E)代表求期望。

此时，可以看出Adadelta已经不用依赖于全局学习率了。

特点：

- 训练初中期，加速效果不错，很快。

- 训练后期，反复在局部最小值附近抖动。

  

##### RMSprop 

RMSprop可以算作Adadelta的一个特例：

当![[公式]](https://www.zhihu.com/equation?tex=%5Crho%3D0.5)时，![[公式]](https://www.zhihu.com/equation?tex=E%7Cg%5E2%7C_t%3D%5Crho%2AE%7Cg%5E2%7C_%7Bt-1%7D%2B%281-%5Crho%29%2Ag_t%5E2)就变为了求梯度平方和的平均数。

如果再求根的话，就变成了RMS（均方根）：

![[公式]](https://www.zhihu.com/equation?tex=RMS%7Cg%7C_t%3D%5Csqrt%7BE%7Cg%5E2%7C_t%2B%5Cepsilon%7D)

此时，这个RMS就可以作为学习率![[公式]](https://www.zhihu.com/equation?tex=%5Ceta)的一个约束： 

![[公式]](https://www.zhihu.com/equation?tex=%5CDelta%7Bx_t%7D%3D-%5Cfrac%7B%5Ceta%7D%7BRMS%7Cg%7C_t%7D%2Ag_t)

特点：

- 其实RMSprop依然依赖于全局学习率。

- RMSprop算是Adagrad的一种发展，和Adadelta的变体，效果趋于二者之间。

- 适合处理非平稳目标 - 对于RNN效果很好。

  

##### Adam 

Adam(Adaptive Moment Estimation)本质上是带有动量项的RMSprop，它利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。Adam的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳。公式如下：

![[公式]](https://www.zhihu.com/equation?tex=m_t%3D%5Cmu%2Am_%7Bt-1%7D%2B%281-%5Cmu%29%2Ag_t)

![[公式]](https://www.zhihu.com/equation?tex=n_t%3D%5Cnu%2An_%7Bt-1%7D%2B%281-%5Cnu%29%2Ag_t%5E2)

![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7Bm_t%7D%3D%5Cfrac%7Bm_t%7D%7B1-%5Cmu%5Et%7D)

![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7Bn_t%7D%3D%5Cfrac%7Bn_t%7D%7B1-%5Cnu%5Et%7D)

![[公式]](https://www.zhihu.com/equation?tex=%5CDelta%7B%5Ctheta_t%7D%3D-%5Cfrac%7B%5Chat%7Bm_t%7D%7D%7B%5Csqrt%7B%5Chat%7Bn_t%7D%7D%2B%5Cepsilon%7D%2A%5Ceta)

其中，![[公式]](https://www.zhihu.com/equation?tex=m_t)，![[公式]](https://www.zhihu.com/equation?tex=n_t)分别是对梯度的一阶矩估计和二阶矩估计，可以看作对期望![[公式]](https://www.zhihu.com/equation?tex=E%7Cg_t%7C)，![[公式]](https://www.zhihu.com/equation?tex=E%7Cg_t%5E2%7C)的估计；![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7Bm_t%7D)，![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7Bn_t%7D)是对![[公式]](https://www.zhihu.com/equation?tex=m_t)，![[公式]](https://www.zhihu.com/equation?tex=n_t)的校正，这样可以近似为对期望的无偏估计。 可以看出，直接对梯度的矩估计对内存没有额外的要求，而且可以根据梯度进行动态调整，而![[公式]](https://www.zhihu.com/equation?tex=-%5Cfrac%7B%5Chat%7Bm_t%7D%7D%7B%5Csqrt%7B%5Chat%7Bn_t%7D%7D%2B%5Cepsilon%7D)对学习率形成一个动态约束，而且有明确的范围。

特点：

- 结合了Adagrad善于处理稀疏梯度和RMSprop善于处理非平稳目标的优点。

- 对内存需求较小。

- 为不同的参数计算不同的自适应学习率。

- 也适用于大多非凸优化 - 适用于大数据集和高维空间。

  

##### Adamax 

Adamax是Adam的一种变体，此方法对学习率的上限提供了一个更简单的范围。公式上的变化如下：

![[公式]](https://www.zhihu.com/equation?tex=n_t%3Dmax%28%5Cnu%2An_%7Bt-1%7D%2C%7Cg_t%7C%29)

![[公式]](https://www.zhihu.com/equation?tex=%5CDelta%7Bx%7D%3D-%5Cfrac%7B%5Chat%7Bm_t%7D%7D%7Bn_t%2B%5Cepsilon%7D%2A%5Ceta)

可以看出，Adamax学习率的边界范围更简单。



##### Nadam 

Nadam类似于带有Nesterov动量项的Adam。公式如下：

![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7Bg_t%7D%3D%5Cfrac%7Bg_t%7D%7B1-%5CPi_%7Bi%3D1%7D%5Et%5Cmu_i%7D)

![[公式]](https://www.zhihu.com/equation?tex=m_t%3D%5Cmu_t%2Am_%7Bt-1%7D%2B%281-%5Cmu_t%29%2Ag_t)

![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7Bm_t%7D%3D%5Cfrac%7Bm_t%7D%7B1-%5CPi_%7Bi%3D1%7D%5E%7Bt%2B1%7D%5Cmu_i%7D)

![[公式]](https://www.zhihu.com/equation?tex=n_t%3D%5Cnu%2An_%7Bt-1%7D%2B%281-%5Cnu%29%2Ag_t%5E2)

![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7Bn_t%7D%3D%5Cfrac%7Bn_t%7D%7B1-%5Cnu%5Et%7D)

![[公式]](https://www.zhihu.com/equation?tex=%5Cbar%7Bm_t%7D%3D%281-%5Cmu_t%29%2A%5Chat%7Bg_t%7D%2B%5Cmu_%7Bt%2B1%7D%2A%5Chat%7Bm_t%7D)

![[公式]](https://www.zhihu.com/equation?tex=%5CDelta%7B%5Ctheta_t%7D%3D-%5Ceta%2A%5Cfrac%7B%5Cbar%7Bm_t%7D%7D%7B%5Csqrt%7B%5Chat%7Bn_t%7D%7D%2B%5Cepsilon%7D)

可以看出，Nadam对学习率有了更强的约束，同时对梯度的更新也有更直接的影响。一般而言，在想使用带动量的RMSprop，或者Adam的地方，大多可以使用Nadam取得更好的效果。

##### 总结

- 对于稀疏数据，尽量使用学习率可自适应的优化方法，不用手动调节，而且最好采用默认值。
- SGD通常训练时间更长，但是在好的初始化和学习率调度方案的情况下，结果更可靠。
- 如果在意更快的收敛，并且需要训练较深较复杂的网络时，推荐使用学习率自适应的优化方法。
- Adadelta，RMSprop，Adam是比较相近的算法，在相似的情况下表现差不多。
- 在想使用带动量的RMSprop，或者Adam的地方，大多可以使用Nadam取得更好的效果。

- 对于稀疏数据，尽量使用学习率可自适应的优化方法，不用手动调节，而且最好采用默认值  
- SGD通常训练时间更长，但是在好的初始化和学习率调度方案的情况下，结果更可靠  
- 如果在意更快的收敛，并且需要训练较深较复杂的网络时，推荐使用学习率自适应的优化方法。  
- Adadelta，RMSprop，Adam是比较相近的算法，在相似的情况下表现差不多。 

损失平面等高线 ：

![img](https://pic1.zhimg.com/80/5d5166a3d3712e7c03af74b1ccacbeac_hd.jpg)

在鞍点处的比较：

# ![img](https://pic1.zhimg.com/80/4a3b4a39ab8e5c556359147b882b4788_hd.jpg)