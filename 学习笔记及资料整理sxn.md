# 学习笔记 190804



## NumPy

### 应用背景

NumPy 是一个运行速度非常快的数学库，主要用于数组计算，包含：

> * 一个强大的N维数组对象 ndarray
> * 广播功能函数
> * 整合 C/C++/Fortran 代码的工具
> * 线性代数、傅里叶变换、随机数生成等功能

NumPy 通常与 SciPy（Scientific Python）和 Matplotlib（绘图库）一起使用， 这种组合广泛用于替代 MatLab，是一个强大的科学计算环境，有助于我们通过 Python 学习数据科学或者机器学习。

### NumPy Ndarray对象

NumPy 最重要的一个特点是其 N 维数组对象 ndarray，它是一系列同类型数据的集合，以 0 下标为开始进行集合中元素的索引。 

 ndarray 对象是用于存放同类型元素的多维数组。 

 ndarray 中的每个元素在内存中都有相同存储大小的区域。 

  ndarray 内部由以下内容组成： 

- 一个指向数据（内存或内存映射文件中的一块数据）的指针。

- 数据类型或 dtype，描述在数组中的固定大小值的格子。

- 一个表示数组形状（shape）的元组，表示各维度大小的元组。

- 一个跨度元组（stride），其中的整数指的是为了前进到当前维度下一个元素需要"跨过"的字节数。

  ![img](https://www.runoob.com/wp-content/uploads/2018/10/ndarray.png)

  >参数说明
  >
  >| object | 数组或嵌套的数列                                          |
  >| ------ | --------------------------------------------------------- |
  >| dtype  | 数组元素的数据类型，可选                                  |
  >| copy   | 对象是否需要复制，可选                                    |
  >| order  | 创建数组的样式，C为行方向，F为列方向，A为任意方向（默认） |
  >| subok  | 默认返回一个与基类类型一致的数组                          |
  >| ndmin  | 指定生成数组的最小维度                                    |

### NumPy 数组

NumPy 数组的维数称为秩（rank），一维数组的秩为 1，二维数组的秩为 2，以此类推。

在   NumPy中，每一个线性的数组称为是一个轴（axis），也就是维度（dimensions）。比如说，二维数组相当于是两个一维数组，其中第一个一维数组中每个元素又是一个一维数组。所以一维数组就是  NumPy 中的轴（axis），第一个轴相当于是底层数组，第二个轴是底层数组里的数组。而轴的数量——秩，就是数组的维数。

很多时候可以声明 axis。axis=0，表示沿着第 0 轴进行操作，即对每一列进行操作；axis=1，表示沿着第1轴进行操作，即对每一行进行操作。

NumPy 的数组中比较重要 ndarray 对象属性有：

| 属性             | 说明                                                         |
| ---------------- | ------------------------------------------------------------ |
| ndarray.ndim     | 秩，即轴的数量或维度的数量                                   |
| ndarray.shape    | 数组的维度，对于矩阵，n 行 m 列                              |
| ndarray.size     | 数组元素的总个数，相当于 .shape 中 n*m 的值                  |
| ndarray.dtype    | ndarray 对象的元素类型                                       |
| ndarray.itemsize | ndarray 对象中每个元素的大小，以字节为单位                   |
| ndarray.flags    | ndarray 对象的内存信息                                       |
| ndarray.real     | ndarray元素的实部                                            |
| ndarray.imag     | ndarray 元素的虚部                                           |
| ndarray.data     | 包含实际数组元素的缓冲区，由于一般通过数组的索引获取元素，所以通常不需要使用这个属性。 |

### NumPy 切片和索引

ndarray对象的内容可以通过索引或切片来访问和修改，与 Python 中 list 的切片操作一样。

ndarray 数组可以基于 0 - n 的下标进行索引，切片对象可以通过内置的 slice 函数，并设置 start, stop 及 step 参数进行，从原数组中切割出一个新数组。

实例代码：

```
import numpy as np
 
a = np.arange(10)
s = slice(2,7,2)   # 从索引 2 开始到索引 7 停止，间隔为2
print (a[s])
```

输出结果为：

```
[2  4  6]
```

### NumPy 广播（Broadcast）

广播(Broadcast)是 numpy 对不同形状(shape)的数组进行数值计算的方式， 对数组的算术运算通常在相应的元素上进行。

如果两个数组 a 和 b 形状相同，即满足 **a.shape == b.shape**，那么 a*b 的结果就是 a 与 b 数组对应位相乘。这要求维数相同，且各维度的长度相同。

实例代码：

```
import numpy as np 
a = np.array([1,2,3,4]) 
b = np.array([10,20,30,40]) 
c = a * b 
print (c)
```

输出：

```
[ 10  40  90 160]
```

> 广播的规则:
>
> * 让所有输入数组都向其中形状最长的数组看齐，形状中不足的部分都通过在前面加 1 补齐。
>
> * 输出数组的形状是输入数组形状的各个维度上的最大值。
> * 如果输入数组的某个维度和输出数组的对应维度的长度相同或者其长度为 1 时，这个数组能够用来计算，否则出错。
> * 当输入数组的某个维度的长度为 1 时，沿着此维度运算时都用此维度上的第一组值。

### NumPy 数组操作

**修改数组形状 **

| 函数      | 描述                                               |
| --------- | -------------------------------------------------- |
| `reshape` | 不改变数据的条件下修改形状                         |
| `flat`    | 数组元素迭代器                                     |
| `flatten` | 返回一份数组拷贝，对拷贝所做的修改不会影响原始数组 |
| `ravel`   | 返回展开数组                                       |

其中reshape:

numpy.reshape() 函数可以在不改变数据的条件下修改形状，格式如下： numpy.reshape(arr, newshape, order='C')   

- `arr`：要修改形状的数组
- `newshape`：整数或者整数数组，新的形状应当兼容原有形状
- order：'C' -- 按行，'F' -- 按列，'A' -- 原顺序，'k' -- 元素在内存中的出现顺序。

**翻转数组**

| 函数        | 描述                       |
| ----------- | -------------------------- |
| `transpose` | 对换数组的维度             |
| `ndarray.T` | 和 `self.transpose()` 相同 |
| `rollaxis`  | 向后滚动指定的轴           |
| `swapaxes`  | 对换数组的两个轴           |

**修改数组维度**

| 维度           | 描述                       |
| -------------- | -------------------------- |
| `broadcast`    | 产生模仿广播的对象         |
| `broadcast_to` | 将数组广播到新形状         |
| `expand_dims`  | 扩展数组的形状             |
| `squeeze`      | 从数组的形状中删除一维条目 |

**连接数组**

| 函数          | 描述                           |
| ------------- | ------------------------------ |
| `concatenate` | 连接沿现有轴的数组序列         |
| `stack`       | 沿着新的轴加入一系列数组。     |
| `hstack`      | 水平堆叠序列中的数组（列方向） |
| `vstack`      | 竖直堆叠序列中的数组（行方向） |

**分割数组**

| 函数     | 数组及操作                             |
| -------- | -------------------------------------- |
| `split`  | 将一个数组分割为多个子数组             |
| `hsplit` | 将一个数组水平分割为多个子数组（按列） |
| `vsplit` | 将一个数组垂直分割为多个子数组（按行） |

**数组元素的添加和删除**

| 函数     | 元素及描述                               |
| -------- | ---------------------------------------- |
| `resize` | 返回指定形状的新数组                     |
| `append` | 将值添加到数组末尾                       |
| `insert` | 沿指定轴将值插入到指定下标之前           |
| `delete` | 删掉某个轴的子数组，并返回删除后的新数组 |
| `unique` | 查找数组内的唯一元素                     |

具体用法参考NumPy文档

### NumPy Matplotlib

>Matplotlib 是 Python 的绘图库。 它可与 NumPy 一起使用，提供了一种有效的 MatLab 开源替代方案。 它也可以和图形工具包一起使用，如 PyQt 和 wxPython。

### NumPy 文档地址

> NumPy 官网 http://www.numpy.org/
>
> Matplotlib 源代码：https://github.com/matplotlib/matplotlib

## Pandas 学习笔记

**Pandas数据帧（DataFrame）**

数据帧(DataFrame)是二维数据结构，即数据以行和列的表格方式排列。

数据帧(DataFrame)的功能特点：

- 潜在的列是不同的类型
- 大小可变
- 标记轴(行和列)
- 可以对行和列执行算术运算

*pandas*中的`DataFrame`可以使用以下构造函数创建 -

```python
pandas.DataFrame( data, index, columns, dtype, copy)
```

> 构造函数的参数如下 -  

| 编号 | 参数      | 描述                                                         |
| ---- | --------- | ------------------------------------------------------------ |
| 1    | `data`    | 数据采取各种形式，如:`ndarray`，`series`，`map`，`lists`，`dict`，`constant`和另一个`DataFrame`。 |
| 2    | `index`   | 对于行标签，要用于结果帧的索引是可选缺省值`np.arrange(n)`，如果没有传递索引值。 |
| 3    | `columns` | 对于列标签，可选的默认语法是 - `np.arange(n)`。 这只有在没有索引传递的情况下才是这样。 |
| 4    | `dtype`   | 每列的数据类型。                                             |
| 5    | `copy`    | 如果默认值为`False`，则此命令(或任何它)用于复制数据。        |

**Pandas 分组**

任何分组(*groupby*)操作都涉及原始对象的以下操作之一。它们是 - 

- 分割对象
- 应用一个函数
- 结合的结果

在许多情况下，我们将数据分成多个集合，并在每个子集上应用一些函数。在应用函数中，可以执行以下操作 -

- *聚合* - 计算汇总统计
- *转换* - 执行一些特定于组的操作
- *过滤* - 在某些情况下丢弃数据

>Pandas对象可以分成任何对象。有多种方式来拆分对象，如 -
>
>- *obj.groupby(‘key’)*
>- *obj.groupby([‘key1’,’key2’])*
>- *obj.groupby(key,axis=1)*

**Pandas可视化** 					 					 					 				

Series和DataFrame上的这个功能只是使用`matplotlib`库的`plot()`方法的简单包装实现。参考以下示例代码 - 

```python
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(10,4),index=pd.date_range('2018/12/18',
   periods=10), columns=list('ABCD'))

df.plot()
Python
```

执行上面示例代码，得到以下结果 - 

![img](http://www.yiibai.com/uploads/images/201711/0511/385181122_97686.png)

如果索引由日期组成，则调用`gct().autofmt_xdate()`来格式化`x`轴，如上图所示。

我们可以使用`x`和`y`关键字绘制一列与另一列。

> 绘图方法允许除默认线图之外的少数绘图样式。 这些方法可以作为`plot()`的`kind`关键字参数提供。这些包括 -
>
> * `bar`或`barh`为条形
> * `hist`为直方图
> * `boxplot`为盒型图
> * `area`为“面积”
> * scatter为散点图

## 实例代码学习

> 运用mnist数据集进行Deep Learning的学习

### mnist_cnn

```python
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
#mnist数据集相关常数
batch_size = 128 #一个训练batch中的训练数据的个数，数字越小是，训练过程                  #越接近随机梯度下降；数字越大的时候约接近梯度下降
num_classes = 10 
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
#将图像像素转化为0到1之间的实数
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# 把标准答案转化为需要的格式（one-hot编码）
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#使用Keras API定义模型
model = Sequential()
#深度为32，过滤器大小为3*3的卷积层
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
#深度为，64过滤器大小为3*3的卷积层
model.add(Conv2D(64, (3, 3), activation='relu'))
#最大池化层
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
#全连接层
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
#全连接层，得到最后输出
model.add(Dense(num_classes, activation='softmax'))

#定义损失函数、优化函数和测评方法
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

#在测试数据上计算准确率
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

最终结果：

```
x_train shape: (60000, 28, 28, 1)
60000 train samples
10000 test samples
Train on 60000 samples, validate on 10000 samples
Epoch 1/12
60000/60000 [==============================] - 4s 74us/step - loss: 0.2602 - acc: 0.9191 - val_loss: 0.0603 - val_acc: 0.9809
Epoch 2/12
60000/60000 [==============================] - 4s 68us/step - loss: 0.0878 - acc: 0.9742 - val_loss: 0.0435 - val_acc: 0.9860
Epoch 3/12
60000/60000 [==============================] - 4s 69us/step - loss: 0.0658 - acc: 0.9806 - val_loss: 0.0340 - val_acc: 0.9878
Epoch 4/12
60000/60000 [==============================] - 4s 68us/step - loss: 0.0524 - acc: 0.9843 - val_loss: 0.0329 - val_acc: 0.9890
Epoch 5/12
60000/60000 [==============================] - 4s 69us/step - loss: 0.0469 - acc: 0.9862 - val_loss: 0.0289 - val_acc: 0.9899
Epoch 6/12
60000/60000 [==============================] - 4s 69us/step - loss: 0.0409 - acc: 0.9872 - val_loss: 0.0272 - val_acc: 0.9911
Epoch 7/12
60000/60000 [==============================] - 4s 69us/step - loss: 0.0389 - acc: 0.9879 - val_loss: 0.0296 - val_acc: 0.9903
Epoch 8/12
60000/60000 [==============================] - 4s 68us/step - loss: 0.0330 - acc: 0.9900 - val_loss: 0.0254 - val_acc: 0.9919
Epoch 9/12
60000/60000 [==============================] - 4s 69us/step - loss: 0.0312 - acc: 0.9908 - val_loss: 0.0263 - val_acc: 0.9905
Epoch 10/12
60000/60000 [==============================] - 4s 68us/step - loss: 0.0288 - acc: 0.9912 - val_loss: 0.0252 - val_acc: 0.9922
Epoch 11/12
60000/60000 [==============================] - 4s 70us/step - loss: 0.0266 - acc: 0.9916 - val_loss: 0.0286 - val_acc: 0.9915
Epoch 12/12
60000/60000 [==============================] - 4s 69us/step - loss: 0.0251 - acc: 0.9919 - val_loss: 0.0300 - val_acc: 0.9915
Test loss: 0.03001883054177006
Test accuracy: 0.9915
```



### keras-mnist-tutorial

在此代码中我主要学习了对测试集一些数据的显示等。（因为此代码大部分与上一程序相同）

>  在跑程序的过程中发现
>
>  ```
>  model.fit(X_train, Y_train,
>         batch_size=128, nb_epoch=4,
>         show_accuracy=True, verbose=1,
>         validation_data=(X_test, Y_test))
>  ```
>
>  这里参数报错了
>
>  > TypeError: Unrecognized keyword arguments: {'show_accuracy': True}
>
>  ，然后我参照mnist_cnn.py修改，把show_accuracy=True删去，还有把nb_epoch改为epochs，不知道是不是版本的原因，可能最新的版本没有这个参数or改进了？
>
>  输出结果为：
>
>  ```
>  Train on 60000 samples, validate on 10000 samples
>  Epoch 1/4
>  60000/60000 [==============================] - 2s 26us/step - loss: 0.0144 - val_loss: 0.0581
>  Epoch 2/4
>  60000/60000 [==============================] - 1s 22us/step - loss: 0.0115 - val_loss: 0.0563
>  Epoch 3/4
>  60000/60000 [==============================] - 1s 22us/step - loss: 0.0115 - val_loss: 0.0553
>  Epoch 4/4
>  60000/60000 [==============================] - 1s 21us/step - loss: 0.0093 - val_loss: 0.0549
>  ```
>
>  ```
>  <keras.callbacks.History at 0x7fa938b21748>
>  ```
>
>  以及最后，关于score的输出：
>
>  ```
>  score = model.evaluate(X_test, Y_test,
>                      show_accuracy=True, verbose=0)
>  print('Test score:', score[0])
>  print('Test accuracy:', score[1])
>  ```
>
>  显示了
>
>  > IndexError: invalid index to scalar variable.
>
>  于是我把score[]改为score，
>
>  ```
>  score = model.evaluate(X_test,Y_test)
>  print('Test accuracy:', score)
>  ```
>
>  最后成功输出accuracy。
>
>  结果：
>
>  ```
>  10000/10000 [==============================] - 0s 33us/step
>  Test accuracy: 0.06877802454089396
>  ```
>
>  

## 深度学习优化方法

### 前言

此处讨论的优化问题是，给定目标函数
$$
f(x)
$$
，我们需要找到一组参数x，使得目标函数的值最小。  

深度学习的优化方法主要有一下几种：SGD, Neserove Momentum,  Adagrad, Adadelta, RMSprop, Adam。  

### SGD

方法：SGD指的是stochastic gradient descent，即随机梯度下降。  对于训练数据集，我们首先将其分成n个batch，每个batch包含m个样本。我们每次更新都利用batch的数据，而非整个训练集。即：

![img](https://images2017.cnblogs.com/blog/1218582/201801/1218582-20180116222037021-1729097658.jpg)
$$
x_{t+1}=x_t+\Delta x_t
$$

$$
\Delta x_t=-\eta g_t
$$

>  其中，η为学习率，gt为x在t时刻的梯度。 

* 优点：

  当训练数据太多时，利用整个数据集更新往往时间上不显示。batch的方法可以减少机器的压力，并且可以更快地收敛。

  当训练集有很多冗余时（类似的样本出现多次），batch方法收敛更快。以一个极端情况为例，若训练集前一半和后一半梯度相同。那么如果前一半作为一个batch，后一半作为另一个batch，那么在一次遍历训练集时，batch的方法向最优解前进两个step，而整体的方法只前进一个step。

* 缺点：

  其更新方向完全依赖于当前的batch，因而其更新十分不稳定。

* 改进方法：

  引入momentum优化方法。

### Momentum

方法：momentum即动量，它模拟的是物体运动时的惯性，即更新的时候在一定程度上保留之前更新的方向，同时利用当前batch的梯度微调最终的更新方向。这样一来，可以在一定程度上增加稳定性，从而学习地更快，并且还有一定摆脱局部最优的能力：

![img](https://images2017.cnblogs.com/blog/1218582/201801/1218582-20180116222155974-1758841902.jpg)
$$
\Delta x_t=\rho \Delta x_{t-1} -\eta g_t
$$

> 其中，ρ 即momentum，表示要在多大程度上保留原来的更新方向，这个值在0-1之间，在训练开始时，由于梯度可能会很大，所以初始值一般选为0.5；当梯度不那么大时，改为0.9。η 是学习率，即当前batch的梯度多大程度上影响最终更新方向，跟普通的SGD含义相同。ρ 与 η 之和不一定为1。

* 优点：

  在一定程度上增加稳定性，从而学习速度相对于SGD更快，而且还有一定摆脱局部最优的能力。

* 改进方法：引入Nesterov Momentum

### Nesterov Momentum

![Nesterov Momentum](http://img.blog.csdn.net/20150906103038485)

这是对传统momentum方法的一项改进，由Ilya Sutskever(2012 unpublished)在Nesterov工作的启发下提出的。首先，按照原来的更新方向更新一步（棕色线），然后在该位置计算梯度值（红色线），然后用这个梯度值修正最终的更新方向（绿色线）。上图中描述了两步的更新示意图，其中蓝色线是标准momentum更新路径。公式描述为：

![img](https://images2017.cnblogs.com/blog/1218582/201801/1218582-20180116222245506-303041880.jpg)
$$
\Delta x_t=\rho \Delta x_{t-1}-\eta \Delta f(x_t +\rho \Delta x_{t-1})
$$

### Adagrad

方法：对学习率进行一个约束，即

![img](https://images2017.cnblogs.com/blog/1218582/201801/1218582-20180116222331021-786003572.jpg)
$$
n_t=n_{t-1}+g_t^2
$$

$$
\Delta \theta_t=-\frac{\eta} {\sqrt{n_t+\epsilon}}*g_t
$$

此处，对gt从1到t进行一个递推形成一个约束项regularizer

![-\frac{1}{\sqrt{\sum_{r=1}^t(g_r)^2+\epsilon}}](https://zhihu.com/equation?tex=-%5Cfrac%7B1%7D%7B%5Csqrt%7B%5Csum_%7Br%3D1%7D%5Et%28g_r%29%5E2%2B%5Cepsilon%7D%7D)
$$
-\frac{1}{\sqrt{\sum_{r=1}^t(g_t)^2+\epsilon}}
$$


>  \epsilon 用来表示分母非0。

* 优点：

  前期gt较小的时候， regularizer较大，能够放大梯度

  后期gt较大的时候，regularizer较小，能够约束梯度 

  适合处理稀疏梯度

* 缺点：

  由公式可以看出，仍依赖于人工设置一个全局学习率 

  \eta设置过大的话，会使regularizer过于敏感，对梯度的调节太大

  中后期，分母上梯度平方的累加将会越来越大，使gradient-->0，使得训练提前结束

### Adadelta

方法：Adadelta是对Adagrad的扩展，最初方案依然是对学习率进行自适应约束，但是进行了计算上的简化。Adagrad会累加之前所有的梯度平方，而Adadelta只累加固定大小的项，并且也不直接存储这些项，仅仅是近似计算对应的平均值。即：

![n_t=\nu*n_{t-1}+(1-\nu)*g_t^2](https://zhihu.com/equation?tex=n_t%3D%5Cnu%2An_%7Bt-1%7D%2B%281-%5Cnu%29%2Ag_t%5E2)

![\Delta{\theta_t} = -\frac{\eta}{\sqrt{n_t+\epsilon}}*g_t](https://zhihu.com/equation?tex=%5CDelta%7B%5Ctheta_t%7D+%3D+-%5Cfrac%7B%5Ceta%7D%7B%5Csqrt%7Bn_t%2B%5Cepsilon%7D%7D%2Ag_t)
$$
n_t=v*n_{t-1}+(1-v)*g_t^2
$$

$$
\Delta \theta=-\frac {\eta} {\sqrt{n_t+\epsilon}}*g_t
$$

在此处Adadelta其实还是依赖于全局学习率的，但是作者做了一定处理，经过近似[牛顿迭代法](https://www.zhihu.com/question/20690553)（求根点）之后：

![E|g^2|_t=\rho*E|g^2|_{t-1}+(1-\rho)*g_t^2](https://zhihu.com/equation?tex=E%7Cg%5E2%7C_t%3D%5Crho%2AE%7Cg%5E2%7C_%7Bt-1%7D%2B%281-%5Crho%29%2Ag_t%5E2)

![\Delta{x_t}=-\frac{\sqrt{\sum_{r=1}^{t-1}\Delta{x_r}}}{\sqrt{E|g^2|_t+\epsilon}}](https://zhihu.com/equation?tex=%5CDelta%7Bx_t%7D%3D-%5Cfrac%7B%5Csqrt%7B%5Csum_%7Br%3D1%7D%5E%7Bt-1%7D%5CDelta%7Bx_r%7D%7D%7D%7B%5Csqrt%7BE%7Cg%5E2%7C_t%2B%5Cepsilon%7D%7D)


$$
E\begin{vmatrix} g^2 \end{vmatrix}_t=\rho *\begin{vmatrix} g^2 \end{vmatrix}_{t-1}+(1-\rho)*g_t^2
$$

$$
\Delta x_t=-\frac {\sqrt{\sum_{r=1}^{t-1}\Delta x_r}} {E\begin{vmatrix} g^2 \end{vmatrix}_t +\epsilon}
$$

> 其中E代表期望

* 优点

  训练初中期，加速效果不错，很快 ；

  训练后期，反复在局部最小值附近抖动；

* 缺点

### RMSprop

RMSprop可以算作Adadelta的一个特例： 

当![\rho=0.5](https://zhihu.com/equation?tex=%5Crho%3D0.5)时，![E|g^2|_t=\rho*E|g^2|_{t-1}+(1-\rho)*g_t^2](https://zhihu.com/equation?tex=E%7Cg%5E2%7C_t%3D%5Crho%2AE%7Cg%5E2%7C_%7Bt-1%7D%2B%281-%5Crho%29%2Ag_t%5E2)就变为了求梯度平方和的平均数。

如果再求根的话，就变成了RMS(均方根)：

![RMS|g|_t=\sqrt{E|g^2|_t+\epsilon}](https://zhihu.com/equation?tex=RMS%7Cg%7C_t%3D%5Csqrt%7BE%7Cg%5E2%7C_t%2B%5Cepsilon%7D)
$$
RMS\begin{vmatrix} g \end{vmatrix}_t=\sqrt{E\begin{vmatrix} g^2 \end{vmatrix}+\epsilon}
$$
此时，这个RMS就可以作为学习率\eta 的一个约束：

![\Delta{x_t}=-\frac{\eta}{RMS|g|_t}*g_t](https://zhihu.com/equation?tex=%5CDelta%7Bx_t%7D%3D-%5Cfrac%7B%5Ceta%7D%7BRMS%7Cg%7C_t%7D%2Ag_t)
$$
\Delta x_t=-\frac \eta {RMS\begin{vmatrix} g \end{vmatrix}_t}*g_t
$$

* 优点：

  其实RMSprop依然依赖于全局学习率。

  RMSprop算是Adagrad的一种发展，和Adadelta的变体，效果趋于二者之间。

  适合处理非平稳目标 - 对于RNN效果很好。

### Adam

>Adam是实际学习中最常用的算法

Adam(Adaptive Moment  Estimation)本质上是带有动量项的RMSprop，它利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。Adam的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳。公式如下：

![m_t=\mu*m_{t-1}+(1-\mu)*g_t](https://zhihu.com/equation?tex=m_t%3D%5Cmu%2Am_%7Bt-1%7D%2B%281-%5Cmu%29%2Ag_t)

   

![n_t=\nu*n_{t-1}+(1-\nu)*g_t^2](https://zhihu.com/equation?tex=n_t%3D%5Cnu%2An_%7Bt-1%7D%2B%281-%5Cnu%29%2Ag_t%5E2)

 

![\hat{m_t}=\frac{m_t}{1-\mu^t}](https://zhihu.com/equation?tex=%5Chat%7Bm_t%7D%3D%5Cfrac%7Bm_t%7D%7B1-%5Cmu%5Et%7D)

 

![\hat{n_t}=\frac{n_t}{1-\nu^t}](https://zhihu.com/equation?tex=%5Chat%7Bn_t%7D%3D%5Cfrac%7Bn_t%7D%7B1-%5Cnu%5Et%7D)

  

![\Delta{\theta_t}=-\frac{\hat{m_t}}{\sqrt{\hat{n_t}}+\epsilon}*\eta](https://zhihu.com/equation?tex=%5CDelta%7B%5Ctheta_t%7D%3D-%5Cfrac%7B%5Chat%7Bm_t%7D%7D%7B%5Csqrt%7B%5Chat%7Bn_t%7D%7D%2B%5Cepsilon%7D%2A%5Ceta)

 

其中，![m_t](https://zhihu.com/equation?tex=m_t)，![n_t](https://zhihu.com/equation?tex=n_t)分别是对梯度的一阶矩估计和二阶矩估计，u和v为衰减率，u通常为0.9，v通常为0.999,可以看作对期望![E|g_t|](https://zhihu.com/equation?tex=E%7Cg_t%7C)，![E|g_t^2|](https://zhihu.com/equation?tex=E%7Cg_t%5E2%7C)的估计；![\hat{m_t}](https://zhihu.com/equation?tex=%5Chat%7Bm_t%7D)，![\hat{n_t}](https://zhihu.com/equation?tex=%5Chat%7Bn_t%7D)是对![m_t](https://zhihu.com/equation?tex=m_t)，![n_t](https://zhihu.com/equation?tex=n_t)的校正，这样可以近似为对期望的无偏估计。可以看出，直接对梯度的矩估计对内存没有额外的要求，而且可以根据梯度进行动态调整，而![-\frac{\hat{m_t}}{\sqrt{\hat{n_t}}+\epsilon}](https://zhihu.com/equation?tex=-%5Cfrac%7B%5Chat%7Bm_t%7D%7D%7B%5Csqrt%7B%5Chat%7Bn_t%7D%7D%2B%5Cepsilon%7D)对学习率形成一个动态约束，而且有明确的范围。

 

* 优点：

  结合了Adagrad善于处理稀疏梯度和RMSprop善于处理非平稳目标的优点

  对内存需求较小 

  为不同的参数计算不同的自适应学习率 

  也适用于大多非凸优化- 适用于大数据集和高维空间

优化方法在实际中的直观体验:

![img](https://img2018.cnblogs.com/blog/1425630/201809/1425630-20180917092038787-2101213597.gif)

损失曲面的轮廓和不同优化算法的时间演化。 注意基于动量的方法的“过冲”行为，这使得优化看起来像一个滚下山的球

![img](https://img2018.cnblogs.com/blog/1425630/201809/1425630-20180917092153639-1928194678.gif)

## 参考文献

菜鸟教程（https://www.runoob.com/numpy/numpy-tutorial.html）

《TensorFlow——湿疹Google深度学习框架》 中国工信出版社

《深度学习最全优化方法总结比较》https://www.cnblogs.com/callyblog/p/8299074.html

《深度学习常见的优化方法》https://www.cnblogs.com/GeekDanny/p/9655597.html