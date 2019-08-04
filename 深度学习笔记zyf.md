# 深度学习笔记

## 一、numpy基础

 numpy数据结构是n维的数组对象，叫做ndarray。Python的list虽然也能表示，但是不高效，随着列表数据的增加，效率会降低。其中，切片操作及其相关符号非常重要：

":"用以表示当前维度的所有子模块；

"-1"用以表示当前维度所有子模块最后一个，"负号用以表示从后往前数的元素,-n即是表示从后往前数的第n个元素"；

：：的含义——start : end : step；

常用函数大全：https://blog.csdn.net/u011995719/article/details/71080987

### 1.一维数组处理

```python
#导入包并重命名
import numpy as np

#定义一维数组
a = np.array([2, 0, 1, 5, 8, 3])
print u'原始数据:', a

#输出最大、最小值及形状
print u'最小值:', a.min()
print u'最大值:', a.max()
print u'形状', a.shape

#数据切片
print u'切片操作:'
print a[:-2]
print a[-2:]
print a[:1]

#排序
print type(a)
a.sort()
print u'排序后:', a
```

输出：

```python
原始数据: [2 0 1 5 8 3]
最小值: 0
最大值: 8
形状 (6L,)
切片操作:
[2 0 1 5]
[8 3]
[2]
<type 'numpy.ndarray'>
排序后: [0 1 2 3 5 8]
```

注释：
1.代码通过np.array定义了一个数组[2, 0, 1, 5, 8, 3]，其中min计算最小值，max计算最大值，shape表示数组的形状，因为是一维数组，故6L（6个数字）。

2.最重要的一个知识点是数组的切片操作，因为在数据分析过程中，通常会对数据集进行"80%-20%"或"70%-30%"的训练集和测试集划分，通常采用的方法就是切片。
(1)a[:-2]表示从头开始获取，"-2"表示后面两个值不取，结果：[2 0 1 5]
(2)a[-2:]表示后往前数两个数字，获取数字至结尾，即获取最后两个值[8 3]
(3)a[:1]表示从头开始获取，获取1个数字，即[2]

### 2.二维数组处理

``` python
#定义二维数组
import numpy as np
c = np.array([[1, 2, 3, 4],[4, 5, 6, 7], [7, 8, 9, 10]])
 
#获取值
print u'形状:', c.shape
print u'获取值:', c[1][0]
print u'获取某行:'
print c[1][:]
print u'获取某行并切片:'
print c[0][:-1]
print c[0][-1:]
 
#获取具体某列值
print u'获取第3列:'
print c[:,np.newaxis, 2]
 
#调用sin函数
print np.sin(np.pi/6)
print type(np.sin(0.5))
 
#范围定义
print np.arange(0,4)
print type(np.arange(0,4))
```

输出结果：

``` python
形状: (3L, 4L)
获取值: 4
获取某行:
[4 5 6 7]
获取某行并切片:
[1 2 3]
[4]
获取第3列:
[[3]
 [6]
 [9]]
0.5
<type 'numpy.float64'>
[0 1 2 3]
<type 'numpy.ndarray'>
```

注释：

1.定义二维数组括号不要弄错，正确的应该是：[[1,2,3],[4,5,6]], 同时计算机的存储下标都是从0开始计算的。

2.获取二维数组中的某行，如第2行数据[4,5,6,7]，采用方法是:c[1] [:]。

3.获取二维数组中的某列，如第3列数据[[3] [6] [9]]，c[:,np.newaxis, 2]。因为通常在数据可视化中采用获取某列数据作为x或y坐标，同时多维数据也可以采用PCA降低成两维数据，再进行显示

## 二、pandas基础

Pandas是面板数据（Panel Data）的简写。它是Python最强大的数据分析和探索工具，因金融数据分析工具而开发，支持类似SQL的数据增删改查，支持时间序列分析，灵活处理缺失数据。它有两个主要的数据结构，Series和DataFrame，记住大小写区分。其中，相关知识需要了解：

data.describe()输出的相关信息——count：总数；mean：均值；std：标准差；min：最小值；25%：四分之一分位数；50%：中位数；75%：四分之三分位数；max：最大值；

数据提取用到三个函数loc,iloc和ix，loc函数按标签值进行提取，iloc按位置进行提取，ix可以同时按标签和位置进行提取。 

在pandas中用函数 isnull 和 notnull 来检测数据丢失：pd.isnull(a)、pd.notnull(b)，Series也提供了这些函数的实例方法：a.isnull()；

Pandas提供了大量的方法能够轻松的对Series，DataFrame和Panel对象进行各种符合各种逻辑关系的合并操作。如：Concat、Merge （类似于SQL类型的合并）、Append （将一行连接到一个DataFrame上）；

常用函数大全：https://blog.csdn.net/liufang0001/article/details/77856255



### 1.读写文件

对于电力用户数据集missing_data.xls文件，内容如下，共3列数据，分别是用户A、用户B、用户C，共21行，对应21天的用电量，其中包含缺失值。

```python
235.8333	324.0343	478.3231
236.2708	325.6379	515.4564
238.0521	328.0897	517.0909
235.9063		        514.89
236.7604	268.8324	
	        404.048     486.0912
237.4167	391.2652	516.233
238.6563	380.8241	
237.6042	388.023	    435.3508
238.0313	206.4349	487.675
235.0729		
235.5313	400.0787	660.2347
	        411.2069	621.2346
234.4688	395.2343	611.3408
235.5	    344.8221	643.0863
235.6354	385.6432	642.3482
234.5521	401.6234	
236	        409.6489	602.9347
235.2396	416.8795	589.3457
235.4896		        556.3452
236.9688		        538.347
```

输入：

```python
#读取数据 header设置Excel无标题头
import pandas as pd
data = pd.read_excel("missing_data.xls", header=None) 
print data
 
#计算数据长度
print u'行数', len(data)
 
#计算用户A\B\C用电总和
print data.sum()
 
#计算用户A\B\C用点量算术平均数
mm = data.sum()
print mm
 
#输出预览前5行数据
print u'预览前5行数据'
print data.head()
 
#输出数据基本统计量
print u'输出数据基本统计量'
print data.describe()
```

输出：

```python
           0         1         2
0   235.8333  324.0343  478.3231
1   236.2708  325.6379  515.4564
2   238.0521  328.0897  517.0909
3   235.9063       NaN  514.8900
4   236.7604  268.8324       NaN
5        NaN  404.0480  486.0912
6   237.4167  391.2652  516.2330
7   238.6563  380.8241       NaN
8   237.6042  388.0230  435.3508
...
行数 21
0    4488.9899
1    6182.3265
2    9416.3276
dtype: float64
0    4488.9899
1    6182.3265
2    9416.3276
dtype: float64
预览前5行数据
          0         1         2
0  235.8333  324.0343  478.3231
1  236.2708  325.6379  515.4564
2  238.0521  328.0897  517.0909
3  235.9063       NaN  514.8900
4  236.7604  268.8324       NaN
输出数据基本统计量
                0           1           2
count   19.000000   17.000000   17.000000
mean   236.262626  363.666265  553.901624
std      1.225465   57.600529   67.707729
min    234.468800  206.434900  435.350800
25%           NaN         NaN         NaN
50%           NaN         NaN         NaN
75%           NaN         NaN         NaN
max    238.656300  416.879500  660.234700
```

注释：

1.因为Excel表格中存在空值，故Python显示为NaN（Not a Number）表示空。

### 2.Series

 Series是一维标记数组，可以存储任意数据类型，如整型、字符串、浮点型和Python对象等，轴标一般指索引。
Series、Numpy中的一维array 、Python基本数据结构List区别：List中的元素可以是不同的数据类型，而Array和Series中则只允许存储相同的数据类型，这样可以更有效的使用内存，提高运算效率。

```python
from pandas import Series, DataFrame
 
#通过传递一个list对象来创建Series，默认创建整型索引；
a = Series([4, 7, -5, 3])
print u'创建Series:'
print a
 
#创建一个带有索引来确定每一个数据点的Series 
b = Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
print u'创建带有索引的Series:'
print b
 
#如果你有一些数据在一个Python字典中，你可以通过传递字典来创建一个Series
sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
c = Series(sdata)
print u'通过传递字典创建Series:'
print c
states = ['California', 'Ohio', 'Oregon', 'Texas']
d = Series(sdata, index=states)
print u'California没有字典为空:'
print d
```

输出：

```python
创建Series:
0    4
1    7
2   -5
3    3
dtype: int64
创建带有索引的Series:
d    4
b    7
a   -5
c    3
dtype: int64
通过传递字典创建Series:
Ohio      35000
Oregon    16000
Texas     71000
Utah       5000
dtype: int64
California没有字典为空:
California        NaN
Ohio          35000.0
Oregon        16000.0
Texas         71000.0
dtype: float64
```

注释：

1.Series的一个重要功能是在算术运算中它会自动对齐不同索引的数据，示例中当自定义的索引California 和字典队员不上时，会自动选择NaN，即结果为空，表示缺失。

### 3.DataFrame

 DataFrame是二维标记数据结构，列可以是不同的数据类型。它是最常用的pandas对象，像Series一样可以接收多种输入：lists、dicts、series和DataFrame等。初始化对象时，除了数据还可以传index和columns这两个参数。类似一张excel表格或者SQL，只是功能更强大。构建DataFrame的方法有很多，最常用的是传入一个字典。

```python
df = pd.DataFrame({"id":[1001,1002,1003,1004,1005,1006], 
 "date":pd.date_range('20130102', periods=6),
  "city":['Beijing ', 'SH', ' guangzhou ', 'Shenzhen', 'shanghai', 'BEIJING '],
 "age":[23,44,54,32,34,32],
 "category":['100-A','100-B','110-A','110-C','210-A','130-F'],
  "price":[1200,np.nan,2133,5433,np.nan,4432]},
  columns =['id','date','city','category','age','price'])
df.info()

```

输出

```python
RangeIndex: 6 entries, 0 to 5
Data columns (total 6 columns):
id          6 non-null int64
date        6 non-null datetime64[ns]
city        6 non-null object
category    6 non-null object
age         6 non-null int64
price       4 non-null float64
dtypes: datetime64[ns](1), float64(1), int64(2), object(2)
memory usage: 368.0+ bytes
```

注释：
1.DataFrame中常常会出现重复行，DataFrame的duplicated方法返回一个布尔型Series，表示各行是否是重复行；还有一个drop_duplicated方法，它返回一个移除了重复行的DataFrame。

## 二、示例代码调试

### 1.mnist_cnn.py

基于MNIST数据集的卷积神经网络。使用CNN对MNIST数据集（包含7千张28*28的单通道（灰度图、黑白图）图片）分类（0-9，10个类别）。mnist为手写数据集。MNIST（Mixed National Institute of Standards and Technology database）是一个计算机视觉数据集，它包含70000张手写数字的灰度图片，其中每一张图片包含 28 X 28 个像素点。

```python
#训练一个基于MNIST数据集的简单卷积神经网络
#12个周期后达到99.25%的精确度，通过参数调整还可提升精确度
#使用一个GRID K520 GPU （图形处理器）每个周期16秒
'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
#Python提供的__future__模块把下一个新版本的特性导入到当前版本
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
#指定进行梯度下降时每个batch包含的样本数，训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步
batch_size = 128
#label的维度数，这里用了10维onehot编码向量
num_classes = 10
#训练终止时的epoch值，由于这里没有设置 inital_epoch，它也是训练的总轮数
epochs = 12

#图像尺寸28*28(input image dimensions)
# input image dimensions
img_rows, img_cols = 28, 28

#下载MNIST数据集，同时获取训练集、验证集（the data, split between train and test sets）
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#tf或th为后端，采取不同参数顺序
#实际采用tf的参数顺序，把100张RGB三通道的16×32（高为16宽为32）彩色图表示为下面这种形式（100,3,16,32），其表达形式是（100,16,32,3），依次分别为样本维，高，宽与通道维。
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

#对训练和测试数据处理，转为float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#对数据进行归一化到0-1 因为图像数据最大是255
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#构建模型
model = Sequential()
#第一层为二维卷积层
#32为filters卷积核的数目，也为输出的维度
#kernel_size卷积核的大小，3x3
#激活函数选为relu 
#第一层必须包含输入数据规模input_shape这一参数，后续层不必包含
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#训练模型，载入数据，verbose=1为输出进度条记录
#validation_data为验证集
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
#开始评估模型效果
#verbose=0为不输出日志信息
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```



输出：

```python

runfile('C:/Users/54228/.spyder-py3/temp.py', wdir='C:/Users/54228/.spyder-py3')
Downloading data from https://s3.amazonaws.com/img-datasets/mnist.npz
11493376/11490434 [==============================] - 777s 68us/step
WARNING: Logging before flag parsing goes to stderr.
W0804 01:10:42.718356 25216 deprecation_wrapper.py:119] From D:\app\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:74: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.

x_train shape: (60000, 28, 28, 1)
60000 train samples
10000 test samples
W0804 01:10:43.178099 25216 deprecation_wrapper.py:119] From D:\app\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:517: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.

W0804 01:10:43.290821 25216 deprecation_wrapper.py:119] From D:\app\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:4138: The name tf.random_uniform is deprecated. Please use tf.random.uniform instead.

W0804 01:10:43.413498 25216 deprecation_wrapper.py:119] From D:\app\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:3976: The name tf.nn.max_pool is deprecated. Please use tf.nn.max_pool2d instead.

W0804 01:10:43.416486 25216 deprecation_wrapper.py:119] From D:\app\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:133: The name tf.placeholder_with_default is deprecated. Please use tf.compat.v1.placeholder_with_default instead.

W0804 01:10:43.437407 25216 deprecation.py:506] From D:\app\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:3445: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
Instructions for updating:
Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
W0804 01:10:43.526170 25216 deprecation_wrapper.py:119] From D:\app\Anaconda3\lib\site-packages\keras\optimizers.py:790: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.

W0804 01:10:43.533183 25216 deprecation_wrapper.py:119] From D:\app\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:3295: The name tf.log is deprecated. Please use tf.math.log instead.

W0804 01:10:43.858312 25216 deprecation.py:323] From D:\app\Anaconda3\lib\site-packages\tensorflow\python\ops\math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
Train on 60000 samples, validate on 10000 samples
Epoch 1/12
60000/60000 [==============================] - 112s 2ms/step - loss: 0.2627 - acc: 0.9197 - val_loss: 0.0621 - val_acc: 0.9803
Epoch 2/12
60000/60000 [==============================] - 120s 2ms/step - loss: 0.0922 - acc: 0.9726 - val_loss: 0.0401 - val_acc: 0.9864
Epoch 3/12
60000/60000 [==============================] - 99s 2ms/step - loss: 0.0676 - acc: 0.9792 - val_loss: 0.0397 - val_acc: 0.9869
Epoch 4/12
60000/60000 [==============================] - 104s 2ms/step - loss: 0.0550 - acc: 0.9839 - val_loss: 0.0320 - val_acc: 0.9893
Epoch 5/12
60000/60000 [==============================] - 110s 2ms/step - loss: 0.0478 - acc: 0.9860 - val_loss: 0.0310 - val_acc: 0.9894
Epoch 6/12
60000/60000 [==============================] - 110s 2ms/step - loss: 0.0423 - acc: 0.9872 - val_loss: 0.0281 - val_acc: 0.9906
Epoch 7/12
60000/60000 [==============================] - 115s 2ms/step - loss: 0.0391 - acc: 0.9882 - val_loss: 0.0300 - val_acc: 0.9899
Epoch 8/12
60000/60000 [==============================] - 86s 1ms/step - loss: 0.0357 - acc: 0.9892 - val_loss: 0.0271 - val_acc: 0.9910
Epoch 9/12
60000/60000 [==============================] - 83s 1ms/step - loss: 0.0332 - acc: 0.9903 - val_loss: 0.0289 - val_acc: 0.9904
Epoch 10/12
60000/60000 [==============================] - 79s 1ms/step - loss: 0.0315 - acc: 0.9900 - val_loss: 0.0274 - val_acc: 0.9911
Epoch 11/12
60000/60000 [==============================] - 80s 1ms/step - loss: 0.0281 - acc: 0.9915 - val_loss: 0.0282 - val_acc: 0.9919
Epoch 12/12
60000/60000 [==============================] - 83s 1ms/step - loss: 0.0281 - acc: 0.9917 - val_loss: 0.0309 - val_acc: 0.9908
Test loss: 0.030856597457616316
Test accuracy: 0.9908
```

### 2.MNIST in Keras.ipynb

```python
 %matplotlib inline
```

#### Import some prerequisites

```python
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7,7) # Make the figures a bit bigger

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
```

#### Load training data

```python
nb_classes = 10

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("X_train original shape", X_train.shape)
print("y_train original shape", y_train.shape)

for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_train[i], cmap='gray', interpolation='none')
    plt.title("Class {}".format(y_train[i]))
```

输出

```python
X_train original shape (60000, 28, 28)
y_train original shape (60000,)
```

![img](https://raw.githubusercontent.com/zhangyf99/Deeplearning/master/static/1-zyf.png)

#### Format the data for training

```python
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
```

输出：

```python
Training matrix shape (60000, 784)
Testing matrix shape (10000, 784)
```

#### Build the neural network

```python
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
```



#### Compile the model

```python
model.compile(loss='categorical_crossentropy', optimizer='adam')
```

#### Train the model

```python
#新版本中不再支持show_accuracy参数，升级后支持的参数可以参阅https://keras.io/models/model/。这里直接把代码中的两处show_accuracy参数删掉
model.fit(X_train, Y_train,
          batch_size=128, nb_epoch=4,
          verbose=1,
          validation_data=(X_test, Y_test))
```

输出：

```python
Train on 60000 samples, validate on 10000 samples
Epoch 1/4
60000/60000 [==============================] - 15s 255us/step - loss: 0.2481 - val_loss: 0.1120
Epoch 2/4
60000/60000 [==============================] - 13s 216us/step - loss: 0.1022 - val_loss: 0.0829
Epoch 3/4
60000/60000 [==============================] - 13s 220us/step - loss: 0.0708 - val_loss: 0.0625
Epoch 4/4
60000/60000 [==============================] - 12s 193us/step - loss: 0.0558 - val_loss: 0.0662
Out[10]:
<keras.callbacks.History at 0x1ccc53690b8>
```



#### Evaluate its performance

```python
score = model.evaluate(X_test, Y_test,
                     verbose=0)
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

输出：

```python
Test score: 0.0662498661814956
```

![img](https://raw.githubusercontent.com/zhangyf99/Deeplearning/master/static/2-zyf.png)

![img](https://raw.githubusercontent.com/zhangyf99/Deeplearning/master/static/3-zyf.png)

修改了两处代码：

1.删除了show_accuracy参数。原因：新版本中不再支持show_accuracy参数，升级后支持的参数可以参阅https://keras.io/models/model/。 这里直接把代码中的两处show_accuracy参数删掉。

2.修改了打印的evaluate()返回值。evaluate()用于评估已经训练过的模型，返回损失值与模型的度量值。但通过print(model.metrics_names)发现这里只返回了损失值。这可能是由于模型具有单个输出且没有度量。

##  三、优化方法
选择使用CNN进行图像分类任务时，需要优化3个主要指标：精度、仿真速度以及内存消耗。这些性能指标与设计的模型息息相关。不同的网络会对这些性能指标进行权衡，比如VGG、Inception以及ResNets等。常见的做法是对这些成熟的模型框架进行微调、比如通过增删一些层、使用扩展的其它层以及一些不同的网络训练技巧等完成相应的图像分类任务。
#### 1.智能卷积设计减少运行时间和内存消耗
CNN总体设计的最新进展已经有一些令人惊叹的替代方案，在不损失太多精度的前提下，可以加快CNN仿真运行的时间并减少内存消耗。以下所有的这些都可以很容易地集成到上述CNN成熟模型之中：

MobileNets：使用深度可分离卷积技术，在仅牺牲1%~5%的精度的条件下，极大地减少了计算量和内存消耗量，精度的降低程度与计算量和内存消耗量的下降成正比。
XNOR-Net：使用二进制卷积，即卷积核只有两种取值：-1或1。通过这种设计使得网络具有很高的稀疏性，因此可以很容易地压缩网络参数而不会占用太多内存。
ShuffleNet：使用逐点群卷积（pointwise group convolution）和信道重排（channel shuffle）大大降低计算成本，同时网络模型的精度要优于MobileNets。
Network Pruning（网络剪枝）：去除CNN模型的部分结构以减少仿真运行时间和内存消耗，但也会降低精度。为了保持精度，去除的部分结构最好是对最终结果没有多大的影响。

#### 2.网络深度

对于CNN而言，有一些常用的方法是增加通道数以及深度来增加精度，但是会牺牲仿真运行速度和内存。然而，需要注意的是，层数增加对精度的提升的效果是递减的，即添加的层越多，后续添加的层对精度的提升效果越小，甚至会出现过拟合现象。

![4](https://raw.githubusercontent.com/zhangyf99/Deeplearning/master/static/网络深度1.png)

![5](https://raw.githubusercontent.com/zhangyf99/Deeplearning/master/static/网络深度2.png)

#### 3.激活函数

对于神经网络模型而言，激活函数是必不可少的。传统的激活函数，比如Softmax、Tanh等函数已不适用于CNN模型，有相关的研究者提出了一些新的激活函数，比如Hinton提出的ReLU激活函数，使用ReLU激活函数通常会得到一些好的结果，而不需要像使用ELU、PReLU或LeakyReLU函数那样进行繁琐的参数调整。一旦确定使用ReLU能够获得比较好的结果，那么可以优化网络的其它部分并调整参数以期待更好的精度。

#### 4.空洞卷积

空洞卷积使用权重之间的间距以便能够使用远离中心的像素，这种操作允许网络在不增加网络参数的前提下增大感受野，即不增加内存消耗。相关论文表明，使用空洞卷积可以增加网络精度，但也增加仿真运行消耗的时间。

![6](https://raw.githubusercontent.com/zhangyf99/Deeplearning/master/static/空洞卷积.png)

#### 5.数据扩充

深度学习依赖于大数据，使用更多的数据已被证明可以进一步提升模型的性能。随着扩充的处理，将会免费获得更多的数据，使用的扩充方法取决于具体任务，比如，你在做自动驾驶汽车任务，可能不会有倒置的树、汽车和建筑物，因此对图像进行竖直翻转是没有意义的，然而，当天气变化和整个场景变化时，对图像进行光线变化和水平翻转是有意义的。

#### 6.算法优化

![ä» SGD å° Adam ââ æ·±åº¦å­¦ä¹ ä¼åç®æ³æ¦è§(ä¸)](https://pic4.zhimg.com/v2-4a3b4a39ab8e5c556359147b882b4788_1200x500.gif)

梯度下降是指，在给定待优化的模型参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta+%5Cin+%5Cmathbb%7BR%7D%5Ed) 和目标函数 ![[公式]](https://www.zhihu.com/equation?tex=J%28%5Ctheta%29) 后，算法通过沿梯度 ![[公式]](https://www.zhihu.com/equation?tex=%5Cnabla_%5Ctheta+J%28%5Ctheta%29) 的相反方向更新 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta) 来最小化 ![[公式]](https://www.zhihu.com/equation?tex=J%28%5Ctheta%29) 。学习率 ![[公式]](https://www.zhihu.com/equation?tex=%5Ceta) 决定了每一时刻的更新步长。对于每一个时刻 ![[公式]](https://www.zhihu.com/equation?tex=t) ，我们可以用下述步骤描述梯度下降的流程：

(1) 计算目标函数关于参数的梯度

![[公式]](https://www.zhihu.com/equation?tex=g_t+%3D+%5Cnabla_%5Ctheta+J%28%5Ctheta%29)

(2) 根据历史梯度计算一阶和二阶动量

![[公式]](https://www.zhihu.com/equation?tex=m_t+%3D+%5Cphi%28g_1%2C+g_2%2C+%5Ccdots%2C+g_t%29)

![[公式]](https://www.zhihu.com/equation?tex=v_t+%3D+%5Cpsi%28g_1%2C+g_2%2C+%5Ccdots%2C+g_t%29)

(3) 更新模型参数

![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta_%7Bt%2B1%7D+%3D+%5Ctheta_t+-+%5Cfrac%7B1%7D%7B%5Csqrt%7Bv_t+%2B+%5Cepsilon%7D%7D+m_t)

其中， ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon) 为平滑项，防止分母为零，通常取 1e-8。

##### 随机梯度下降算法（SGD）

没有动量的概念，即

![[公式]](https://www.zhihu.com/equation?tex=m_t+%3D+%5Ceta+g_t)

![[公式]](https://www.zhihu.com/equation?tex=v_t+%3D+I%5E2)

![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon+%3D+0)

这时，更新步骤就是最简单的

![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta_%7Bi%2B1%7D%3D+%5Ctheta_t+-+%5Ceta+g_t)

SGD 的缺点在于收敛速度慢，可能在鞍点处震荡。并且，如何合理的选择学习率是 SGD 的一大难点。

##### Momentum

SGD 在遇到沟壑时容易陷入震荡。为此，可以为其引入动量 Momentum[3]，加速 SGD 在正确方向的下降并抑制震荡。

![[公式]](https://www.zhihu.com/equation?tex=m_t+%3D+%5Cgamma+m_%7Bt-1%7D+%2B+%5Ceta+g_t)

SGD-M 在原步长之上，增加了与上一时刻步长相关的 ![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma+m_%7Bt-1%7D) ，![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma) 通常取 0.9 左右。这意味着参数更新方向不仅由当前的梯度决定，也与此前累积的下降方向有关。这使得参数中那些梯度方向变化不大的维度可以加速更新，并减少梯度方向变化较大的维度上的更新幅度。由此产生了加速收敛和减小震荡的效果。

![img](https://pic4.zhimg.com/80/v2-2476080e4cdfd489ae64ae3ceeafe48b_hd.jpg)图 1(a): SGD

![img](https://pic4.zhimg.com/80/v2-b9388fd6e465d82687680f9d16edcd2b_hd.jpg)图 1(b): SGD with momentum

从图 1 中可以看出，引入动量有效的加速了梯度下降收敛过程。

##### Nesterov Accelerated Gradient

![img](https://pic4.zhimg.com/80/v2-fecd469405501ad82788f068985b25cb_hd.jpg)图 2: Nesterov update

更进一步的，人们希望下降的过程更加智能：算法能够在目标函数有增高趋势之前，减缓更新速率。

NAG 即是为此而设计的，其在 SGD-M 的基础上进一步改进了步骤 1 中的梯度计算公式：

![[公式]](https://www.zhihu.com/equation?tex=g_t+%3D+%5Cnabla_%5Ctheta+J%28%5Ctheta+-+%5Cgamma+m_%7Bt-1%7D%29)

参考图 2，SGD-M 的步长计算了当前梯度（短蓝向量）和动量项 （长蓝向量）。然而，既然已经利用了动量项来更新 ，那不妨先计算出下一时刻 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta) 的近似位置 （棕向量），并根据该未来位置计算梯度（红向量），然后使用和 SGD-M 中相同的方式计算步长（绿向量）。这种计算梯度的方式可以使算法更好的「预测未来」，提前调整更新速率。

##### Adagrad

SGD、SGD-M 和 NAG 均是以相同的学习率去更新 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta) 的各个分量。而深度学习模型中往往涉及大量的参数，不同参数的更新频率往往有所区别。对于更新不频繁的参数（典型例子：更新 word embedding 中的低频词），我们希望单次步长更大，多学习一些知识；对于更新频繁的参数，我们则希望步长较小，使得学习到的参数更稳定，不至于被单个样本影响太多。

Adagrad[4] 算法即可达到此效果。其引入了二阶动量：

![[公式]](https://www.zhihu.com/equation?tex=v_t+%3D+%5Ctext%7Bdiag%7D%28%5Csum_%7Bi%3D1%7D%5Et+g_%7Bi%2C1%7D%5E2%2C+%5Csum_%7Bi%3D1%7D%5Et+g_%7Bi%2C2%7D%5E2%2C+%5Ccdots%2C+%5Csum_%7Bi%3D1%7D%5Et+g_%7Bi%2Cd%7D%5E2%29)

其中， ![[公式]](https://www.zhihu.com/equation?tex=v_t+%5Cin+%5Cmathbb%7BR%7D%5E%7Bd%5Ctimes+d%7D) 是对角矩阵，其元素 ![[公式]](https://www.zhihu.com/equation?tex=v_%7Bt%2C+ii%7D) 为参数第 ![[公式]](https://www.zhihu.com/equation?tex=i) 维从初始时刻到时刻 ![[公式]](https://www.zhihu.com/equation?tex=t)的梯度平方和。

此时，可以这样理解：学习率等效为 ![[公式]](https://www.zhihu.com/equation?tex=%5Ceta+%2F+%5Csqrt%7Bv_t+%2B+%5Cepsilon%7D) 。对于此前频繁更新过的参数，其二阶动量的对应分量较大，学习率就较小。这一方法在稀疏数据的场景下表现很好。

##### RMSprop

在 Adagrad 中， ![[公式]](https://www.zhihu.com/equation?tex=v_t) 是单调递增的，使得学习率逐渐递减至 0，可能导致训练过程提前结束。为了改进这一缺点，可以考虑在计算二阶动量时不累积全部历史梯度，而只关注最近某一时间窗口内的下降梯度。根据此思想有了 RMSprop[5]。记 ![[公式]](https://www.zhihu.com/equation?tex=g_t+%5Codot+g_t) 为 ![[公式]](https://www.zhihu.com/equation?tex=g_t%5E2) ，有

![[公式]](https://www.zhihu.com/equation?tex=v_t+%3D+%5Cgamma+v_%7Bt-1%7D+%2B+%281-%5Cgamma%29+%5Ccdot+%5Ctext%7Bdiag%7D%28g_t%5E2%29)

其二阶动量采用指数移动平均公式计算，这样即可避免二阶动量持续累积的问题。和 SGD-M 中的参数类似，![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma) 通常取 0.9 左右。

##### Adam

Adam[6] 可以认为是 RMSprop 和 Momentum 的结合。和 RMSprop 对二阶动量使用指数移动平均类似，Adam 中对一阶动量也是用指数移动平均计算。

![[公式]](https://www.zhihu.com/equation?tex=m_t+%3D+%5Ceta%5B+%5Cbeta_1+m_%7Bt-1%7D+%2B+%281+-+%5Cbeta_1%29g_t+%5D)

![[公式]](https://www.zhihu.com/equation?tex=v_t+%3D+%5Cbeta_2+v_%7Bt-1%7D+%2B+%281-%5Cbeta_2%29+%5Ccdot+%5Ctext%7Bdiag%7D%28g_t%5E2%29)

其中，初值

![[公式]](https://www.zhihu.com/equation?tex=m_0+%3D+0)

![[公式]](https://www.zhihu.com/equation?tex=v_0+%3D+0)

注意到，在迭代初始阶段，![[公式]](https://www.zhihu.com/equation?tex=m_t) 和 ![[公式]](https://www.zhihu.com/equation?tex=v_t) 有一个向初值的偏移（过多的偏向了 0）。因此，可以对一阶和二阶动量做偏置校正 (bias correction)，

![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7Bm%7D_t+%3D+%5Cfrac%7Bm_t%7D%7B1-%5Cbeta_1%5Et%7D)

![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7Bv%7D_t+%3D+%5Cfrac%7Bv_t%7D%7B1-%5Cbeta_2%5Et%7D)

再进行更新，

![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta_%7Bt%2B1%7D+%3D+%5Ctheta_t+-+%5Cfrac%7B1%7D%7B%5Csqrt%7B%5Chat%7Bv%7D_t%7D+%2B+%5Cepsilon+%7D+%5Chat%7Bm%7D_t)

可以保证迭代较为平稳。