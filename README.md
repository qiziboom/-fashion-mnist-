# -fashion-mnist-
2020人工智能课程设计程序

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os

fashion_MNIST = input_data.read_data_sets('D:/fashion-MNIST数据集', one_hot=True)#独热编码（仅一位为1）

x = tf.placeholder(tf.float32, [None,784])#训练集图片（数据类型、行不定，列为784）#限制每次训练的量
y = tf.placeholder(tf.float32, [None, 10])#训练集标签（数据类型、行不定、列为10）

sess = tf.InteractiveSession()

def weight_variable(shape):#初始化所有权重
    initial = tf.truncated_normal(shape, mean=0,stddev=0.1)#产生截断正态分布随机数（输出张量的维度，均值，标准差）
    return tf.Variable(initial)#创建变量

def bias_variable(shape):#初始化所有偏置
    initial = tf.constant(0.1, shape=shape)#创建常量（value、shape） #举例tf.constant[1,2,3],[3,2]),输出：三行两列矩阵[1,2],[3,3],[3,3],
    # 矩阵先依次填充value值，空余位置用最后一个value值填充
    return tf.Variable(initial)

#定义恒等残差模块：输入维度=输出维度
def identity_block(X_input, kernel_size, in_filter, out_filters, stage, block):

    # defining name basis
    block_name = 'res' + str(stage) + block
    f1, f2, f3 = out_filters
    with tf.variable_scope(block_name):
        X_shortcut = X_input

        #first
        W_conv1 = weight_variable([1, 1, in_filter, f1])#weight_variable参数：卷积核长、宽、输入通道数、输出通道数
        X = tf.nn.conv2d(X_input, W_conv1, strides=[1, 1, 1, 1], padding='SAME')#主要参数： tf.nn.conv2d（输入，卷积核，stride，padding）
        # tf 中stride是一维四元素的向量，第一个和第四个必须是1，只能修改中间两个：水平滑动、垂直滑动
        b_conv1 = bias_variable([f1])
        X = tf.nn.relu(X+ b_conv1)
        #second
        W_conv2 = weight_variable([kernel_size, kernel_size, f1, f2])
        X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
        b_conv2 = bias_variable([f2])
        X = tf.nn.relu(X+ b_conv2)
        #third
        W_conv3 = weight_variable([1, 1, f2, f3])
        X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
        b_conv3 = bias_variable([f3])
        X = tf.nn.relu(X+ b_conv3)
        #final step
        add = tf.add(X, X_shortcut)#跳跃连接
        #b_conv_fin = bias_variable([f3])
        add_result = tf.nn.relu(add)

    return add_result

#conv_block模块，由于该模块定义时输入和输出尺度不同，故需要进行卷积操作来改变尺度，从而得以相加
def convolutional_block( X_input, kernel_size, in_filter, out_filters, stage, block, stride=2):

    # defining name basis
    block_name = 'res' + str(stage) + block
    with tf.variable_scope(block_name):
        f1, f2, f3 = out_filters
        x_shortcut = X_input
        #first
        W_conv1 = weight_variable([1, 1, in_filter, f1])#仅改变神经元个数(输出维度)
        X = tf.nn.conv2d(X_input, W_conv1,strides=[1, stride, stride, 1],padding='SAME')
        b_conv1 = bias_variable([f1])
        X = tf.nn.relu(X + b_conv1)
        #second
        W_conv2 =weight_variable([kernel_size, kernel_size, f1, f2])
        X = tf.nn.conv2d(X, W_conv2, strides=[1,1,1,1], padding='SAME')#(实际卷积操作)
        b_conv2 = bias_variable([f2])
        X = tf.nn.relu(X+b_conv2)
        #third
        W_conv3 = weight_variable([1,1, f2,f3])
        X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1,1], padding='SAME')
        b_conv3 = bias_variable([f3])
        X = tf.nn.relu(X+b_conv3)
        #shortcut path
        W_shortcut =weight_variable([1, 1, in_filter, f3])
        x_shortcut = tf.nn.conv2d(x_shortcut, W_shortcut, strides=[1, stride, stride, 1], padding='VALID')
        #final
        add = tf.add(x_shortcut, X)
        #建立最后融合的权重
        #b_conv_fin = bias_variable([f3])
        add_result = tf.nn.relu(add)
    return add_result

x1 = tf.reshape(x, [-1, 28, 28, 1])
w_conv1 = weight_variable([2, 2, 1, 64])#参数：卷积核长、宽、输入通道数、输出通道数
x1 = tf.nn.conv2d(x1, w_conv1, strides=[1, 2, 2, 1], padding='SAME')
b_conv1 = bias_variable([64])
x1 = tf.nn.relu(x1+b_conv1)
# 这里操作后变成14x14x64
x1 = tf.nn.max_pool(x1, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

# stage 2
x2 = convolutional_block(X_input=x1, kernel_size=3, in_filter=64,  out_filters=[64, 64, 256], stage=2, block='a', stride=1)
# conv_block后，尺寸14x14x256
x2 = identity_block(x2, 3, 256, [64, 64, 256], stage=2, block='b' )
x2 = identity_block(x2, 3, 256, [64, 64, 256], stage=2, block='c')
# 张量尺寸14x14x256
x2 = tf.nn.max_pool(x2, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# 7x7x256
flat = tf.reshape(x2, [-1, 7*7*256])

w_fc1 = weight_variable([7 * 7 *256, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(flat, w_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

#建立损失函数，在这里采用交叉熵函数

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#初始化变量

sess.run(tf.global_variables_initializer())
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"###"=0"为默认值，输出所有信息；"=1"屏蔽通知信息；"=2"屏蔽通知信息和warning；"=3"屏蔽通知、warning、和报错
for i in range(3001):
    batch = fashion_MNIST.train.next_batch(10)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})
print("test accuracy %g"%accuracy.eval(feed_dict={x: fashion_MNIST.test.images, y: fashion_MNIST.test.labels, keep_prob: 1.0}))
