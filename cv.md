2018.01.19
1、cnn  卷积核 默认写的是f*f大小 其实第三维默认和输入的第三维相同 举例来说：
假设输入为256x256x3的RGB图像，卷积核大小为3x3（第三维和输入相同，为3），个数为96个，则卷积的过程为每个3x3(x3)的卷积核和256x256x3的图像做卷积，得到的是一个hxh的feature图(第三维被消去，具体计算就是卷积核的3个通道（第三维）的3x3和RGB的3个通道分别做卷积后得到的值对应点相加最终得到一个feature图)，故最后的结果应该为hxhx96的特征图.故可以算出需要的参数总共为96(卷积核数)x3(输入的第三维)x3x3。
2、fcn的最后一个卷积层应该为kxkxn，其中kxk的feature图中的每一个像素点代表了原输入图像的一个patch，即最初卷积核划过的一部分，kxk的feature图中每个像素点的值就代表该部分patch属于某一类的得分，总共n个类别。把卷积核的每个像素点的值相加并求均值，就可以知道该图片属于该类的得分。

2018.01.20
1、fcn的上采样（恢复到原图像大小）：
    a.矩阵操作化的卷积（以2维为例，可推广到n维）：卷积层的所有卷积核的参数W形成的稀疏矩阵C为mxn,其中n为输入图拉成一维向量（l2r,u2d）对应的元素个数，m为输出图拉成一维向量（l2r,u2d）的元素个数。这样的话就可以把这个卷积层的结构重新看作全连接层的结构（输入和输出都被拉成了一维的向量）。然后反向传播时则使用转置的矩阵Ct来进行计算。
    b.cnn误差项的反向传播：具体见https://www.cnblogs.com/pinard/p/6494810.html
     1)、全连接层到卷积层/池化层
        同全连接一样计算
     2)、已知卷积层l的第k个卷积核的δl，求上一层的δ(l-1)
        δ(l−1)=δlx∂zl/∂z(l−1)=δl∗rot180(Wl)⊙f′(z(l−1))
        ∂J(W,b)/∂Wl=δl∗rot180(a(l−1)), 其中a(l-1)即为上一层最终输入到本层的特征图。
        ∂J(W,b)/∂bl=∑u,v(δl)u,v 即把δl各个子矩阵的项分别求和最终得一向量。
        note:对含有卷积(*)的式子进行求导时，卷积核需要翻转180度。
     3）、已知池化层的反向传播误差δl，求上一层的δ(l-1)
        δ(l−1)=upsample(δl)⊙f′(z(l−1))
        note:f=a*b,则∂f/∂a=axrot180(b)。

2018.01.22
1、现阶段的一些目标检测分割：RCNN->FastRCNN->FasterRCNN->R-FCN

2018.01.23
1、fcn的训练：和训练一般的分类网络一致，一般使用softmax交叉熵损失函数：
    Si=e^i/∑j(e^j),即假若存在一个数组表示一张图片属于各类的得分，则该图片属于各类的概率等于e^得分/e^得分的和。
2、fcn的标签制作，要把图片中的各个类别标注为不同种的颜色。。。。真麻烦，最后计算交叉熵损失是复原后的图片和标签图片分别用作logits和labels。
3、fcn的转置卷积层的参数也是通过反向传播来进行更新。
4、EuclideanLoss（欧拉距离损失），用作卷积层生成的heatmap和真实标签之间回归？？
    re:ground truth的heatmap标签的制作方式：根据对应的生成的和heatmap的尺寸生成同样大小的gaussian kernel，以对应的关节点为中心。

2018.01.24
1、fcn的卷积层，padding模式为"same"，故最终生成的heatmap大小仅与池化层有关。(最后两个1x1x1x1的卷积层虽然padding是"valid"但并不影响输出的大小)
2、关于intermediate loss，目的是为了缓和由于层数过深而导致的反向传播梯度消失问题，具体实现即在中间层多计算几个loss并加到最终loss中即可。
3、转置卷积的前向和反向传播:与对应的卷积层的C矩阵想反，前向传播时乘的时Ct，反向传播时乘的是C。


2018.01.25
1、residual blocks，对于blocks中的第一个blcok，为了使得输入的维度(x)和另一路经过卷积操作后输出的维度(F(x))相等，所以需要identity mapping那一路加一个1x1的卷积层以保证F(x)和x的维度相同才能进行相加。(或者使用zero padding，不会增加额外的参数，但没啥作用)另外，除了从第三层开始的residual blocks的第一个block的stride为2之外，其余的block的padding都是"same"模式。
2、关于tensorflow的variable_scope的问题：
3、注意，caffe中的batch normalization是分两步完成的，所以一个bn层后面还跟了一个scale层。
4、tensorflow中关于BN（Batch Normalization）的函数主要有两个，分别是：
tf.nn.moments
tf.nn.batch_normalization
5、关于如何将heatmap中的关节点位置转换为原图中的关节点位置的问题：
    方法1)、直接对heatmap进行resize至原图大小
    方法2)、转置卷积恢复至原图大小？
    方法3)、从heatmap的关节点位置映射回原感受野的中心位置？

2018.01.26
1、今天准备完成rhd数据集的相关处理和生成。
    1)、各个指节的英文：Thumb大拇指 index middle ring pinky小拇指
    2)、wrist 腕部 palm 手掌
    3)、tf.where(condition,x,y,name) 满足condition的返回x中的元素，不满足condition的返回y中的元素。
    4)、tf.one_hot(indices,depth,on\_value,off\_value,axis),其中indices表示每行的最值的索引，depth表示增加的一维,加到axis指定的位置，默认为-1，即加到最里面一维。其中索引的位置的值为on\_value,其他位置的值为off\_value。
    note:tensorflow的维度顺序，ex:[1,2,3,5]依次为从外到里，第0维，第1维，第2维，第3维。
    5)、python的切片[begin:end:delta],delta为负数时表示反过来取。
    6)、tf.concat(val,axis,name)，将val沿着第axis维连接起来，ex:
        # tensor t3 with shape [2, 3]
        # tensor t4 with shape [2, 3]
        tf.shape(tf.concat([t3, t4], 0))  # [4, 3]
        tf.shape(tf.concat([t3, t4], 1))  # [2, 6]  
    7)、tf.stack(val,axis,name),即stack之,增加一个第axis维,大小为val这个list的长度。
    8)、tf.reduce_min/max(input,axis,keep\_dims),沿第axis维寻找最值。若keep\_dim为false，设input为[dim0,dim1,...,dimn]，axis为k，则输出为[dim0,...dimk-1,dimk+1,...,dimn]

2018.01.29
1、tf.expande_dims(input,axis,name),在扩展一第axis维，大小为1。
2、tf.image.crop\_and\_resize(image,boxes,box\_ind,crop\_size,name),先从原图中cropboxes指定大小的crop\_image,再进行resize到crop\_size大小。其中boxes为一[batch\_size,4]的矩阵，第1维4个依次为左上角y，左上角x，右下角y，右下角x，并且是归一化后的坐标(即每个原值都除了原图像的height或width),box\_ind为[batch\_size],指定对应的第几张图像，crop\_size指定resize之后的图像大小。
3、tf.tile(input,multiples)将input的每一维重复multiples次形成一个新的tensor,ex:
    input:[a,b,c,d],multiples=[2],output:[a,b,c,d,a,b,c,d]
    input:[[2,4],[1,2]] ,multiples=[2,2],output:[[2,4,2,4],[2,4,2,4],[1,2,1,2],[1,2,1,2]] 2x2变成了4x4，注意,multiples的长度必须和input的维度相同。
4、Tensor.set\_shape(shape),由于tensorflow是图计算的模型，所以在真正计算之前并不知道tensor的真正大小，set\_shape用以提供额外的shape信息。
5、Q：python的staticmethod中的变量？
6、已知的关于segment的方法:
    1)、R-FCN
    2)、cpm
    TODO:了解目标检测定位和人体姿态检测的异同。
7、tf.transpose(input,perm=None) 根据perm中的维度先后重新排列input的各个维度，perm为none时即转置input。
8、背景的heatmap的值如何设置？(暂时设置为0)

2018.01.30
1、python的bool型转换为tensor？tf.cast
2、TODO:tf.Variable(),tf.get_variable()
3、使用卷积相关操作时传入的filter记得先创建相应的变量，不要传一个[x1,x2,x3,x4],这样传入的只是一个一维tensor！！！
4、tf.add_n(input,name),当要将多个值相加时用这个，这个不支持广播。
5、tf.trainable_variables(scope=Node) 返回所有可训练的变量。
6、tensorflow中任何tensor的值必须在sess.run(tensor)之后才知道。
7、训练参数的初始化。。。？

2018.01.31
1、不让一个variable训练更新的方法：
    1)、使用LOCAL_VARIABLES
        my\_local = tf.get\_variable("my\_local", shape=(), 
        collections=[tf.GraphKeys.LOCAL_VARIABLES])
    2)、设置trainable=false
        my\_non\_trainable = tf.get\_variable("my\_non\_trainable", 
                                   shape=(), 
                                   trainable=False)
    3)、opt.minimize()中var_list传入需要训练的variables。
2、关于sharing_variables的问题(即是否共享一些参数，如conv中的weights、bias
等):
    1)、显式传递tf.Variable对象，这样的对象是共享的？
    2)、隐式使用tf.variable_scope():
        a.不同名称的scope下的variable不共享；
        b.相同名称的scope，再设置resue=True，则variable是共享的。
3、ground truth heatmap中背景类怎么制作？
4、转置卷积需要加bias和激活吗？github上的实现好像只加了bias。
5、bilinear插值，插入的点的值由已存在的点和该点的距离决定，详见:http://cv-tricks.com/image-segmentation/transpose-convolution-in-tensorflow/

2018.02.02
1、二维guassian核函数中sigma似乎对loss影响非常大，sigma一大那么loss也变大，且越来越难下降。

2018.02.27
1、xshell从本机向服务器上传文件：rz 文件名；从服务器向本机下载文件：sz 文件名。
2、二维gaussian kernel制作(tensorflow)：tf.contrib.distributions.MultivariateNormalFullCovariance类，需要的参数：各个维度的均值，协方差矩阵。得到一个概率密度分布，再使用prob方法得到对应具体值时的值。
    note:inv 代表矩阵求逆;det代表求矩阵的值。

2018.02.28
1、psp-network:
    a.pyramid pooling,即多个池化尺度捕获不同子区域的信息，最后再concat，此方法优于直接全区域池化。
    b.使用了中继监督，并赋予中间的loss一定的权重，有助于训练，但测试时不使用。
    c.从实验结果来看，平均池化比最大池化好。
    d.使用了数据增强。
2、TODO
    读取预训练好的模型(不同的框架，如caffe到tf?)
3、heatmap预测关节点响应图得到的只是在图片上的2d位置，如何预测真实2d、3d关节位置？
4、多个人/手时的预测，已经存在遮掩情况下的预测？
5、衡量准确度时使用的是图像2d的uv坐标还是xy坐标？(应该是uv坐标)

2018.03.01
#TODO
1、mpii数据集的读取处理相关操作。
    a.输入和标签：.mat文件中一部分数据的缺失问题(为空)
2、多人时预测的heatmap如何区分各个人？(需要先使用目标检测框架如YOLO、mask RCNN等定位出每个人)


2018.03.02
1、读取.mat文件，scipy.io.loadmat(),mat文件版本更高的使用hdfs。
2、numpy.void类型的tmp，要查看各个类型使用tmp.dtype(返回一个dtype对象);各个fields使用tmp.dtype.fields(返回mappingproxy对象);tmp.dtype.fields.keys()返回dict_keys对象，里面包含各个key，若要直接的key列表，使用list(tmp.dtype.fields.keys())。

2018.03.03
1、测试时图片的尺寸处理？图片先被crop到256x256，需要先提供人的中心位置的信息？然后再进行预测。
2、坐标的转换问题。
3、from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
    print_tensors_in_checkpoint_file("test-ckpt/model-2", None, True)
    用以输出checkpoint中保存的东西
    def print_tensors_in_checkpoint_file(file_name, tensor_name, all_tensors,
                                     all_tensor_names):
  """Prints tensors in a checkpoint file.
  If no `tensor_name` is provided, prints the tensor names and shapes
  in the checkpoint file.
  If `tensor_name` is provided, prints the content of the tensor.
  Args:
    file_name: Name of the checkpoint file.
    tensor_name: Name of the tensor in the checkpoint file to print.
    all_tensors: Boolean indicating whether to print all tensors.
    all_tensor_names: Boolean indicating whether to print all tensor names.
4、tensorflow保存与恢复模型：http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/

2018.03.05
1、tf.boolean_mask(tensor,mask)
2、切片操作会造成降维，切记！！！！！！！！

2018.03.06
1、tf.Data API,新的生成batch的API。
    a.两个抽象：tf.data.Dataset、tf.data.Iterator。
    b.针对tfrecords的通用步骤：(有些未加参数)
        dataset=tf.data.TFRecordDataset(tf_file)
        dataset=dataset.map()
        dataset=dataset.shuffle()
        dataset=dataset.batch(self.batch_size)
        dataset=dataset.repeat()
        #以下为迭代器迭代着取数据
        iterator=dataset.make_one_shot_iterator()
        batched_images,batched_labels=iterator.get_next()

2、读取tfrecord时若各张image、各张label的维度不一定一致该如何处理？
    报错时都是由于reshape的shape参数不对，读取时记得shape这个tensor要么所有值已知，要么所有值未知，不能混！！
    （如[features['h'],features['w'],3]这样会报错）

2018.03.07
1、py3的tf的/操作，不管分子分母的类型，结果都是float64类型。而且分子分母类型需要一致。
2、逆序一个tensor [::-1]
3、tf.argmax(input,axis) 返回最大位置的索引,axis为None的话就按照0处理。

2018.03.08
1、取得一个运行时形状才确定的tensor的形状的某个维度，用tf.shape(xx)[x]。