2018.01.19
1、cnn  卷积核 默认写的是f*f大小 其实第三维默认和输入的第三维相同 举例来说：
假设输入为256x256x3的RGB图像,卷积核大小为3x3（第三维和输入相同,为3）,个数为96个,则卷积的过程为每个3x3(x3)的卷积核和256x256x3的图像做卷积,得到的是一个hxh的feature图(第三维被消去,具体计算就是卷积核的3个通道（第三维）的3x3和RGB的3个通道分别做卷积后得到的值对应点相加最终得到一个feature图),故最后的结果应该为hxhx96的特征图.故可以算出需要的参数总共为96(卷积核数)x3(输入的第三维)x3x3。
2、fcn的最后一个卷积层应该为kxkxn,其中kxk的feature图中的每一个像素点代表了原输入图像的一个patch,即最初卷积核划过的一部分,kxk的feature图中每个像素点的值就代表该部分patch属于某一类的得分,总共n个类别。把卷积核的每个像素点的值相加并求均值,就可以知道该图片属于该类的得分。

2018.01.20
1、fcn的上采样（恢复到原图像大小）：
    a.矩阵操作化的卷积（以2维为例,可推广到n维）：卷积层的所有卷积核的参数W形成的稀疏矩阵C为mxn,其中n为输入图拉成一维向量（l2r,u2d）对应的元素个数,m为输出图拉成一维向量（l2r,u2d）的元素个数。这样的话就可以把这个卷积层的结构重新看作全连接层的结构（输入和输出都被拉成了一维的向量）。然后反向传播时则使用转置的矩阵Ct来进行计算。
    b.cnn误差项的反向传播：具体见https://www.cnblogs.com/pinard/p/6494810.html
     1)、全连接层到卷积层/池化层
        同全连接一样计算
     2)、已知卷积层l的第k个卷积核的δl,求上一层的δ(l-1)
        δ(l−1)=δlx∂zl/∂z(l−1)=δl∗rot180(Wl)⊙f′(z(l−1))
        ∂J(W,b)/∂Wl=δl∗rot180(a(l−1)), 其中a(l-1)即为上一层最终输入到本层的特征图。
        ∂J(W,b)/∂bl=∑u,v(δl)u,v 即把δl各个子矩阵的项分别求和最终得一向量。
        note:对含有卷积(*)的式子进行求导时,卷积核需要翻转180度。
     3）、已知池化层的反向传播误差δl,求上一层的δ(l-1)
        δ(l−1)=upsample(δl)⊙f′(z(l−1))
        note:f=a*b,则∂f/∂a=axrot180(b)。

2018.01.22
1、现阶段的一些目标检测分割：RCNN->FastRCNN->FasterRCNN->R-FCN

2018.01.23
1、fcn的训练：和训练一般的分类网络一致,一般使用softmax交叉熵损失函数：
    Si=e^i/∑j(e^j),即假若存在一个数组表示一张图片属于各类的得分,则该图片属于各类的概率等于e^得分/e^得分的和。
2、fcn的标签制作,要把图片中的各个类别标注为不同种的颜色。。。。真麻烦,最后计算交叉熵损失是复原后的图片和标签图片分别用作logits和labels。
3、fcn的转置卷积层的参数也是通过反向传播来进行更新。
4、EuclideanLoss（欧拉距离损失）,用作卷积层生成的heatmap和真实标签之间回归？？
    re:ground truth的heatmap标签的制作方式：根据对应的生成的和heatmap的尺寸生成同样大小的gaussian kernel,以对应的关节点为中心。

2018.01.24
1、fcn的卷积层,padding模式为"same",故最终生成的heatmap大小仅与池化层有关。(最后两个1x1x1x1的卷积层虽然padding是"valid"但并不影响输出的大小)
2、关于intermediate loss,目的是为了缓和由于层数过深而导致的反向传播梯度消失问题,具体实现即在中间层多计算几个loss并加到最终loss中即可。
3、转置卷积的前向和反向传播:与对应的卷积层的C矩阵想反,前向传播时乘的时Ct,反向传播时乘的是C。


2018.01.25
1、residual blocks,对于blocks中的第一个blcok,为了使得输入的维度(x)和另一路经过卷积操作后输出的维度(F(x))相等,所以需要identity mapping那一路加一个1x1的卷积层以保证F(x)和x的维度相同才能进行相加。(或者使用zero padding,不会增加额外的参数,但没啥作用)另外,除了从第三层开始的residual blocks的第一个block的stride为2之外,其余的block的padding都是"same"模式。
2、关于tensorflow的variable_scope的问题：
3、注意,caffe中的batch normalization是分两步完成的,所以一个bn层后面还跟了一个scale层。
4、tensorflow中关于BN（Batch Normalization）的函数主要有两个,分别是：
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
    3)、tf.where(condition,x,y,name) 满足condition的返回x中的元素,不满足condition的返回y中的元素。
    4)、tf.one_hot(indices,depth,on\_value,off\_value,axis),其中indices表示每行的最值的索引,depth表示增加的一维,加到axis指定的位置,默认为-1,即加到最里面一维。其中索引的位置的值为on\_value,其他位置的值为off\_value。
    note:tensorflow的维度顺序,ex:[1,2,3,5]依次为从外到里,第0维,第1维,第2维,第3维。
    5)、python的切片[begin:end:delta],delta为负数时表示反过来取。
    6)、tf.concat(val,axis,name),将val沿着第axis维连接起来,ex:
        # tensor t3 with shape [2, 3]
        # tensor t4 with shape [2, 3]
        tf.shape(tf.concat([t3, t4], 0))  # [4, 3]
        tf.shape(tf.concat([t3, t4], 1))  # [2, 6]  
    7)、tf.stack(val,axis,name),即stack之,增加一个第axis维,大小为val这个list的长度。
    8)、tf.reduce_min/max(input,axis,keep\_dims),沿第axis维寻找最值。若keep\_dim为false,设input为[dim0,dim1,...,dimn],axis为k,则输出为[dim0,...dimk-1,dimk+1,...,dimn]

2018.01.29
1、tf.expande_dims(input,axis,name),在扩展一第axis维,大小为1。
2、tf.image.crop\_and\_resize(image,boxes,box\_ind,crop\_size,name),先从原图中cropboxes指定大小的crop\_image,再进行resize到crop\_size大小。其中boxes为一[batch\_size,4]的矩阵,第1维4个依次为左上角y,左上角x,右下角y,右下角x,并且是归一化后的坐标(即每个原值都除了原图像的height或width),box\_ind为[batch\_size],指定对应的第几张图像,crop\_size指定resize之后的图像大小。
3、tf.tile(input,multiples)将input的每一维重复multiples次形成一个新的tensor,ex:
    input:[a,b,c,d],multiples=[2],output:[a,b,c,d,a,b,c,d]
    input:[[2,4],[1,2]] ,multiples=[2,2],output:[[2,4,2,4],[2,4,2,4],[1,2,1,2],[1,2,1,2]] 2x2变成了4x4,注意,multiples的长度必须和input的维度相同。
4、Tensor.set\_shape(shape),由于tensorflow是图计算的模型,所以在真正计算之前并不知道tensor的真正大小,set\_shape用以提供额外的shape信息。
5、Q：python的staticmethod中的变量？
6、已知的关于segment的方法:
    1)、R-FCN
    2)、cpm
    TODO:了解目标检测定位和人体姿态检测的异同。
7、tf.transpose(input,perm=None) 根据perm中的维度先后重新排列input的各个维度,perm为none时即转置input。
8、背景的heatmap的值如何设置？(暂时设置为0)

2018.01.30
1、python的bool型转换为tensor？tf.cast
2、TODO:tf.Variable(),tf.get_variable()
3、使用卷积相关操作时传入的filter记得先创建相应的变量,不要传一个[x1,x2,x3,x4],这样传入的只是一个一维tensor！！！
4、tf.add_n(input,name),当要将多个值相加时用这个,这个不支持广播。
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
2、关于sharing_variables的问题(即是否共享一些参数,如conv中的weights、bias
等):
    1)、显式传递tf.Variable对象,这样的对象是共享的？
    2)、隐式使用tf.variable_scope():
        a.不同名称的scope下的variable不共享；
        b.相同名称的scope,再设置resue=True,则variable是共享的。
3、ground truth heatmap中背景类怎么制作？
4、转置卷积需要加bias和激活吗？github上的实现好像只加了bias。
5、bilinear插值,插入的点的值由已存在的点和该点的距离决定,详见:http://cv-tricks.com/image-segmentation/transpose-convolution-in-tensorflow/

2018.02.02
1、二维guassian核函数中sigma似乎对loss影响非常大,sigma一大那么loss也变大,且越来越难下降。

2018.02.27
1、xshell从本机向服务器上传文件：rz 文件名；从服务器向本机下载文件：sz 文件名。
2、二维gaussian kernel制作(tensorflow)：tf.contrib.distributions.MultivariateNormalFullCovariance类,需要的参数：各个维度的均值,协方差矩阵。得到一个概率密度分布,再使用prob方法得到对应具体值时的值。
    note:inv 代表矩阵求逆;det代表求矩阵的值。

2018.02.28
1、psp-network:
    a.pyramid pooling,即多个池化尺度捕获不同子区域的信息,最后再concat,此方法优于直接全区域池化。
    b.使用了中继监督,并赋予中间的loss一定的权重,有助于训练,但测试时不使用。
    c.从实验结果来看,平均池化比最大池化好。
    d.使用了数据增强。
2、TODO
    读取预训练好的模型(不同的框架,如caffe到tf?)
3、heatmap预测关节点响应图得到的只是在图片上的2d位置,如何预测真实2d、3d关节位置？
4、多个人/手时的预测,已经存在遮掩情况下的预测？
5、衡量准确度时使用的是图像2d的uv坐标还是xy坐标？(应该是uv坐标)

2018.03.01
#TODO
1、mpii数据集的读取处理相关操作。
    a.输入和标签：.mat文件中一部分数据的缺失问题(为空)
2、多人时预测的heatmap如何区分各个人？(需要先使用目标检测框架如YOLO、mask RCNN等定位出每个人)


2018.03.02
1、读取.mat文件,scipy.io.loadmat(),mat文件版本更高的使用hdfs。
2、numpy.void类型的tmp,要查看各个类型使用tmp.dtype(返回一个dtype对象);各个fields使用tmp.dtype.fields(返回mappingproxy对象);tmp.dtype.fields.keys()返回dict_keys对象,里面包含各个key,若要直接的key列表,使用list(tmp.dtype.fields.keys())。

2018.03.03
1、测试时图片的尺寸处理？图片先被crop到256x256,需要先提供人的中心位置的信息？然后再进行预测。
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
1、tf.boolean_mask(tensor,mask) 用以隐藏tensor中的特定元素。
2、切片操作会造成降维,切记！！！！！！！！

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
    报错时都是由于reshape的shape参数不对,读取时记得shape这个tensor要么所有值已知,要么所有值未知,不能混！！
    （如[features['h'],features['w'],3]这样会报错）

2018.03.07
1、py3的tf的/操作,不管分子分母的类型,结果都是float64类型。而且分子分母类型需要一致。
2、逆序一个tensor [::-1]
3、tf.argmax(input,axis) 返回最大位置的索引,axis为None的话就按照0处理。

2018.03.08
1、取得一个运行时形状才确定的tensor的形状的某个维度,用tf.shape(xx)[x]。

2018.03.09
1、写csv文件：
 f=open('result.csv','a',newline='')
    writer=csv.writer(f)
    for key in ret:
        res=[key,cat]
        for elem in ret[key]:
            res.append(elem)
        writer.writerow(res)

2、载入模型
        saver = tf.train.import_meta_graph(meta文件位置)
        saver.restore(sess,check_point位置)
        graph = tf.get_default_graph()
        取得一个tensor
        graph.get_tensor_by_name(xxx) 

3、为了便于test,输入最好使用place_holder以便测试时换输入。

4、字符串存入tfrec,先将str转换为bytes,xx.encode()
sess.run()之后,再使用xx.decode()来解码为字符串。

5、get_shape()、shape()区别？
    tf中tensor有两种shape,分别为static (inferred) shape和dynamic (true) shape,其中static shape用于构建图,由创建这个tensor的op推断（inferred）得来,故又称inferred shape。如果该tensor的static shape未定义,则可用tf.shape()来获得其dynamic shape。
    a.tf.shape()用以获取动态shape,即在运行时才知道的shape;get_shape()返回的是静态shape。
    b.tf.shape()中数据的类型可以是tensor, list, array;a.get_shape()中a的数据类型只能是tensor,且返回的是一个元组（tuple）
    c.set_shape()、reshape()的区别。set_shape更新tensor的static shape,不改变dynamic shape。reshape创建一个具备不同dynamic shape的新的tensor。

6、gpu并行化使用,模型并行如何设置。

2018.03.11
1、将loss的更新放在cpu上以降低显存使用？
2、tf.estimator 
3、logging device placement,sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))。这样会输出op和tensor放置的位置。

2018.03.13
1、制作heatmap时sigma(标准差)是个影响很大的参数,太大的话难以拟合,太小的话容易过拟合。。。


2018.03.14
1、生成gt_heatmap时如果resize的话那么峰值也会相应地变小,容易直接拟合为0？？？
尝试不resize gt_heatmap而是resize输出,这样gt_heatmap的峰值就不会变小了。

2018.03.19
1、np.meshgrid(x,y)(以二维为例,更高维类似),返回一个xv、yv的矩阵,假设x为[n],y为[m],则xv、yv均为[m,n],对于xv而言,每一行相同,且为x,对应y而言,每一列相同,且均为y,结合xv、yv各个相同索引的值即可得到一个对应的位置。
ex:
    x:[1,2,3]
    y:[0,1]
    ->xv:[[1,2,3],[1,2,3]];yv:[[0,0,0],[1,1,1]]
    结合对应位置即可得各个点的位置：(1,0)、(2,0)、(3,0)、(1,1)、(2,1)、(3,1)。

2、hourglass 是对位置信息非常敏感的结构,所以训练的时候如果切出了目标,那么测试时也需要切出目标(即需要先知道目标的位置)!!!!

3、在原图上画圈,cv2.circle()

4、图像变换相关(以2d affine transform 即仿射变换为例)
首先,记K=[x,y,1]T(T表示转置),M=[[a,b,c],[d,e,f],[0,0,1]](2维坐标加一维是为了实现平移变换),因为只关心二维坐标,所以可以把M矩阵的第三维略去,则M=[[a,b,c],[d,e,f]]
    a.identity transform(无变换)
    M=[[1,0,0],[0,1,0]]
    b.scale(缩放)
    M=[[p,0,0],[0,q,0]]
    c.rotate
    M=[[cosa,-sina,0],[sina,cosa,0]]。
    d.shear
    M=[[1,b,0],[d,1,0]]
    note:上述操作可叠加,即各种复杂的变换均可以通过一个矩阵来实现。(各种变换矩阵的乘积)
    ??如何实现crop操作？？

2018.03.20
1、np.prod(x) 返回乘积;np.vstack((a,b)),按列stack a和b中的元素。
    ex:
    a = np.array([[1], [2], [3]])
    b = np.array([[2], [3], [4]])
    np.vstack((a,b))
    array([[1],
           [2],
           [3],
           [2],
           [3],
           [4]])

2、np.resize(a,new_shape);注意当new_shape大于原形状时,会添加元素(copy of a);np.matmul(a,b),矩阵乘积,如果a、b均大于二维,则最后两维用以做矩阵乘积并根据前面的维进行broadcasting。

3、当图像经过仿射变换之后,各个转换后的像素点的位置大概率不在一个像素点的坐标中心(即新的像素点位置可能出现小数的情况),此时就需要进行插值操作。
    注意,在实际操作过程中仿射变换分三步：
    a.生成目标坐标点的sampling grid,即根据图片尺寸生成对应的目标图像target坐标grid;
    b.生成转换矩阵M并与target坐标相乘,得到各个target点在源图像下的坐标source grid；
    c。利用插值变换计算target各个坐标点的像素值。

4、tensorflow的affine transformation实现：
tf.contrib.image.transform(
    images,
    transforms,
    interpolation='NEAREST',
    name=None
),直接把三步合为一步实现了。

5、bn和relu放在conv前/后？

2018.03.22
1、cpm、hourglass都要求输入的图像目标在中央位置,应该说训练时和测试时要一致,不抠出目标的比抠出目标的效果差。
2、bottom-up方法,不需要预先知道各个人的位置信息(PAF,但是,为什么呢？)；top-down,主要为detector+单人姿态估计。
相比而言,top-down方法对硬件的要求高,实时性+准确性方面,top-down方法的alphapose超过了bottom-up方法的openpose及mask-rcnn等,但是我决定这个太吃硬件了。
6、tf.nn.l2_loss() tf.losses.mean_squared_error(),后者默认是除了像素平均值的。。。
    note:
        a.mse均方误差损失：tf.reduce_mean(tf.square(pred-gt)) 即对每个样例误差的平方之和求均值。
        b.l2_loss:tf.sqrt(tf.reduce_sum(tf.square(pred-gt)))
    作为损失函数时,两者的最终效果会差很多？？ 

2018.03.24
1、数据特征的标准化,目的是为了把各个范围的特征数据标准化到一个一样的尺度,有利于加快学习的速度和防止学习的扭曲。
2、batch normalization。
    起因：每一层输入的数据的均值和方差都不一样,数据的分布也不一样,这样一来就难以有效地进行前向传播和反向传播,例如如果采用tanh激活函数,若输入中的某个值x过大,则易进入tanh函数的饱和区,使得x对应的输出接近1,也就是说输入就算再扩大经过激励后产生的输出也没有什么差别,即神经元对大的数据已经不敏感了,所以需要对输入的数据进行归一化,使得输入经过激励函数的敏感区域,有利于神经网络学习到有效的特征,而在反向传播时,经过每一层的梯度是需要和对应层的参数w相乘的,而由于数据已经经过了标准化,使得不同scale的参数w前面有一个不同的系数,scale大w的系数小,而scale小的w系数大,这样就有效地减小了梯度爆炸或弥散的可能性。
    具体做法：通过计算一个batch内的均值与方差来进行特征归一化,使得之后的数据均值为0,方差为1。注意,bn应该放在激励之前做,因为对于激励函数而言,输入数据的分布非常重要,batch normalization的目的就是使得输入的分布在激励函数的敏感区域内,这样更有利于激励函数的非线性化操作。
    具体公式：
    对于一个小批次中的输入数据x1-xm,可得其均值为x_avg=sum(xi)/m,而方差则为var=sum((xi-x_avg)**2)/m,则经过归一化之后的数据为xi_norm=(xi-x_avg)/sqrt(var+a),易知xi_norm的均值为0,方差为var/(var+a)(加个很小的数a是为了避免除以0),最后输出的结果yi=gamma\*xi_norm+beta,其中的gamma和beta是需要网络自己去学习和更新的参数,更通俗的叫法,gamma称为scale,即放缩尺度,beta称为shift,即偏移。
3、group normalization(组归一化)。
    起因：batch normalization需要较大的batch size,否则会导致批统计不准确而提高模型的错误率。


2018.03.26
1、预测各个关键点时,被遮挡的关键点对应的heatmap清零与否对最终效果的影响？


2018.03.29
1、双线性插值不可微分？所以在计算网络中不要用？要改为转置卷积？作者也是这样做的(可以用) 不管了。

2018.03.30
1、numpy经过shuffle、batch后喂入tf的一种方法。
    for epoch in range(0,2000):
        permutation=np.random.permutation(input_size)
        permutation=permutation[0:batch_size]
        batch=[train_set[permutation],train_label[permutation]]
        sess.run(train_step,feed_dict={X:batch[0],Yreal:batch[1]}
        

2、python的多进程/线程相关,学习一下:
    a.主要用到的库:multiprocessing(多进程,对于计算密集型推荐这个)、Queue+threading(这两个管多线程,Queue用以保存多线程处理完的结果)。
    python的Global Interpreter Lock(GIL)让python在同一时刻只能处理一个东西,解释器的C语言实现部分在完全并行执行时并不是线程安全的。实际上,解释器被一个GIL保护着,它确保任何时候都只有一个Python线程执行,这是为了保证线程读写时的安全性(因为各个线程共用一个内存空间，所以在某个线程要对内存空间进行读写操作时必须获得锁)。这对单核cpu来说没什么影响，但是对多核cpu来说就有了，即每次只有一个cpu上能运行线程，其他线程只能睡眠直至GIL锁被释放，所以要执行计算密集型任务时推荐使用多进程,io密集型推荐使用多线程(因为大部分时间都耗在等待io上，不吃cpu)。
    b.协程 主要是关键字yield，经常与multiprocessing联合使用以提高效率。它不是新的线程，而是在一个线程中可以通过yiled中断的一段程序，中断之后可以转而执行其他代码段，在合适时候再转回来执行，减去了多线程切换的开销,Python对协程的支持是通过generator实现的(即使用yield的代码段就变成一个生成器)。
        1、使用yield关键字的代码段就变成了一个协程；
        2、协程的接受值的方式：调用send()方法向其发送数据；
        3、接收数据后的协程不会立即执行，而是由next()方法调用后才执行;(等同于send(None))
        4、调用close()方法以关闭一个协程。
    总的来说，yield类似于return，返回yield右边的值，但是下次再执行时从yield之后开始执行;send()方法传的参数传给yield，用以赋值给yield左边的值。
    ex:
        def consumer():
            r = ''
            while True:
                n = yield r#返回的值
                if not n:
                    return
                print('[CONSUMER] Consuming %s...' % n)
                r = '200 OK'

        def produce(c):
            c.send(None)#启动协程 返回的为''
            n = 0
            while n < 5:
                n = n + 1
                print('[PRODUCER] Producing %s...' % n)
                r = c.send(n)#n为发送的值
                print('[PRODUCER] Consumer return: %s' % r)
            c.close()#关闭协程

        c = consumer()
        produce(c)
    
    c.Event类(mp的Event类直接从threading中copy，官方文档上说的，所以两者是一个东西),用于进程/线程间的事件响应。一个进程/线程发送信号(set()方法),其他进程/线程等待接收这个信号(wait())。
    ex:master进程/线程产生Event实例,slaver进程/线程不停work直至master进程/线程调用了event.set(),然后所有进程/线程终止。

    d.Queue类(包括manager.Queue类)实现了多生产者多消费者模型，即可以放心地从queue中get和向queue中put。Queue的实现是以deque作为底层的，put最终调用的是deque的append,而deque的append是线程安全的。
    note:进程间的通信方式主要包括:Queue、Pipe、Event、Socket。
    note2:mp.Queue()和mp.Manager().Queue()是不同的。
    e.multiprocessing.Process产生的子进程和主进程是并行的，即各自的地址空间是独立的，不共享,刚创建时把父进程的所有资源给子进程复制一份(linux下通过fork调用完成，在windows下不存在fork调用，采用的方法如下：将父进程的所有资源pickle后通过pipe传送给子进程，是一个heavy的过程)。Process进程创建时，子进程会将主进程的Process对象完全复制一份，这样在主进程和子进程各有一个Process
   对象，但是p1.start()启动的是子进程，主进程中的Process对象作为一个静态对象存在，不执行。
    f.Manager类，注意其也返回一个子进程。官方文档上说：Manager processes will be shutdown as soon as they are garbage collected or their parent process exits. 所以，什么时候会被gc并shutdown呢？

3、原图经过旋转之后heatmap的坐标还是得重新计算，因为如果直接把对应的heatmap图选转一定角度，可能出现以下情况：原图中的某个关节点被旋转出去了，但是虽然对应heatmap中的关节点也被旋转出去，但由于是一个二维高斯分布，对于的heatmap还存在大于零的点，这样会导致误判。

2018.03.31
1、py传递函数参数时，可以把函数的参数一并传进。(见test.py)
    ex:
        def fun1(x,y):
            pass
        def fuc2(func):
            pass
        
        func2(func1(1,2))

2018.04.02
1、生成对应坐标的mesh:
    x=np.linspace(0,255,256)
    y=np.linspace(0,255,256)
    x_t,y_t=np.mesh_grid(x,y)

2、cv2.resize(img,dsize),dsize坐标为先x后y！！！

3、shit！ windows下不能使用multiprocessing来进行并行的数据生成。。因为generator objects不能被pickle,原因是，on Windows, multiprocessing cannot marshall objects that contain generators across process boundaries.(即generator不能被pickle，也就不能通过pipe传送到另一个进程)
在linux系统下，受益于fork，子进程可以继承父进程的许多共享资源(拷贝)；而在windows下,由于不存在fork调用，新产生的子进程和父进程完全是没有关系的，子进程不继承父进程的任何资源，所以为了模拟fork调用(使得子进程父进程拥有一样的资源),父进程必须把所有的data先进行pickle，然后通过pipe传给子进程,而由于generator是不能被pickle的，原因很多，主要原因如下:
    a.需要保存generator的bytecode,但这个东西并不保证向下兼容；
    b.需要保存generator的frame,包括local variables, closures(结束位置？) and the instruction pointer(指令指针)等,要保存这个，需要使得整个解释器能够被pickle，故这需要对整个cpython解释器核心部分进行大量修改，不太现实。
            (以下为keras enquegenerator的建议解决办法)
            # On Windows, avoid **SYSTEMATIC** error in `multiprocessing`:
            # `TypeError: can't pickle generator objects`
            # => Suggest multithreading instead of multiprocessing on Windows
            raise ValueError('Using a generator with `use_multiprocessing=True`'
                             ' is not supported on Windows (no marshalling of'
                             ' generators across process boundaries). Instead,'
                             ' use single thread/process or multithreading.')
            (以下为我的解决方法)
            总的来说就是使用迭代器来代替生成器。具体来说，即写一个用以数据处理和生成的类，然后:
            1、类中建立一个叫做next()的方法,数据的预处理和最后图片和标签batch的生成均写进next()方法中，而后generatorenqueuer接收的参数即这个类的一个实例即可。
            2、原本生成器自动保存的一些局部变量及状态等，直接存进类中。 

4、multiprocessing 共享numpy的array。(TODO)
    a.np.memmap
    b.sharedmem模块
    note:可以直接从原始csv、json文件读入并直接处理，这样也非常快(非常容易就把cpu、gpu利用率占满了)，而且占用内存小。

2018.04.03
1、multiprocessing，父进程中打开的文件在子进程中却是关闭的？
    note:三种方法start一个子进程：
        a.spawn(windows下默认),子进程仅继承能够调用run()方法的必要的资源，不必要的文件描述符和handle将不会被继承。但是什么是必要的资源呢？
        b.fork(linux下默认)
        c.forkserver

2、cv2、plt 的imshow()方法，由于cv2默认使用bgr通道，所以为了正常显示图像，需要调用：   
   image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   cv2.addWeighted(img1,w1,img2,w2,gamma)用以将两者图片进行叠加操作。


3、mp.manager()的坑:
    ex:
        class A():
            def __init__(self):
                self.l=[]
                self.mg=None
            
            def start(self,num):
                self.mg=mp.Manager()
                for i in range(num):
                    #if I used self.mg=mp.Manager(),
                    # and the target=self.fuck,then an error occured:OSError: handle is closed
                    #but if the target=fuck,then it works fine.
                    #if I don't use self.mg=mp.Manager(),then both work fine.
                    p1=mp.Process(target=fuck)
                    p1.daemon=True
                    p1.start()

            def fuck(self):
                b=9

        def fuck():
            b=10

        if __name__=='__main__':
            freeze_support()
            a=A()
            a.start(5)
        若调用mp.Manager,则不能再target中使用self.fuck,若不调用，则两者都可。(TODO)

4、tensorboard相关：
    tf.summary.:image、histgram、scalar(常用的三种)
    记得最后需要tf.summary.merge_all()


2018.04.04
1、python的生成器，若是nested(即多个嵌套的),则每个都需要调用一次next()方法。
2、cv2的函数里要输入坐标或size时默认顺序是先x后y的。
3、np.where(condition):
    返回的是一个tuple,其中tuple的长度为输入的array的维数，
    tuple中的每个elem为一个一维array，其中array中的每个元
    素代表对应维的索引，结合各个elem中相同位置的元素则可得到
    一个完整的索引。

4、常用的四种交叉熵(cross-entropy)损失函数的定义：
    首先，最大似然估计(MLE):已知一组数据的观测值x,未知的分布模型model,则有x出现的概率P(x|model),最大似然估计就是寻找模型的一组参数值θ使得出现观测值x的概率P(x|model)最大化。即求：
    argmaxθ(p(x1|θ)\*p(x2|θ)\*p(xn|θ)),由于连乘可能使得结果接近于0，所以实际操作时一般取对数(对数使得连乘变成连加，而又由于log为单调递增，故不会改变极值的位置)
    其次是相对熵，假设存在两个数据的分布p、q，则其相对熵定义为：
    D(p||q)=∑p(x)log(p(x)/q(x))=∑p(x)\*log(p(x))-∑p(x)\*log(q(x));其中H(p)=∑p(x)\*log(p(x))称为分布p的熵，
    而CE(p,q)=-∑p(x)\*log(q(x))即称为p、q的交叉熵。
    然后，one-hot编码，主要是为了将标签转换为数字特征，对一些不存在相应大小关系的标签进行编码,表明属于/不属于。

    a.sigmoid、 logistic回归。
    适用条件：每个实例属于的类可以不唯一，即可以用于多元问题，但是不可以直接用在多分类问题中，可通过one-hot编码进行变通从而达到多分类的效果(即从多分类问题变为多个二分类问题(属于、不属于))，综上，即可用在多元多分类问题中。
    令z为真实标签值([batch_size,classes]，值非0即1)，x为预测的标签值,因为是二分类问题，分布的取值非0即1，所以交叉熵可以简化为:
    CE=-(p(x)\*log(q(x))+(1-p(x))\*log(1-q(x)))
    即有损失函数：
    L=-(z\*log(sigmoid(x))+(1-z)\*log(1-sigmoid(x))),其中z、x若未经过one-hot编码，则适用于传统的二分类问题(单元)，若经过one-hot编码，则适用于多元/单元多分类问题。
    b.softmax。 
    softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)
    适用条件：多分类问题，每个实例属于的类必须唯一,因为softmax函数将各个输入映射到一个概率，且总和为1。
    L=-∑z*log(softmax(x)),其中z、x均已经过one-hot编码。

2018.04.05
1、bn的顺序问题:
    a.conv->bn->relu  (post-activation)
      He 15年的论文：Convolution then batch normalization then ReLU as described by:
      Deep Residual Learning for Image Recognition
      https://arxiv.org/pdf/1512.03385.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.
    b.bn->relu->conv  (pre-activation)
      He 16年的论文：Batch normalization then ReLu then convolution as described by:
      Identity Mappings in Deep Residual Networks
      https://arxiv.org/pdf/1603.05027.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
      用后面这个效果更好。
    
2018.04.08
1、cv2.BlurGaussian(src,size,sigmaX),对一张图片做高斯模糊。注意，size必须为奇数。
2、cv2.resize()不限制通道数。

2018.04.09
1、tf.merge(input),指定merge的summary，input为tf.summary.xxx返回的值。

2018.04.10
1、person re-id？