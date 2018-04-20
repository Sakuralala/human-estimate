//论文阅读。
//感觉可以研究的两个方向:
1、bottom-up一步生成各个关键点坐标的精确性，(ae好像对分类已经做的不错了)
2、3d的pose预测。
3、multi-scale的问题。
before:有空补。
补:
2018.03.27
1、Learning Feature Pyramids for Human Pose Estimation(在mpii单人测试依然排名第一)
主体结构还是hourglass，也属于top-down方法，不过作者主要focus在第二步，即关键点detection。主要创新点有以下几个：
a.对residual module进行了深入挖掘，一口气提出了四种称之为PRM(Pyramid Residual Modules,金字塔残差模块)的结构,具体来说就是传统的residual module都是针对一个scale的feature maps来进行特征提取，而PRM采用了多个scale的feature maps(具体方法就是下面提到的fractional pooling，分数卷积)。
b.fractional pooling,即卷积的缩小倍数不是整数倍，为的是获取多个scale的feature maps，但实测效果不佳，有可能是参数没调好或者其他未知原因。
c.提出了一中多branch的初始化方式，但是在github上的代码里并未释出。
d.解决了variance explosion的问题,即在hourglass的上采样那里identity mapping改为了一个卷积操作。

2018.03.28
1、Adversarial PoseNet:A Structure-aware Convolutional Network for Human Pose Estimation
适用于：单人，detection那步也被省略了。把hourglass和gan网络结合了。(作者在github上说1月就放出代码，现在去看，他竟然把那个responsity删了。。。。)
网络结构：
    a.生成网络G，主体结构为hourglass的copy，区别在于输出，改为两种heatmaps，一种为gt,一种为occlussion；另外，在一个hourglass中又分为两个部分，上半部分产生的gt，下半部分产生occl。
    b.判别网络分为两个P与C，其中C用以判断一个heatmap的置信度，输出一个1*16的vector，每个element表示对应关键点heatmap位置的得分；P用以判断关节点的位置是否正确，也输出一个1\*16的vector，每个element代表是否在正确位置。
    loss为g的loss(应该为mse)+两个判别网络的损失。

2、RMPE：Regional Multi-Person Pose Estimation(最新更新的版本在mpii多人测试中已经排名第一，但没有登出不知为何)
SJTU的一篇论文，典型的top-down方法，其中两个部分的主体方法都是用的现成的，即faster-rcnn+pyranet。
主要的创新点的话有以下几个：
a.一个对称的stn网络，其中取得参数theta的localisation网络在定位网络之后，de-transform在pyranet之后(用以将经过仿射变换的图像变换回原坐标);
b.一个并行的stack hourglass网络，这个网络是为了不影响主路的精度，从实验结果来看并没啥大影响(降了1.7个百分点);
c.参数化的nms(非最大值抑制)，为的是减小detection部分的冗余，消除标准就是一堆公式，感觉很吃硬件,对最终结果影响很大，降了大概5个百分点；
d.Pose-guided Proposals Generator,对于划分好的atom pose(元姿态),通过学习不同atom pose的detected bounding box和gt bounding box之间的相对偏移的分布，并通过k-means算法聚类找到相似的pose，然后通过detection生成的bounding box来生成一个对应的伪gt bounding box来提升性能,这个对最终结果影响最大，有8个百分点；

3、




2018.03.23
1、cpn(级联金字塔网络)
用于:关键点检测,属于top-down方法。
大体方法:
a.网络结构。属于top-down类,其中第一步的detector基于fpn(特征金字塔网络)+mask rcnn中的roiAlign,属于直接拿来用的部分,略过不谈;第二步即关键点检测部分,采用的是两个net,
    其一称为globalnet,用以检测easy points,即容易定位的关键点,整体是个U-shape,类似fpn,原理大概是浅层特征能够较好地定位关键点信息,但是由于感受野较小,难以提供用以识别的语义信息;深层特征由于感受野较大拥有较丰富的语义信息但由于conv和pool的原因分辨率较小难以提供准确的关键点信息;因此使用U-shape结构可以较好地融合这些信息来获得比较好的检测定位效果(说白了就是将各个层的信息相加呗),至于loss的计算,由于是在每个融合特征层进行分别预测,那么尺寸是不一样的,所以对每一层而言heatmap的尺寸也是不同的(应该是进行resize操作了的？);
    其二称为refinenet,用来检测定位前面globalnet不能很好地预测的关键点hard points,如被遮挡的部分,具体来说就是globalnet中的每层feature经过一定数量的bottleneck(应该就是residual模块,浅层用的少,深层用的多)后再经过upsampling操作,最后不同层的结果进行concat(与hourglass的+不同)。另外,为了让refinenet专注于定位hard points,原文采用了一个称之为hard points online studying的方法,说白了就是设定一个hardpoints的数量m,然后在refinenet反向传播loss的时候只把所有关键点里loss为前m个的传回去(具体实现,个人觉得是把其他的关键点的loss置为0即可),原文中选的m值为总关键点数量的一半,效果最好。
b.数据预处理。原文实验表明256\*192的效果和256\*256的效果一样好,因为人的尺寸一般不是方形的,但更节省内存,可以用于提高batch size;图片尺寸越大,效果越好;rotation、scale等常用手段,以及large batch,这个很重要。

2.fpn(特征金字塔网络)
用于:理论上可用于任何cnn结构。
大体方法:
a.网络结构,和一般的bottom-up不断subsampling并提取高层特征再top-down不断upsampling融合各层特征且只在最后一个upsampling层进行预测不同,fpn在top-down阶段的每一层融合特征后都进行预测,上一层的特征经过upsampling后与subsampling阶段对应channel的特征进行融合然后再进行预测,每层都如此。
b.数据预处理,未明。

2018.03.26
1.associated embeddings:xxxx
用于:多人目标检测和实例分割(好像这块作者做的不是很好暂时不提了)。本文作者从开始就一直在输出一个观点，即许多cv任务都可以看作是两个大的过程:detection+grouping,检测+分组。用在human pose estimation这方面，就是检测各个关键点+将各个关键点分离为独立的个人，本文所设计的两个损失函数也是基于这一观点出发的。(不知道是不是观点输出太强烈了所有被ICCV给rejected了。。。)
大体方法:
a.网络结构,大致同之前的hourglass,但是存在以下更改:
    1、残差模块改为3*3卷积;
    2、在每个stage进行subsampling时输出的特征图增多(256->386->512->768);
    3、输出的heatmap变为两种,分别为:
        detection heatmaps,即检测各个关节点位置的heatmaps,且为多人的(即一个heatmap中可能存在peaks),对应的detection loss即为mse，和之前的相同;
        tag heatmaps,这个比较有意思，有点类似半监督的方式？(不太懂),即让网络自己去学习对各个不同的人的不同关节点进行编码,使得同个人的各个关节点的编码值应该是一样的，而不同的人之间关节点的编码值应该是区别较大的，这就引出了作者设计的group loss,主要分为两部分:
            第一部分是针对单人内部各个关键点而言的，本质上就是计算方差，即先根据gt位置找到tag heatmaps中的各个值，然后先算出所有关键点的值的均值，接着计算方差，然后对图中的所有人进行相同的操作并对结果取均值即构成了group loss的第一部分；
            第二部分是针对不同人之间而言的，计算出每个人各个关键点的均值之后，所有人两两之间的均值的差平方后再取负并作为exp的指数，易知这一项应该越小越好，即不同个体间的编码值差距越大越好。
        两部分相加即构成了所谓的group loss。
        根据两个损失的定义可以看到，其实detection和group在训练时是没有什么交集的(loss之间可能存在相互制约？这方面没细想),训练时group操作使用的关键点坐标是ground truth的而不是predicted(说不定可以在这个地方做下文章？),而且从后文的实验结果可以看出，作者在一部分测试集上也使用了ground truth 的坐标而不是predicted的坐标来进行实验，结果提升非常大，这也从侧面说明了限制最终结果的瓶颈在于关键点的detection，且从实验结果放出的图可以看出，tag heatmaps对不同的人的关键点的分割已经十分准确(对于单个人的各个关节点而言，tag的value基本上在一个数值上；对于不同人之间的关键点而言，数值基本上差距十分明显)。另外，在具体实现过程中作者在group loss之前加了一个系数1e-3。
b.具体实现细节:
    1、测试时如何将各个关键点分发到相应的个人？
        采用的方法是迭代并从躯干、头部再到四肢，根据躯干关键点对应的detection heatmap通过nms(非最大值抑制，说白了就是通过设置一个threshold找到各个peaks,感觉这个阈值的设置还是挺重要的，设置小了会导致判别人数多了，设置大了可能导致判别人数比实际的少)来确定人数，并找到对应tag heatmap中peak位置对应的值，这样就形成了初始的状态，然后开始迭代过程：根据各个关键点的detection heatmaps找到对应peak位置tag heatmaps的值，并根据其peak的值的大小及对应tag的值的大小共同判断是否属于现有人物中的某个的关键点(具体来说还是设置一个阈值),若对于现有的所有人都不match，则新增一个人物(说明该人物的某些关键点被遮挡或截断了)；重复上述过程直至所有的关键点都有所属人物为止。
        另外，在测试中作者采用了多个scale来应对不同scale的人的问题，相应的detection heatmaps取了avg，而对于tag heatmaps则进行了concat操作变成一个向量，然后直接比较向量距离而不是如上述所说根据peak值大小及tag值大小共同决定所属类别。

2.A simple yet effective baseline for 3d human pose estimation
大体方法：
a.网络结构，hourglass+直接回归。感觉没啥亮点，要说有的话，可能就是回归那里用了个类似残差模块的东西，即把最初输入加到输出里了。

2018.03.27
1、Towards accurate multi-person estimation in the wild+PersonLab
a.网络结构：faster-rcnn+cnn(先扣出单个人再进行关键点定位,具体结构为resnet类)
把问题看成是分类+回归问题，首先是分类(如下a所述)：将heatmap中各个像素点分为两类，值为1的表示在真实位置一定范围内，否则为0；然后是回归(如下b所述)：即对每一个关键点k生成一个矢量图，表明每个像素点的位置与关键点k的位置的偏移(xk-x),从图上来看，就类似于对于每个gt关键点k，其他像素点均从自己出发指向k。
主要是改变了heatmaps的生成方式，从原本简单的二维高斯核变为：
    a.对于距离gt keypoint位置小于一定距离R的像素点值均置为1，形成一个圆形;
    b.offset vector，即每个像素点的值为其位置与真实位置的差，是个2-channel向量(personlab中限定了像素点为a中的值为1的)。
    最终生成的heatmap由这两个融合而成，是一种hough transform(霍夫变换)的形式。
    所谓hough voting:
    对于图像空间中的任意一个点(x,y)对应于霍夫空间中的一条曲线，然后霍夫空间中的对应曲线上的每一个点都进行+1操作;对图像中每个点都进行上述操作后，最后在霍夫空间中强度最大的一个点就是要找的点(霍夫空间的坐标),再将其映射回原图像空间即可得到对应的形状。
    然后论文中的bilinear kernel，又被称为Triangular kernel，即三角核函数。一维时的表达式为f(x)=1-|x|,也很容易推出二维时的表达式:g(x,y)=1-sqrt(x\*\*2+y\*\*2)
    
    另外，a、b的生成是使用了atrous conv(空洞卷积)。
    与此同时，相应的损失函数也改了：
    对于a而言，使用的损失函数为logistic loss，即经典的二分类损失,计算loss是是对整幅图片尺寸大小进行计算(排除了关键点未完全标记的部分);
    对于b而言，使用的损失函数为L1 loss(personlab)/Huber loss(G-RMI),计算loss时只计算在R范围内的(personlab)并且,除以R以normlize(personlab);
    另外两种loss间存在权重关系。
    Q:
    1、对于被遮挡的点，probmaps应该为全零还是和未被遮挡的点一致？对于不存在的点，probmaps应该为全零？那么对应的offset应该为什么形式？
    2、通过probmaps和offsets来得到locmaps时,若使用循环那么速度太慢，但如果使用6维的张量([h,w,b,h,w,n])则会报memory error，所以要怎么办呢？
b.


2、ArtTrack: Articulated Multi-person Tracking in the Wild
deeper cut的进化版，相比于其他的不咋地。

3、Binarized Convolutional Landmark Localizers for Human Pose Estimation and Face Alignment with Limited Resoures
目标是在保存性能的情况下减小gpu显存的利用,主要使用的是二值卷积，即权重的值不是1就是-1。
结论：binary效果相比非binary的不太行(降了大概10多个百分点),但是作者声称可以在单个cpu上实现实时的效果;但是使用它改进的residual module能在减小计算负担的情况下稍微提升测试结果。


2018.04.10
1、Faster-RCNN two-stage算法，第一步产生region proposal(RPN网络)，第二步对这些proposal进行分类及回归操作(fast rcnn网络)。
可以看作是RPN+Fast-RCNN,其中RPN用以产生候选区域和概率，用作Fast RCNN的输入。
    a.RPN的大致流程：
    ①.输入的图像通过一个cnn(resnet等)提取特征并生成一系列feature maps;
    ②.在feature maps上做slide操作。具体来说，即对feature map做3*3卷积(不知道这个是干啥，可能是为了再提取一次信息、进一步提高reception field？)。对于一个h\*w的feature map上的每一个像素点均对应有k个不同的anchor(即候选区域，不同的scale和ratio,为什么叫做anchor，本人的理解是这个anchor候选框就像是一个基准一样，后面得到预测的bounding box时是通过上一次更新后的anchor作为基准再进行相应的线性变换从而得到的和gt bounding box更接近的框,另外，anchor的四个坐标是根据图片的大小预先生成的，基本上可以涵盖各种各样的形状),故对于一张feature map，可以产生h\*w\*k个anchor;
    ③.分两路，一路用作分类，即对每个anchor进行正负label的判别，产生2k个scores(分为前景和背景，对feature map中的每个像素点而言),故最终的输出为h\*w\*2k,使用的是softmax交叉熵损失(logistic也可);另一路用作回归，对feature map中的每个像素点均产生4k个坐标((x,y,w,h),中心坐标和anchor的宽、高,表示预测的值)，总共为h\*w\*4k。
        note:关于回归的loss，不是直接计算预测的四个值和真实的差距，而是使用了所谓的parameterizated coor,即学习一种变换t,使得anchor经过变换t后得到的bounding box(也是新的anchor)和gt bounding box差距减小，设真实的变换为t*,则优化的目标就是是学习的变换t和真实的变换t*之间的差距尽可能地小，这里回归部分使用的是smooth_l1_loss。另外，论文中采取的变换为平移(改变x、y)+缩放(改变h、w)，对应的变换公式为：
        x_pred=t_x\*w_a+x_a,->t_x=(x_pred-x_a)/w_a
        y_pred=t_y\*h_a+y_a,->t_y=(y_pred-y_a)/h_a
        w_pred=w_a\*exp(t_w),->t_w=log(w_pred/w_a)
        h_pred=h_a\*exp(t_h),->t_h=log(h_pred/h_a)
        其中t=[t_x,t_y,t_w,t_h]即为需要学习的参数。
    ④.proposal layer。大致步骤如下：
        根据t来生成新的anchors；
        使用分类路的分数大小来对anchor进行排序，取前N个；
        将anchor坐标映射回原图像进行边界超出判断，剔除超出图像尺寸的anchor；
        剔除尺寸过小的anchor；
        进行NMS(非最大值抑制);
        再次排序，取前x个作为输出(即候选),注意此处输出的是[x1,y1,x2,y2]为左上角和右下角的坐标；
    RPN部分到此即结束。
    b.ROI pooling，因为传统的cnn训练好之后输入图片的尺寸必须固定，而在RPN部分生成的proposal尺寸并不固定，传统的解决方法即crop或warp，但会破坏原图的结构，故faster rcnn采用了所谓的roi pooling,接收两个输入，一个是共享conv产生的feature maps，一个是proposals。
    原理：先将proposal映射回原feature map尺度h\*w，然后将每个proposal垂直和水平分别分为pooled_h份和pooled_w份，并对每一份进行max pooling处理，这样输出的就是固定尺度的proposal了(pooled_h\*pooled_w,注意实际过程中可能不整除，所以在tf中需要自己写一个op,在最新的tensorflow object detection模型中，采用的实现方式是：we use Tensorflow’s “crop and resize” operation which uses bilinear interpolation to resample part of an image onto a fixed sized grid)。
    c.再次对上一步获取的固定大小的proposals进行分类(和前面不同，是类别分类而不是前后景分类)与回归(和前面相同，为的是进一步生成更精确的坐标)，得到更为精确的坐标和分类结果。


    d.实际训练过程：
        1、在已经训练好的model上，训练RPN网络，对应stage1_rpn_train.pt
        2、利用步骤1中训练好的RPN网络，收集proposals，对应rpn_test.pt
        3、第一次训练Fast RCNN网络，对应stage1_fast_rcnn_train.pt
        4、第二训练RPN网络，对应stage2_rpn_train.pt
        5、再次利用步骤4中训练好的RPN网络，收集proposals，对应rpn_test.pt
        6、第二次训练Fast RCNN网络，对应stage2_fast_rcnn_train.pt

2018.04.11
1、YOLO类 one-stage算法，使用一个cnn框架进行回归和分类。
    v1:



2018.04.20
1、Adversarial Complementary Learning for Weakly Supervised Object Localization
用fcn做分类时最终产生的c个类别的feature maps，研究发现分类任务往往是靠目标类别的某一部分来进行的，即对应的featuremaps只有对应的部分响应较高，这篇文章使用的方法就是强制先把第一个分类器产生的locationmaps的响应部分置0然后扔给第二个分类器强迫它通过别的部分来进行分类，这样最后把两个locationmaps进行max(pixel_a,pixel_b)，即找到每个对应像素位置的最大值来得到整个目标的响应。

