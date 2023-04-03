	实现基于center loss的人脸识别算法
特征提取网络可选择ResNet18、ResNet50等网络结构。
本次实验网络模型是ResNet，如下表所示，共有5组卷积。第一组卷积的输入大小是224x224，第五组卷积的输出大小是7x7，缩小了32倍。每次缩小2倍，总共缩小5次。
数据集采用LFW(Labeled Faces in the Wild) 人脸数据库

实验环境
ubuntu系统
python 3.6
pytorch 0.4
torchvision 0.2
matplotlib
opencv-python
requests
scikit-learn
tqdm


	运行main.py来训练模型
python main.py --arch resnet18 --batch_size 64 --epochs 50
#网络 batchsize epoch都可手动更改


	用训练好的模型做人脸验证。对于images文件夹中给定的四张人脸图像，计算其两两之间的特征向量的距离。
python main.py --verify-model logs/models/epoch_50.pth.tar --verify-images images/Taylor1.jpg, images/Taylor2.jpg
运行该指令会输出两张图片特征之间的distance。
