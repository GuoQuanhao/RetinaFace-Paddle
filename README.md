# RetinaFace-Paddle

论文提出了一种强大的单阶段面部检测器`RetinaFace`，通过联合外监督（`extra-supervised`）和自监督（`self-supervised`）多任务学习的优势，在各种规模的人脸上执行像素级人脸定位，并实现了在`WIDER FACE`数据集上的最佳表现（**SOTA**）
***

## 数据文件部署
数据文件可以在[AIStudio下载](https://aistudio.baidu.com/aistudio/datasetdetail/75233)，`wider_val.txt`文件可在此处下载[[baidu](https://pan.baidu.com/s/14I8gNODfGkLKOy3BL3on5Q)/mrnh]
```
/home/aistudio
|-- Data
|   |-- widerface
|      |-- train
|          |-- images
|          |-- label.txt
|      |-- test
|          |-- images
|          |-- label.txt
|      |-- val
|          |-- images
|          |-- label.txt
|          |-- wider_val.txt
|-- RetinaFace
|   |-- widerface_evaluate(验证时需要自行配置)
|   |-- layers
|   |-- curve
|   |-- utils
|   |-- data
|   |-- models
|   |-- detect.py
|   |-- convert_to_onnx.py
|   |-- test.py
|   |-- train.py
|   |-- test_widerface.py
```

## 训练

**在使用中需要下载的所有资源文件，训练日志，ground_truth，训练模型以及预训练模型都可在[[baidu](https://pan.baidu.com/s/14I8gNODfGkLKOy3BL3on5Q)/mrnh]**

### train from scratch
```shell
python train.py --network resnet50
```
### train from checkpoint
```shell
python train.py --network resnet50 --resume_epoch 20 --resume_net ./weights/Resnet50_epoch_20.pdparams
```
需要注意到：

1.在训练之前，训练配置保存在`data/config.py`以及`train.py`的`args`参数

2.使用多卡训练`multi_gpu_train.py`时，其gpu选取采用spawn机制，你需要手动设置而不是从`data/config.py`获取

3.经过验证，本项目还支持轻量化模型`mobilenetV1X0.25`训练，同样你需要下载预训练模型

```shell
python train.py --network mobile0.25
```


## 评估
从[此处](https://github.com/wondervictor/WiderFace-Evaluation)下载评估代码，将其命名为`widerface_evaluate`如数据文件配置所示

生成预测文本文件
```shell
python test_widerface.py --trained_model ./torch2paddlemodel/Resnet50_Final.pdparams --network resnet50
```

进入`widerface_evaluate`，将下载的`ground_truth`放入`widerface_evaluate`文件夹下，编译并评估
```
cd /home/aistudio/Retinaface/widerface_evaluate
python setup.py build_ext --inplace
python evaluation.py
```
注意到如果使用[GitHub仓库](https://github.com/GuoQuanhao/RetinaFace-Paddle)在预测时出现报错需要修改`evaluation.py`的`89`行`boxes = np.array(list(map(lambda x: [float(a) for a in x.rstrip('\r\n').split(' ')], lines))).astype('float')`为`boxes = np.array(list(map(lambda x: [float(a) for a in x.rstrip('\r\n').split(' ')[:-1]], lines))).astype('float')`

**推荐直接使用[AIStudio仓库](https://aistudio.baidu.com/aistudio/projectdetail/2251677?contributionType=1)**

评估精度如下
| Style/R-50 | easy | medium | hard |
|:-|:-:|:-:|:-:|
| PaddlePaddle (same parameter with Mxnet) | **95.01 %** | **94.20%** | **90.02%** |
| PyTorch (same parameter with Mxnet) | 94.82 % | 93.84% | 89.60% |
| Mxnet | 94.86% | 93.87% | 88.33% |

![](https://ai-studio-static-online.cdn.bcebos.com/22ea493018bd45378480db42315bae950e829a02cd904a4b8b91b8e699211a04)

**在没有复杂后处理的情况下，PaddlePaddle训练结果均高于其余框架**

## TODO

**convert_to_onnx.py编写完成但暂未验证其实用性**

## 推理
```
python detect.py
```
你可以从`78`行指定图片路径，默认将读取`./curve/test.jpg`，推理效果如下

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/98750cebde9f470e83462ac2fcbcefcf63a7bf569b2e4946912485d02ac32f19" width="600"/><img src="https://ai-studio-static-online.cdn.bcebos.com/52dcf289e44341ae834be0bee1b9e92ca8d76fdb2b294840b2e8ad2c7ffc8317" width="600"/></center>

# **关于作者**
<img src="https://ai-studio-static-online.cdn.bcebos.com/cb9a1e29b78b43699f04bde668d4fc534aa68085ba324f3fbcb414f099b5a042" width="100"/>


| 姓名        |  郭权浩                           |
| --------     | -------- | 
| 学校        | 电子科技大学研2020级     | 
| 研究方向     | 计算机视觉             | 
| CSDN主页        | [Deep Hao的CSDN主页](https://blog.csdn.net/qq_39567427?spm=1000.2115.3001.5343) |
| GitHub主页        | [Deep Hao的GitHub主页](https://github.com/GuoQuanhao) |
如有错误，请及时留言纠正，非常蟹蟹！
后续会有更多论文复现系列推出，欢迎大家有问题留言交流学习，共同进步成长！
