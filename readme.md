## 1.**[[albumentations]](https://github.com/albu/albumentations)**
    基于numpy,opencv,imgaug的图像增强库，取值精华组合而成
    集成了大量的transformations
    相比于其他库，速度更快
    支持images,masks,key points,bounding box 的增强
    在pytorch中使用非常方便
    在各种竞赛中被广泛使用，比如kaggle,topcoder,cvpr
    由kaggle大佬制作而成
    
## 2.**[catalyst](https://github.com/catalyst-team/catalyst)**
    用于pytorch的高级工具
    侧重于开发复用工具，快速实验和ideas的复用
    打破循环，使用catalyst
    
    此工具包含以下features
    - Universal train/inference loop通用的训练和推理循环
    - Configuration files for model/data hyperparameters.为模型和数据配置超参数
    - Reproducibility – all source code and environment variables will be saved.复用-所有的源码和参数都可以被保存
    - Callbacks – reusable train/inference pipeline parts.回调-可复用的训练和推理pipeline
    - Training stages support.
    - Easy customization.容易定制
    - PyTorch best practices (SWA, AdamW, 1Cycle, FP16 and more).
## 3.**[segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch)**
    基于pytorch的图像分割模型库
    包含以下features
    - 创建模型的高阶API,两行命令创建模型
    - 可用于二分类和多分类的四种模型架构
    - 30个encoder for 每个架构
    - 每一个encoder都包含预训练模型
```
使用
import segmentation_models_pytorch as smp
model = smp.Unet()
model = smp.Unet('resnet34', encoder_weights='imagenet')

from segmentation_models_pytorch.encoders import get_preprocessing_fn
preprocess_input = get_preprocessing_fn('resnet18', pretrained='imagenet')
```

## 4.**[pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt)**
    基于pytorh的深度学习模型原型和kaggle farming的快速构建库，
    包含一下features
    Easy model building using flexible encoder-decoder architecture.
    - Modules: CoordConv, SCSE, Hypercolumn, Depthwise separable convolution and more.
    - GPU-friendly test-time augmentation TTA for segmentation and classification
    - GPU-friendly inference on huge (5000x5000) images 对高分辨图很友好，将大图切成块进行处理
    - Every-day common routines (fix/restore random seed, filesystem utils, metrics)
    - Losses: BinaryFocalLoss, Focal, ReducedFocal, Lovasz, Jaccard and Dice losses, Wing Loss and more.
    - Extras for Catalyst library (Visualization of batch predictions, additional metrics)