## ROS-SAM: High-Quality Interactive Segmentation for Remote Sensing Moving Object (ROS-SAM)

>[ROS-SAM: High-Quality Interactive Segmentation for Remote Sensing Moving Object](https://arxiv.org/abs/2503.12006)     
>CVPR 2025

We propose ROS-SAM to extend SAM for high-quality remote moving object segmentation. Refer to our paper for more details.

### Quick start

```git clone https://github.com/ShanZard/ROS-SAM.git
   cd  ROS-SAM
   conda create -f environment.yml```

### Prepare your dataset 
Preparing this dataset is a complex process.  
>(1). Download the [SAT-MTB](https://ieeexplore.ieee.org/document/10130311) dataset at this [link](http://www.csu.cas.cn/gb/kybm/sjlyzx/gcxx_sjj/sjj_wxxl/202211/t20221121_6551405.html).  
>(2). Prepare the dataset format according to the setting of ```ROSSAM_Dataset```. The most important thing is that you need to prepare a ```boxes.npy [N,4] [xmin,ymin,xmax,ymax]``` and ```onehotmask.npy [N,H,W]``` for each image.  
>(3). Follow the example of the ```data``` folder and put in the corresponding file. ```data\train``` have a example.

### Model checkpoints  

Our model is based on SAM and HQ-SAM, and all checkpoints can be downloaded from [HQ-SAM](https://github.com/SysCV/sam-hq).  
With the L version, for example, you need to do the following.   
```mkdir pretrained_checkpoint```  
```download sam_vit_l_0b3195.pth\sam_vit_l_maskdecoder.pth\sam_hq_vit_l.pth```

### Training and inference

```cd train ```   
``` accelerate launch train.py ```    
``` CUDA_VISIBLE_DEVICES=0 accelerate launch inference.py ``` 

Notably, inference process only work in single GPU.

### Citation

If you find ROS-SAM useful in your research or refer to the provided baseline results, please star ⭐ this repository and consider citing
```
@inproceedings{rossam,
    title={ROS-SAM: High-Quality Interactive Segmentation for Remote Sensing Moving Object},
    author={Shan, Zhe and Liu, Yang and Zhou, Lei and Yan, Cheng and Wang, Heng and Xie, Xia},
    booktitle={CVPR},
    year={2025}
}  
```
### Acknowledgments

- Thanks [SAM](https://github.com/facebookresearch/segment-anything), [HQ-SAM](https://github.com/SysCV/sam-hq), [SAM-LoRA](https://github.com/JamesQFreeman/Sam_LoRA/tree/main) and [SAT-MTB](http://www.csu.cas.cn/gb/kybm/sjlyzx/gcxx_sjj/sjj_wxxl/202211/t20221121_6551405.html) for their public code and released dataset and models.
