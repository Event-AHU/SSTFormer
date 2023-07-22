<div align="center">
<img src="https://github.com/Event-AHU/SSTFormer/blob/main/figures/pokerevent2.png" width="600">
  
**A New Benchmark Dataset \& A SNN-ANN Framework for RGB-Event based Pattern Recognition**

------

<p align="center">
  • <a href="https://sites.google.com/view/viseventtrack/">Project</a> •
  <a href="https://arxiv.org/abs/2108.05015">arXiv</a> • 
  <a href="https://github.com/wangxiao5791509/RGB-DVS-SOT-Baselines">Baselines</a> •
  <a href="https://youtu.be/U4uUjci9Gjc">DemoVideo</a> • 
  <a href="https://youtu.be/vGwHI2d2AX0">Tutorial</a> •
</p>

</div>




# SCTFormer
Official PyTorch implementation of "SSTFormer: Bridging Spiking Neural Network and Memory Support Transformer for Frame-Event based Recognition", Xiao Wang, Zongzhen Wu, Yao Rong, Lin Zhu,
Bo Jiang, Jin Tang, Yonghong Tian. 



## Abstract 
Event camera-based pattern recognition is a newly arising research topic in recent years. Current researchers usually transform the event streams into images, graphs, or voxels, and adopt deep neural networks for event-based classification. Although good performance can be achieved on simple event recognition datasets, however, their results may be still limited due to the following two issues. Firstly, they adopt spatial sparse event streams for recognition only, which may fail to capture the color and detailed texture information well. Secondly, they adopt either Spiking Neural Networks (SNN) for energy-efficient recognition with suboptimal results, or Artificial Neural Networks (ANN) for energy-intensive, high-performance recognition. However, seldom of them consider achieving a balance between these two aspects. In this paper, we formally propose to recognize patterns by fusing RGB frames and event streams simultaneously and propose a new RGB frame-event recognition framework to address the aforementioned issues. The proposed method contains four main modules, i.e., memory support Transformer network for RGB frame encoding, spiking neural network for raw event stream encoding, multi-modal bottleneck fusion module for RGB- Event feature aggregation, and prediction head. Due to the scarce of RGB-Event based classification dataset, we also propose a large-scale PokerEvent dataset which contains 114 classes, and 27102 frame-event pairs recorded using a DVS346 event camera. Extensive experiments on two RGB-Event based classification datasets fully validated the effectiveness of our proposed framework. We hope this work will boost the development of pattern recognition by fusing RGB frames and event streams. 

<img src="https://github.com/Event-AHU/SSTFormer/blob/main/figures/firstIMG.jpg" width="800">



## ANN-SNN Framework for RGB-Event based Recognition 
<img src="https://github.com/Event-AHU/SSTFormer/blob/main/figures/framework_0505.jpg" width="800">  


## Download PokerEvent Dataset 
<img src="https://github.com/Event-AHU/SSTFormer/blob/main/figures/PokerEvent_samples.jpg" width="800"> 

```
* BaiduYun:

* DropBox: 
```


## Environment Setting 

conda create -n event  python=3.8 -y
conda activate event
pip3 install openmim
mim install mmcv-full
mim install mmdet  # optional
mim install mmpose  # optional
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip3 install -e .

## Tutorial on Training and Testing

```
```


```
```

## Experimental Results and Visualization 
<img src="https://github.com/Event-AHU/SSTFormer/blob/main/figures/SCNN_feature_map.jpg" width="800">




## Acknowledgement 
Our code is implemented based on ***. 



## Reference 
```
*** 
```






