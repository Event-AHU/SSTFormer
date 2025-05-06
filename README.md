<div align="center">
<img src="https://github.com/Event-AHU/SSTFormer/blob/main/figures/pokerevent2.png" width="600">
  
**A New Benchmark Dataset \& A SNN-ANN Framework for RGB-Event based Pattern Recognition**

------

<p align="center">
</p>
</div>




# SSTFormer
Official PyTorch implementation of **SSTFormer: Bridging Spiking Neural Network and Memory Support Transformer for Frame-Event based Recognition**, 
Xiao Wang, Yao Rong, Zongzhen Wu, Lin Zhu, Bo Jiang, Jin Tang, Yonghong Tian. [[arXiv](https://arxiv.org/abs/2308.04369)]



## News 

* [2025.05.05] SSTFormer is accepted by Journal **IEEE Transactions on Cognitive and Developmental Systems (TCDS)**  


## Abstract 
Event camera-based pattern recognition is a newly arising research topic in recent years. Current researchers usually transform the event streams into images, graphs, or voxels, and adopt deep neural networks for event-based classification. Although good performance can be achieved on simple event recognition datasets, however, their results may be still limited due to the following two issues. Firstly, they adopt spatial sparse event streams for recognition only, which may fail to capture the color and detailed texture information well. Secondly, they adopt either Spiking Neural Networks (SNN) for energy-efficient recognition with suboptimal results, or Artificial Neural Networks (ANN) for energy-intensive, high-performance recognition. However, seldom of them consider achieving a balance between these two aspects. In this paper, we formally propose to recognize patterns by fusing RGB frames and event streams simultaneously and propose a new RGB frame-event recognition framework to address the aforementioned issues. The proposed method contains four main modules, i.e., memory support Transformer network for RGB frame encoding, spiking neural network for raw event stream encoding, multi-modal bottleneck fusion module for RGB- Event feature aggregation, and prediction head. Due to the scarce of RGB-Event based classification dataset, we also propose a large-scale PokerEvent dataset which contains 114 classes, and 27102 frame-event pairs recorded using a DVS346 event camera. Extensive experiments on two RGB-Event based classification datasets fully validated the effectiveness of our proposed framework. We hope this work will boost the development of pattern recognition by fusing RGB frames and event streams. 

<img src="https://github.com/Event-AHU/SSTFormer/blob/main/figures/firstIMG.jpg" width="800">



## ANN-SNN Framework for RGB-Event based Recognition 
<img src="https://github.com/Event-AHU/SSTFormer/blob/main/figures/framework0722.jpg" width="800">  


## Download PokerEvent Dataset 
<img src="https://github.com/Event-AHU/SSTFormer/blob/main/figures/PokerEvent_samples.jpg" width="800"> 

```
* BaiduYun (178GB): 链接：https://pan.baidu.com/s/1vQnHZUqQ1o58SajvtE-uHw?pwd=AHUE 提取码：AHUE 

* DropBox (178GB): https://www.dropbox.com/scl/fo/w658kwhfi3qa8naul3eeb/h?rlkey=zjn4b69wa1e3mhid8p6hh8v75&dl=0
```


## Environment Setting 
```
conda create -n event  python=3.8 -y
conda activate event
pip3 install openmim
mim install mmcv-full
mim install mmdet  # optional
mim install mmpose  # optional
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip3 install -e .
```
### Some detailed settings about the path of the dataset 

SSTFormer：

The path of Rgb data, in /SSTFormer/configs/recognition/SSTFormer/SSTFormer.py, fill in the path of RGB data and the path of dataset labels in the following figure.

<img src="https://github.com/Event-AHU/SSTFormer/blob/main/figures/SSTFormer_RGB.jpg" width="300">

Path of event data，in SSTFormer/SSTFormer/mmaction/datasets/transforms/loading.py

<img src="https://github.com/Event-AHU/SSTFormer/blob/main/figures/SSTFormer_Event.jpg" width="400">

Please note that a portion of the RGB data path has been truncated here to ensure that the path you added is correct.




SpikingF_MST：

Path of event data，in SpikingF_MST/train.py

<img src="https://github.com/Event-AHU/SSTFormer/blob/main/figures/SpikingF_MST_Event.jpg" width="400">

Path of Rgb data, in SpikingF_MST/train.py

<img src="https://github.com/Event-AHU/SSTFormer/blob/main/figures/SpikingF_MST_RGB.jpg" width="400">

Path of dataset labels, in SpikingF_MST/train.py

<img src="https://github.com/Event-AHU/SSTFormer/blob/main/figures/SpikingF_MST_trainlabels.jpg" width="400">
<img src="https://github.com/Event-AHU/SSTFormer/blob/main/figures/SpikingF_MST_testlabels.jpg" width="400">


## Train & Test
```
SSTFormer:
train_SSTFormer.sh
test_SSTFormer.sh


SpikingF-MST:
sh train_SpikingFMST.sh
```


## Experimental Results and Visualization 
<img src="https://github.com/Event-AHU/SSTFormer/blob/main/figures/SCNN_feature_map.jpg" width="800">




## Acknowledgement 
Our code is implemented based on 
<a href="https://github.com/open-mmlab/mmaction2">MMAction2</a>, 
<a href="https://github.com/zhouchenlin2096/Spikingformer">Spikingformer</a>.



## Reference 
```
@misc{wang2025sstformer,
      title={SSTFormer: Bridging Spiking Neural Network and Memory Support Transformer for Frame-Event based Recognition}, 
      author={Xiao Wang and Yao Rong and Zongzhen Wu and Lin Zhu and Bo Jiang and Jin Tang and Yonghong Tian},
      year={2025},
      eprint={2308.04369},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2308.04369}, 
}
```






