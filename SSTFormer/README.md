<div align="center">
<img src="https://github.com/Event-AHU/SSTFormer/blob/main/figures/pokerevent2.png" width="600">
  
**A New Benchmark Dataset \& A SNN-ANN Framework for RGB-Event based Pattern Recognition**

------

<p align="center">
</p>
</div>




# SSTFormer
Official PyTorch implementation of **SSTFormer: Bridging Spiking Neural Network and Memory Support Transformer for Frame-Event based Recognition**, 
Xiao Wang, Zongzhen Wu, Yao Rong, Lin Zhu, Bo Jiang, Jin Tang, Yonghong Tian. [[arXiv](https://arxiv.org/abs/2308.04369)]



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
## Train & Test
```
SSTFormer:
python tools/train.py configs/recognition/SSTFormer/SSTFormer.py --work-dir work_dirs/SSTFormer  --seed 0 --deterministic
python tools/test.py configs/recognition/SSTFormer/SSTFormer.py  work_dirs/SSTFormer/checkpoint --eval top_k_accuracy 

SpikingF-MST:
cd SpikingF_MST
sh train.sh
```


## Experimental Results and Visualization 
<img src="https://github.com/Event-AHU/SSTFormer/blob/main/figures/SCNN_feature_map.jpg" width="800">




## Acknowledgement 
Our code is implemented based on 
<a href="https://github.com/open-mmlab/mmaction2">MMAction2</a>, 
<a href="https://github.com/zhouchenlin2096/Spikingformer">Spikingformer</a>.



## Reference 
```
@misc{wang2023sstformer,
      title={SSTFormer: Bridging Spiking Neural Network and Memory Support Transformer for Frame-Event based Recognition}, 
      author={Xiao Wang and Zongzhen Wu and Yao Rong and Lin Zhu and Bo Jiang and Jin Tang and Yonghong Tian},
      year={2023},
      eprint={2308.04369},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```






