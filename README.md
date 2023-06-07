# Youku-mPLUG 10M Chinese Large-Scale Video Text Dataset
Youku-mPLUG: A 10 Million Large-scale Chinese Video-Language Pre-training Dataset and Benchmarks
<p align="center">
<img src="assets/youku_mplug_logo.png" alt="examples for youku-mplug"/>
</p>

## What is Youku-mPLUG?
We release the public largest Chinese high-quality video-language dataset (10 million) named **Youku-mPLUG**, which is collected 
from a well-known Chinese video-sharing website, named Youku, with strict criteria of safety, diversity, and quality.

<p align="center">
<img src="assets/pretrain_data.jpg" alt="examples for youku-mplug"/>
</p>
<p align="center">
<img src="assets/examples.png" alt="examples for youku-mplug"/>
</p>
<p align="center">
<font size=2 color="gray">Examples of video clips and titles in the proposed Youku-mPLUG dataset.</font>
</p>

We provide 3 different downstream multimodal video benchmark datasets to measure the capabilities of pre-trained models. The 3 different tasks include:
- Video Category Prediction：Given a video and its corresponding title, predict the category of the video.
- Video-Text Retrieval：In the presence of some videos and some texts, use video for text retrieval and text for video retrieval.
- Video Captioning：In the presence of a video, describe the content of the video.
<p align="center">
<img src="assets/downstream_data.jpg" alt="examples for youku-mplug downstream dataset"/>
</p>


## Data statistics
The dataset contains 10 million videos in total, which are of high quality and distributed in 20 super categories can 45 categories.

<p align="center">
<img src="assets/statics.jpg" alt="statistics" width="60%"/>
</p>
<p align="center">
<font size=2 color="gray">The distribution of categories in Youku-mPLUG dataset.</font>
</p>


## Download
You can download all the videos through this [link](https://modelscope.cn/datasets/modelscope/Youku-AliceMind/summary) 

## Citing Youku-mPLUG

If you find this dataset useful for your research, please consider citing our paper.

```bibtex
@misc{xu2023youku_mplug,
    title={Youku-mPLUG: A 10 Million Large-scale Chinese Video-Language Dataset for Pre-training and Benchmarks},
    author={Haiyang Xu, Qinghao Ye, Xuan Wu, Ming Yan, Yuan Miao, Jiabo Ye, Guohai Xu, Anwen Hu, Yaya Shi, Chenliang Li, Qi Qian, Que Maofei, Ji Zhang, Xiao Zeng, Fei Huang},
    year={2023},
    eprint={2306.xxxxx},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```