# Youku-mPLUG Datasets
Youku-mPLUG: A 10 Million Large-scale Chinese Video-Language Pre-training Dataset and Benchmarks

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

We also provide three Chinese video-language benchmark datasets including Video-Text Retrieval, Video Category Prediction, and Video Captioning.
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