# MST
TensorFlow code for our ICCV 2019 paper "Multimodal Style Transfer via Graph Cuts" 
# Multimodal Style Transfer via Graph Cuts
This repository is for MST introduced in the following paper

[Yulun Zhang](http://yulunzhang.com/), [Chen Fang](https://scholar.google.com/citations?user=Vu1OqIsAAAAJ&hl=en), [Yilin Wang](http://yilinwang.org), [Zhaowen Wang](https://scholar.google.com/citations?user=lwlYARMAAAAJ&hl=en), [Zhe Lin](https://sites.google.com/site/zhelin625/), [Yun Fu](http://www1.ece.neu.edu/~yunfu/), and [Jimei Yang](https://eng.ucmerced.edu/people/jyang44), "Multimodal Style Transfer via Graph Cuts", ICCV 2019, [[arXiv]](https://arxiv.org/pdf/1904.04443.pdf) 

The code is tested on Ubuntu 16.04 environment (Python3.6, TensorFlow1.2, scikit-learn 0.19.1, PyMaxflow 1.2.11, CUDA9.0, cuDNN5.1) with Titan X/1080Ti/Xp GPUs. 

For K-Means clustering, we use sklearn package (scikit-learn 0.19.1).
For Graph-cuts, we use PyMaxflow package (PyMaxflow 1.2.12).
More enviroment information are available at src/[environments.md](https://github.com/yulunzhang/MST/blob/master/src/environments.md).

PyTorch code for MST is on the way.

## Test
```bash
# single content and style image pair
python test_MST.py --content='data/content/brad_pitt.jpg' --style='data/style/sketch.jpg'
# content and style image dir
python test_MST.py --content_dir='data/content' --style_dir='data/style'

```

## Results
The test data (29 content images, 61 style images) are available at [GoogleDrive](https://drive.google.com/file/d/1lD6wi-Uaw5tymYBtjwe5tmhxGrJathLs/view?usp=sharing) and [BaiduYun](https://pan.baidu.com/s/1sF7pGD19umLaogAWIF8R_w).
The visual results (29x61 stylized images) are available at [GoogleDrive](https://drive.google.com/file/d/1nl9KuB9q_hcfzSKY1PBxSfpKQmNEhI9p/view?usp=sharing) and [BaiduYun](https://pan.baidu.com/s/14cJgqLpaw_A8bhSmlAGW7w). The results are produced by using the model and parameters (e.g., alpha=1.0, K=3) of the paper. 

![Visual](/data/visual_v4.png)

## Citation
If you find the code/results helpful in your resarch or work, please cite the following papers.
```
@InProceedings{zhang2019multimodal,
  author = {Zhang, Yulun and Fang, Chen and Wang, Yilin and Wang, Zhaowen and Lin, Zhe and Fu, Yun and Yang, Jimei},
  title = {Multimodal Style Transfer via Graph Cuts},
  booktitle = {ICCV},
  year = {2019}
}

