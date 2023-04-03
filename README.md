# MetaViewer

The PyTorch implementation of the paper [MetaViewer: Towards A Unified Multi-View Representation](https://arxiv.org/pdf/2303.06329.pdf), *CVPR 2023*.


### Requirements
- python 3.8
- troch 1.13.1
- higher 0.2.1
- timm 0.6.12


### Clustering demo on the CALTECH101_20 dataset

Please download the [dataset & checkpoints(option)](https://drive.google.com/drive/folders/1CJea-3DSZvebw_3INZRQ6vmFUMlAEQQV?usp=share_link) and replace the corresponding folders before training/testing.

Train:  
```python
python main.py --model MetaViewer --channels -1 500 500 2000 256 --meta_channels -1 32
```

Test:
```python
python main.py --model MetaViewer --channels -1 500 500 2000 256 --meta_channels -1 32 --testing
```


### Citation (waiting for updates)
```
@article{wang2023metaviewer,
  title={MetaViewer: Towards A Unified Multi-View Representation},
  author={Wang, Ren and Sun, Haoliang and Ma, Yuling and Xi, Xiaoming and Yin, Yilong},
  journal={arXiv preprint arXiv:2303.06329},
  year={2023}
}
```

### Acknowledgments

Work&Code is inspired by [AE2-Nets](https://github.com/willow617/AE2-Nets), [MFLVC](https://github.com/SubmissionsIn/MFLVC), [CPM_Nets](https://github.com/hanmenghan/CPM_Nets), [GBML](https://github.com/sungyubkim/GBML) ... 

### Contact
If you have any questions, feel free to contact Ren Wang (xxlifelover@gmail.com). 
