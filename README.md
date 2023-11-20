# MetaViewer

The PyTorch implementation of the paper [MetaViewer: Towards A Unified Multi-View Representation](https://arxiv.org/pdf/2303.06329.pdf), *CVPR 2023*.


### Requirements
- python 3.8
- troch 1.13.1
- higher 0.2.1
- timm 0.6.12


### Clustering demo on the CALTECH101_20 dataset

Please download the [dataset & checkpoints(optional)](https://drive.google.com/drive/folders/1CJea-3DSZvebw_3INZRQ6vmFUMlAEQQV?usp=share_link) and replace the corresponding folders before training/testing.

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
@inproceedings{DBLP:conf/cvpr/0011SMXY23,
  author       = {Ren Wang and
                  Haoliang Sun and
                  Yuling Ma and
                  Xiaoming Xi and
                  Yilong Yin},
  title        = {MetaViewer: Towards {A} Unified Multi-View Representation},
  booktitle    = {{IEEE/CVF} Conference on Computer Vision and Pattern Recognition,
                  {CVPR} 2023, Vancouver, BC, Canada, June 17-24, 2023},
  pages        = {11590--11599},
  publisher    = {{IEEE}},
  year         = {2023},
  url          = {https://doi.org/10.1109/CVPR52729.2023.01115},
  doi          = {10.1109/CVPR52729.2023.01115},
  timestamp    = {Tue, 29 Aug 2023 15:44:40 +0200},
  biburl       = {https://dblp.org/rec/conf/cvpr/0011SMXY23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

### Acknowledgments

Work&Code is inspired by [AE2-Nets](https://github.com/willow617/AE2-Nets), [MFLVC](https://github.com/SubmissionsIn/MFLVC), [CPM_Nets](https://github.com/hanmenghan/CPM_Nets), [GBML](https://github.com/sungyubkim/GBML) ... 

### Contact
If you have any questions, feel free to contact Ren Wang (xxlifelover@gmail.com). 
