# Landmarking



<p align="center">
  <img src="./assets/framework.png" alt="drawing", width="700"/>
</p>

## Installation
The code is tested with ``python=3.8.8``, ``torch=2.0.0``, and ``torchvision=0.15.0``.
```
git clone https://github.com/MarwaAbderrahim/Landmarking.git
cd Landmarking-main
```
Create a new conda environment and install required packages accordingly.
```
conda create -n det3d python=3.8.8
conda activate det3d
pip3 install -r requirements.txt
```

## Data for training
First, the users need to prepare medical images and their corresponding landmark annotations. The ``assets`` folder contains an example image (``case_001.nii.gz``) and landmark annotation file (``case_001.csv``). Then, generate landmark masks (e.g, ``case_001_landmark_mask.nii.gz``) with the folowing code :
```
cd Landmarking-main/detection3d/scripts/
python gen_landmark_mask.py
```

Finally, prepare dataset splitting files for training (``train.csv``) and testing (``test.csv``). 

## Training
Run the following code for training.
The user may need to modify training settings in ``./config/lmk_train_config.py``. By default, the model will be saved in ``./saves/weights``.
```
cd detection3d
python lmk_det_train.py
```

## Evaluation
Run the following code to evaluate a trained model on a single GPU.
```
python lmk_det_infer.py -i "image-path/image-name.nii.gz" " -m "model path" -o "output folder path "
``` 

## Citation
```bibtex
@article{liu2021skullengine,
  title={SkullEngine: A Multi-stage CNN Framework for Collaborative CBCT Image Segmentation and Landmark Detection},
  author={Liu, Qin and Deng, Han and Lian, Chunfeng and Chen, Xiaoyang and Xiao, Deqiang and Ma, Lei and Chen, Xu and Kuang, Tianshu and Gateno, Jaime and Yap, Pew-Thian and others},
  journal={arXiv preprint arXiv:2110.03828},
  year={2021}
}
```
