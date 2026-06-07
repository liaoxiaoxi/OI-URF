# OI-URF
OI-URF:Unified Multi-Modal Registration and Fusion via Operator-Level Adaptation and Interaction-Guided Alignment

By Xiaoxi Liao, Kai Wang*, Wei Jiang, Hongke Zhang

<img width="1582" height="445" alt="方法图 - 副本" src="https://github.com/user-attachments/assets/69c81cc1-6ef1-4fbd-aa46-95ef2bac382d" />

<img width="760" height="507" alt="PROB - 副本" src="https://github.com/user-attachments/assets/4c25bbc2-104a-43ab-b0e3-402166f1a4cf" />

<img width="1565" height="372" alt="MGTI - 副本" src="https://github.com/user-attachments/assets/7df1e3b7-0437-4b20-9e66-5cbef2a0e906" />

## Requirements

- Python 3.8
- torch 2.1.2+cu118
- torchvision 0.16.2+cu118
- opencv-python 4.13.0.92
- kornia 0.6.3
- numpy 1.26.4
- Pillow 10.3.0
- tqdm 4.64.1
- matplotlib 3.8.2

You can install the main dependencies with:

```bash
pip install torch==2.1.2 torchvision==0.16.2 opencv-python==4.13.0.92 kornia==0.6.3 numpy==1.26.4 pillow==10.3.0 tqdm==4.64.1 matplotlib==3.8.2
```
## To test
```bash
python testcolor.py
```
## To train
```bash
python traincolor.py
```

### Dataset Download

The **M3FD** and **RoadScene** datasets used in this work are publicly available. Their download links can be found below:
- [Google Drive](https://drive.google.com/drive/folders/1H-oO7bgRuVFYDcMGvxstT1nmy0WF_Y_6)

### Harvard Medical Image Fusion Datasets

The PET-MRI, SPECT-MRI, and CT-MRI datasets used in this work are derived from the Harvard Whole Brain Atlas. The original data source can be accessed from:

- [Harvard Whole Brain Atlas](https://www.med.harvard.edu/AANLIB/home.html)


