# nmODE

**0. Main Environments**
- python 3.8
- [pytorch 1.8.0](https://download.pytorch.org/whl/cu111/torch-1.8.0%2Bcu111-cp38-cp38-win_amd64.whl)
- [torchvision 0.9.0](https://download.pytorch.org/whl/cu111/torchvision-0.9.0%2Bcu111-cp38-cp38-linux_x86_64.whl)

**1. Prepare the dataset.**

- The ISIC17,ISIC18 and PH2 datasets, divided into a 7:3 ratio, can be found here {[Baidu](https://pan.baidu.com/s/1g5lzfPIfIxf-lGRJX3pBzw?pwd=5m7h) 
- 
- After downloading the datasets, you are supposed to put them into './data/isic17/' , './data/isic18/' and './data/ph2/' , and the file format reference is as follows. (take the ISIC17 dataset as an example.)

- './data/isic17/'
  - train
    - images
      - .png
    - masks
      - .png
  - val
    - images
      - .png
    - masks
      - .png

**2. Code Structure.**
- Modify the parameters regarding the experiment in the file config_setting.py
- Put the model file in the models folder, to change the decoder to nmODE please refer to the provided UNet example

**3. Train the Model.**

```
python train.py
```

**4. Obtain the outputs.**
- After trianing, you could obtain the outputs in './results/'
