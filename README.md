

# RWKVSR
**This repository is implementation of the " Receptance Weighted Key-Value Network for Hyperspectral Image Super-Resolution"by PyTorch.**

## DataSet
**Three public datasets, i.e., [CAVE](https://cave.cs.columbia.edu/repository/Multispectral), [Harvard](https://vision.seas.harvard.edu/hyperspec/explore.html)and  [Pavia Center](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Pavia_University_scene) are employed to verify the effectiveness of the proposed RWKVSR. Since there are too few images in these datasets for deep learning algorithm, we augment the training data. With respect to the specific details, please see the implementation details section.**

**Moreover, we also provide the code about data pre-processing in folder data pre-processing. The folder contains three parts, including training set augment, test set pre-processing, and band mean for all training set.**

## Requirement
`conda creat RWKVSR python=3.9.19`

`conda activite RWKVSR`

#安装pytorch等：

`pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113`

`pip install einops`

`pip install scipy`

`pip install thop`

`pip install fvcore`

`pip install tensorboardX`
 ## Training
 **The ADAM optimizer with beta_1 = 0.9, beta _2 = 0.999 is employed to train our network. The learning rate is initialized as 11^-4 for all layers, which decreases by a half at every 35 epochs.
You can train or test directly from the command line as such:**

`python train.py --cuda --datasetName CAVE --upscale_factor 2`

`python test.py --cuda --model_name checkpoint/model_2_epoch_XXX.pt`

## result
**To qualitatively measure the proposed RWKVSR, three evaluation methods are employed to verify the effectiveness of the algorithm, including Peak Signal-to-Noise Ratio (PSNR), Structural SIMilarity (SSIM), and Spectral Angle Mapper (SAM).**

|Scale|CAVE|Harvard|Pavia Centre|
|-|-|-|-|
|x2|46.003/0.9906/2.877|49.077/0.9913/2.238|36.090/0.9525/4.20|
|x3|41.909/0.9778/3.020|43.929/0.9739/3.469|29.160/0.8249 /6.50|
|x4|39.673/0.9671/3.516|42.416/0.9687/3.151|29.880/0.8173/6.60|

## Citation
If you find this work helpful, please consider citing our paper:

```bibtex
@ARTICLE{11222729,
  author={Yang, Xiaofei and Li, Sihuan and Cao, Weijia and Tang, Dong and Ban, Yifang and Zhou, Yicong},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={RWKVSR: Receptance Weighted Key-Value Network for Hyperspectral Image Super-Resolution}, 
  year={2025},
  doi={10.1109/TCSVT.2025.3626779}
}

