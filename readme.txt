conda creat RWKVSR python=3.9.19
conda activite RWKVSR

#安装pytorch等
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install einops
pip install scipy
pip install thop
pip install fvcore
pip install tensorboardX