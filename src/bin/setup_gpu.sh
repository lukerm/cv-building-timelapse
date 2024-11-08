# Note: you need about XXX GB volume before images
# Use a p3.2xlarge (~$1 per hour)
# attach instance role: cv-building-timelapse-training (for reading from & saving to S3)

# <<< AMI START: Ubuntu 22.04 Server >>>

sudo apt update
sudo apt -y upgrade

sudo apt install -y awscli

# following: https://gist.github.com/denguir/b21aa66ae7fb1089655dd9de8351a202
# nvidia drivers
sudo apt autoremove nvidia* --purge
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers devices
sudo apt install -y nvidia-driver-525  # note: exact number (525) can change (check recommended version from `ubuntu-drivers devices`)
# Note: 525.xx is compatible with CUDA 12.0
sudo reboot
nvidia-smi

# python + dependencies
sudo apt install -y python3 python3-pip
pip install torch torchvision

# <<< AMI END: CUDA pytorch GPU image training (ami-04173b0e1f2d9734c) >>>

git clone git@github.com:lukerm/cv-building-timelapse
cd cv-building-timelapse/

aws s3 sync s3://lukerm-ds-closed/cv-building-timelapse/data/experiments/256/ data/experiments/256/

pip install -r requirements.txt

# find and replace instances of my username with 'ubuntu' in image_paths csv files
export kp="R1"
cd data/experiments/256/
sed -i -e 's/luke/ubuntu/g' train/image_paths_$kp_clean.csv
sed -i -e 's/luke/ubuntu/g' val/image_paths_$kp_clean.csv
cd ../../../


# TODO: tmux setup
# TODO: configure experiment ID and hyperparameters
PYTHONPATH=. python3 src/train/train_unet.py
