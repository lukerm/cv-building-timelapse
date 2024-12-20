#  Copyright (C) 2024 lukerm of www.zl-labs.tech
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


# Note: you need about 30GB volume before images
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

# <<< AMI END: CUDA pytorch GPU image training (ami-04173b0e1f2d9734c / ami-0dae2b0d39396a108) >>>

git clone git@github.com:lukerm/cv-building-timelapse
cd cv-building-timelapse/

#aws s3 sync s3://lukerm-ds-closed/cv-building-timelapse/data/experiments/256/ data/experiments/256/
aws s3 sync s3://lukerm-ds-closed/cv-building-timelapse/data/experiments/512/ data/experiments/512/

pip install -r requirements.txt

# find and replace instances of my username with 'ubuntu' in image_paths csv files
# note: not necessary for grouped image paths for new 512 dataset
export kp="R1"
cd data/experiments/256/
cp train/image_paths_${kp}.csv train/image_paths_${kp}_clean.csv
cp val/image_paths_${kp}.csv val/image_paths_${kp}_clean.csv
sed -i -e 's/luke/ubuntu/g' train/image_paths_${kp}_clean.csv
sed -i -e 's/luke/ubuntu/g' val/image_paths_${kp}_clean.csv
cd ../../../


# TODO: tmux setup
# TODO: configure experiment ID and hyperparameters
PYTHONPATH=. python3 src/train/train_unet.py


# Run this every so often in a tmux pane to save models and training logs to S3
cd models/experiments/512/
aws s3 sync . s3://lukerm-ds-closed/cv-building-timelapse/models/experiments/512/

# Run this code stub in a tmux pane to check for instance termination - it should shout WARNING when the instance is
# about to be terminated (TOKEN lasts for 6 hours)
TOKEN=`curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600"`
for i in {1..4320}; do
  sleep 5;
  echo '.';
  DATA=`curl -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-action --silent`;
  if [[ $DATA == *"action"* ]]; then echo "WARNING"; fi;
done
echo "LOOP END"
