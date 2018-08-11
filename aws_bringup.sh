#!/bin/bash

pip install tqdm
pip3 install opencv-python
mkdir data & cd data
wget "https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/advanced_deep_learning/data_road.zip"
unzip data_road.zip
rm data_road.zip
