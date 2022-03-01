#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import cv2
import glob
import os

#模型生成数据的路径
test_paths = glob.glob(os.path.join('data/real_test/res/', 'rescolor/*.png'))
#标注数据路径
real_paths = glob.glob(os.path.join('data/real_test/fortest/', '*.png'))

# 打开一个文件
fo = open("statistic.txt", "w")

fo.write('data of model predicting:\n')

for label_path in test_paths:
	print(label_path)
	label = cv2.imread(label_path)
	label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
	label[label != 38] = 0
	label[label == 38] = 1


	mSum = label.sum()
	fo.write(f'{label_path}:{mSum}/408960={mSum/408960}\n')

fo.write('model of marking\n')
for label_path in real_paths:
	print(label_path)
	label = cv2.imread(label_path)
	# #转灰度图
	label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
	label[label != 38] = 0
	label[label == 38] = 1
	mSum = label.sum()
	fo.write(f'{label_path}:{mSum}/408960={mSum/408960}\n')

# 关闭打开的文件
fo.close()


