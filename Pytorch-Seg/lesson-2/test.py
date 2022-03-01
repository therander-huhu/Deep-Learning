import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import cv2
import glob
import os

def getResDir(originDir):
    dirs = originDir.replace('\\', "/").split('/')
    newDir = ''
    for i in range(len(dirs)):
        if i < len(dirs) - 1:
            newDir += dirs[i] + '/'
        else:
            newDir += 'rescolor\\' + dirs[i]

    return newDir

test_paths = glob.glob(os.path.join('data/real_test/', 'res/*.png'))

i = 0
for label_path in test_paths:
	if i < 1:
		print(label_path)
		label = cv2.imread(label_path)

		# print(torch.tensor(label).shape)
		# print(torch.tensor(label).unique())
		# #转灰度图
		label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

		# print(torch.tensor(label).shape)
		# print(torch.tensor(label).unique())

		# 转色彩图
		label = cv2.cvtColor(label, cv2.COLOR_GRAY2BGR)

		# print(torch.tensor(label).unique())

		channel0 = label[:, :, 0]
		channel1 = label[:, :, 1]
		channel2 = label[:, :, 2]

		#green
		channel0[channel0 == 75] = 0 
		channel1[channel1 == 75] = 128 
		channel2[channel2 == 75] = 0 

		#red
		channel0[channel0 == 38] = 0
		channel1[channel1 == 38] = 0 
		channel2[channel2 == 38] = 128

		cv2.imwrite(getResDir(label_path), label, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

		# print(torch.tensor(label).unique())

		# label = cv2.cvtColor(label, cv2.COLOR_GRAY2BGR)
		# label = cv2.applyColorMap(label, cv2.COLORMAP_HOT)
		# label = cv2.cvtColor(label, cv2.COLOR_RGB2BGR)

		# print(torch.tensor(label).shape)
		# print(torch.tensor(label).unique())
		# cv2.imshow('show', label)
		# cv2.waitKey(0)
	# i = i + 1


