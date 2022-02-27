import numpy as np
import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random

class ISBI_Loader(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.jpg'))

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip
        
    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('image', 'label').replace('jpg', 'png')
        # 读取训练图片和标签图片
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        # print((type(image)))
        # print(image.shape)
        # 将数据转为单通道的图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # print((type(image)))
        # print(image.shape)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        # print(type(label))
        # print(label.shape)
        # cv2.imshow("fuck", label)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        image = image.reshape(1, image.shape[0], image.shape[1])
        # print((type(image)))
        # print(image.shape)
        label = label.reshape(1, label.shape[0], label.shape[1])
        # print(type(label))
        # print(label.shape)
        # 处理标签，将像素值为255的改为1
        # print(image.max())
        # print(label.max())

        label[label == 38] = 1
        label[label == 75] = 2
        # print(type(label))
        # print(label.shape)
        # np.where(label == 38, 1, label)
        # np.where(label == 75, -1, label)
        # if label.max() > 1:
        #     label = label / 255
        # 随机进行数据增强，为2时不做处理
        # flipCode = random.choice([-1, 0, 1, 2])
        # if flipCode != 2:
            # image = self.augment(image, flipCode)
            # label = self.augment(label, flipCode)
        
        # width = image.shape[0]
        # heigh = image.shape
        # print(type(image))
        # print(image.shape)
        # print(label.shape)
        width = 848
        if(image.shape[2] > width):
            image = image[:, :, 0:width]
        if(label.shape[2] > width):
            label = label[:, :, 0:width]
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)

    
if __name__ == "__main__":
    isbi_dataset = ISBI_Loader("../data/real_train/")
    print("数据个数：", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=1,
                                               shuffle=True)
    torch.set_printoptions(profile="full")
    for image, label in train_loader:
        print(label)
        # print(type(label))
        break