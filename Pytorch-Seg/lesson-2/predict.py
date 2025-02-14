import glob
import numpy as np
import torch
import os
import cv2
from model.unet_model import UNet

def getResDir(originDir):
    dirs = originDir.split('/')
    newDir = ''
    for i in range(len(dirs)):
        if i < len(dirs) - 1:
            newDir += dirs[i] + '/'
        else:
            newDir += 'res/' + dirs[i]

    return newDir.replace('jpg', "png")


if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    net = UNet(n_channels=1, n_classes=3)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('best_model.pth', map_location=device))
    # 测试模式
    net.eval()
    # 读取所有图片路径
    tests_path = glob.glob('data/real_test/*.jpg')
    # 遍历素有图片
    for test_path in tests_path:
        # 保存结果地址
        # save_res_path = test_path.split('.')[0] + '_res.png'
        print(test_path)
        save_res_path = getResDir(test_path)
        print(save_res_path)
        # 读取图片
        img = cv2.imread(test_path)
        # 转为灰度图
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 转为batch为1，通道为1，大小为512*512的数组
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        # 转为tensor
        img_tensor = torch.from_numpy(img)
        # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        # 预测
        pred = net(img_tensor)
        #print(pred.shape)
        #print(pred[0][0].shape)
        #print(pred[0][1].shape)
        
        
        # 提取结果
        pred = np.array(pred.data.cpu())
        pred = np.argmax(pred, axis = 1)[0]
        #print(pred.shape)
        #print(type(pred))
        #print(pred.min())
        #print(pred.mean())
        #print(pred.max())
        # 处理结果
        pred[pred < 0.5] = 0
        pred[pred >= 1.5] = 75
        pred[(pred >= 0.5) * (pred < 1.5)] = 38
        
        print(torch.tensor(pred).unique())
        # 保存图片

        cv2.imwrite(save_res_path, pred, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        # break;
