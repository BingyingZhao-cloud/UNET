import os
import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
from config import TEST_IMAGE_DIR, IMAGE_SIZE, INPUT_CHANNELS, OUTPUT_CHANNELS, CHECKPOINT_DIR, MODEL_NAME
from modules import UNet

# 定义一个测试数据集类，仅用于加载图像
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir (str): 测试图像存放目录，文件后缀应为 '_sat.jpg'
            transform (callable, optional): 对测试图像进行变换
        """
        self.image_dir = image_dir
        self.image_list = sorted([f for f in os.listdir(image_dir) if f.endswith('_sat.jpg')])
        self.transform = transform if transform is not None else T.Compose([
            T.Resize(IMAGE_SIZE),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, image_name

def predict():
    # 设置设备为 GPU（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = UNet(in_channels=INPUT_CHANNELS, out_channels=OUTPUT_CHANNELS, bilinear=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, MODEL_NAME)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 加载测试集：如果 TEST_IMAGE_DIR 中的图像文件名后缀为 '_sat.jpg'
    test_dataset = TestDataset(image_dir=TEST_IMAGE_DIR)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    # 遍历测试集，进行推理并可视化结果
    for images, image_names in test_loader:
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
            # 使用 sigmoid 得到概率，二值化处理，0.5为阈值
            preds = (torch.sigmoid(outputs) > 0.5).float().cpu().squeeze(0)
            # 预测 mask 的尺寸为 [1, H, W]
            # 将图像与预测结果可视化
            orig_image = images.cpu().squeeze(0).permute(1, 2, 0).numpy()
            pred_mask = preds.squeeze(0).numpy() * 255

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(orig_image)
            plt.title(f'Input Image: {image_names[0]}')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(pred_mask, cmap='gray')
            plt.title('Predicted Road Mask')
            plt.axis('off')
            plt.show()

if __name__ == '__main__':
    predict()
