import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from config import TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, BINARIZATION_THRESHOLD, IMAGE_SIZE
from transforms import JointCompose, Resize, ToTensor

# 定义 joint_transform：同时对 image 和 mask 进行 Resize 和 ToTensor 转换
joint_transform = JointCompose([
    Resize(IMAGE_SIZE),
    ToTensor(),
])

class RoadSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=joint_transform, threshold=BINARIZATION_THRESHOLD):
        """
        Args:
            image_dir (str): 卫星图像目录，文件后缀应为 '_sat.jpg'
            mask_dir  (str): 标签目录，标签文件后缀应为 '_mask.png'
            transform (callable): 同时处理 image 和 mask 的变换
            threshold (int): 二值化阈值
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        # 筛选出以 '_sat.jpg' 结尾的文件列表
        self.image_list = sorted([f for f in os.listdir(image_dir) if f.endswith('_sat.jpg')])
        self.transform = transform
        self.threshold = threshold
        self.has_mask = mask_dir is not None
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # 获取当前图像文件名
        image_name = self.image_list[idx]
        # 根据文件名规则，构造对应的标签文件名
        mask_name = image_name.replace('_sat.jpg', '_mask.png')

        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        # 检查文件是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

        # 读取图像与标签
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 灰度

        # 二值化处理：将大于等于 threshold 的像素设为 1，其余为 0
        mask_np = np.array(mask)
        mask_bin = (mask_np >= self.threshold).astype(np.uint8)

        # 应用 joint_transform 同步对 image 和 mask 进行变换（如 Resize 和 ToTensor）
        image, mask_bin = self.transform(image, mask_bin)

        return image, mask_bin

# 测试数据集加载
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    # 使用 config 中的路径加载数据
    dataset = RoadSegmentationDataset(
        image_dir=TRAIN_IMAGE_DIR,
        mask_dir=TRAIN_MASK_DIR
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)
    imgs, masks = next(iter(loader))
    print('images:', imgs.shape)  # 例如：[2, 3, H, W]
    print('masks :', masks.shape)  # 例如：[2, 1, H, W]
