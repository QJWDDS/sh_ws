import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import os
import numpy as np
from torchvision import transforms

class DroneDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): 包含多个时间戳文件夹的根目录 
                            (例如: '~/sh_ws/dataset/raw_data')
            transform (callable, optional): 图像预处理函数
        """
        self.root_dir = root_dir
        self.transform = transform
        self.all_data = []

        # 1. 遍历根目录下的所有子文件夹
        print(f"Scanning data in: {root_dir} ...")
        valid_folders = 0
        
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Directory not found: {root_dir}")

        for folder_name in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder_name)
            
            # 检查是不是目录，且包含 data.csv
            csv_path = os.path.join(folder_path, 'data.csv')
            if os.path.isdir(folder_path) and os.path.exists(csv_path):
                try:
                    # 读取该文件夹的 CSV
                    df = pd.read_csv(csv_path)
                    
                    # 关键步骤：在 DataFrame 中增加一列，记录它属于哪个文件夹
                    # 这样我们在 __getitem__ 时才能找到正确的图片路径
                    df['folder_path'] = folder_path
                    
                    self.all_data.append(df)
                    valid_folders += 1
                    print(f"  -> Loaded {len(df)} samples from: {folder_name}")
                except Exception as e:
                    print(f"  [WARN] Failed to load {folder_name}: {e}")

        if valid_folders == 0:
            raise RuntimeError(f"No valid data folders found in {root_dir}")

        # 2. 将所有 CSV 数据合并成一个巨大的 DataFrame
        self.combined_frame = pd.concat(self.all_data, ignore_index=True)
        print(f"Total dataset size: {len(self.combined_frame)} samples from {valid_folders} folders.")

    def __len__(self):
        return len(self.combined_frame)

    def __getitem__(self, idx):
        # 获取当前行数据
        row = self.combined_frame.iloc[idx]
        
        # 1. 解析路径
        img_name = row['img_name']
        folder_path = row['folder_path'] # 获取该图片所属的文件夹路径
        
        # 拼凑完整的图片路径: folder/images/xxx.jpg
        img_path = os.path.join(folder_path, 'images', img_name)
        
        # 2. 读取图片
        image = cv2.imread(img_path)
        if image is None:
            # 如果某张图坏了，打印警告并随机返回另一张图（防止训练中断）
            # print(f"Warning: Corrupt image {img_path}")
            return self.__getitem__(np.random.randint(0, len(self)))
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 3. 预处理
        if self.transform:
            image = self.transform(image)
        
        # 4. 获取标签 (v_body_x, v_body_y, v_body_z, yaw_setpoint)
        # 注意：现在 'folder_path' 是最后一列，所以我们需要按列名取值，或者取前4个数值列
        # 假设 CSV 结构是: img_name, v_x, v_y, v_z, yaw, folder_path
        # 我们取中间的数值部分
        label_cols = ['v_body_x', 'v_body_y', 'v_body_z', 'yaw_rate_cmd']
        label_values = row[label_cols].values.astype(np.float32)
        
        label = torch.from_numpy(label_values)
        
        return image, label