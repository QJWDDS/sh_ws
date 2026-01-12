import pandas as pd
import numpy as np
import os
import shutil
from PIL import Image
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# --- 配置 ---
RAW_DATA_DIR = Path(os.path.expanduser('~/sh_ws/document/default_data/e2e/20260105_175038')) 
REPO_ID = "local/drone_interception_v1"
FPS = 30

def convert_data():
    csv_path = RAW_DATA_DIR / 'data.csv'
    img_dir = RAW_DATA_DIR / 'images'
    
    if not csv_path.exists():
        print(f"Error: CSV not found at {csv_path}")
        return

    # 1. 安全清理旧数据
    output_dir = Path.home() / ".cache/huggingface/lerobot" / REPO_ID
    if output_dir.exists():
        print(f"Removing existing dataset at {output_dir}")
        shutil.rmtree(output_dir)

    # 2. 读取 CSV
    df = pd.read_csv(csv_path)
    num_frames = len(df)
    
    # 3. 定义特征 (修改处：State shape 改为 (6,), names 对应增加)
    features = {
        "observation.images.camera": {"dtype": "image", "shape": (3, 480, 640), "names": ["channels", "height", "width"]},
        "observation.state": {
            "dtype": "float32", 
            "shape": (6,), 
            "names": ["dummy_0", "dummy_1", "dummy_2", "dummy_3", "dummy_4", "dummy_5"]
        }, 
        "action": {"dtype": "float32", "shape": (4,), "names": ["vx", "vy", "vz", "yaw_rate"]},
    }

    # 4. 创建数据集
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        robot_type="quadrotor",
        features=features,
        use_videos=True 
    )

    print(f"Processing {num_frames} frames from {RAW_DATA_DIR}...")
    
    for index, row in df.iterrows():
        img_name = row['img_name']
        img_path = img_dir / img_name
        image = Image.open(img_path)
        
        action = np.array([
            row['v_body_x'], 
            row['v_body_y'], 
            row['v_body_z'], 
            row['yaw_rate_cmd']
        ], dtype=np.float32)
        
        # 修改处：State 设置为 6 维全 0 向量
        state = np.zeros(6, dtype=np.float32)

        dataset.add_frame({
            "observation.images.camera": image,
            "observation.state": state,
            "action": action,
            "task": "Intercept red ball",
        })

    # 5. 保存并结束
    dataset.save_episode() 
    dataset.finalize()
    print(f"Dataset successfully created at {REPO_ID}")

if __name__ == "__main__":
    convert_data()