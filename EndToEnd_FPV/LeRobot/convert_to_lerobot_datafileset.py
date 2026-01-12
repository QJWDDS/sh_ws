import pandas as pd
import numpy as np
import os
import shutil
from PIL import Image
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# --- 顶层 e2e 目录（包含多个时间戳子文件夹） ---
ROOT_DIR = Path(os.path.expanduser('~/sh_ws/document/default_data/e2eindependent'))

REPO_ID = "local/drone_interception_v7"
FPS = 30

def convert_data():

    # 输出数据目录
    output_dir = Path.home() / ".cache/huggingface/lerobot" / REPO_ID

    # 1. 清理旧数据集
    if output_dir.exists():
        print(f"Removing existing dataset at {output_dir}")
        shutil.rmtree(output_dir)

    # 2. 定义特征
    features = {
        "observation.images.camera": {
            "dtype": "image",
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"]
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["dummy_0","dummy_1","dummy_2","dummy_3","dummy_4","dummy_5"]
        },
        "action": {
            "dtype": "float32",
            "shape": (4,),
            "names": ["vx", "vy", "vz", "yaw_rate"]
        },
    }

    # 3. 创建数据集
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        robot_type="quadrotor",
        features=features,
        use_videos=True
    )

    # 4. 遍历所有时间戳文件夹
    for folder in sorted(ROOT_DIR.iterdir()):

        if not folder.is_dir():
            continue

        csv_path = folder / "data.csv"
        img_dir = folder / "images"

        if not csv_path.exists():
            print(f"Skipping {folder}, no CSV found")
            continue

        print(f"Processing episode from {folder}")

        df = pd.read_csv(csv_path)

        # 遍历该飞行的所有帧
        for _, row in df.iterrows():

            img_name = row["img_name"]
            img_path = img_dir / img_name
            image = Image.open(img_path)

            action = np.array([
                row['v_body_x'],
                row['v_body_y'],
                row['v_body_z'],
                row['yaw_rate_cmd']
            ], dtype=np.float32)

            state = np.zeros(6, dtype=np.float32)

            dataset.add_frame({
                "observation.images.camera": image,
                "observation.state": state,
                "action": action,
                "task": "Intercept red ball"
            })

        # ⭐⭐ 每个文件夹 = 一个 episode
        dataset.save_episode()

    # 5. 完成写入
    dataset.finalize()
    print(f"Dataset successfully created at {REPO_ID}")

if __name__ == "__main__":
    convert_data()
