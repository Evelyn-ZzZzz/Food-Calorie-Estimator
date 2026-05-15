from ultralytics import YOLO
import os

os.environ['WANDB_DISABLED'] = 'true'  # 关闭wandb

# 加载预训练模型
model = YOLO('yolov8n.pt')  # 会自动下载权重

# 开始训练
results = model.train(
    data='/root/autodl-fs/Nutrition5k/data.yaml',
    epochs=100,
    batch=16,           # 4090D 16G显存，16够用
    imgsz=640,
    device=0,
    workers=8,
    patience=20,        # 20轮不提升则停止
    save=True,
    project='Nutrition5k_detection',
    name='yolov8n_Nutrition5k',
    exist_ok=True,
    pretrained=True,
    optimizer='auto',
    lr0=0.01,
    seed=42
)

print("训练完成！模型保存在:", model.ckpt_path)