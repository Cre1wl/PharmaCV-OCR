from ultralytics import YOLO

# Загрузка предобученной модели
model = YOLO("yolov8n.pt")

# Обучение
results = model.train(
    data="yolo_dataset/dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    device="cpu",  # используйте "cuda" если есть NVIDIA GPU
    workers=0,     # для Windows может потребоваться 0
    verbose=True
)

print("✅ Обучение завершено!")
print(f"Модель сохранена в: runs/detect/train/weights/best.pt")