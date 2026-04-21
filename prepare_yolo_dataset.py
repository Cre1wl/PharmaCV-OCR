import json
import shutil
from pathlib import Path

# ===== ПУТИ =====
ANNOTATIONS_FILE = Path("D:/PyCharm 2025.3/PythonProject/ForLabelStudio/pharma_dataset/Annotations.json")
IMAGES_DIR = Path("D:/PyCharm 2025.3/PythonProject/ForLabelStudio/pharma_dataset/images")
OUTPUT_DIR = Path("D:/PyCharm 2025.3/PythonProject/ForLabelStudio/yolo_dataset")

# Соответствие классов
CLASS_MAPPING = {
    "drug_name": 0,
    "expiry_date": 1,
    "batch_number": 2,
    "barcode": 3,
    "package": 4
}

# Создаём структуру YOLO
for split in ["train", "val"]:
    (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

# Загружаем аннотации
with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
    tasks = json.load(f)

print(f"📋 Загружено {len(tasks)} записей из JSON\n")

# Создаём словарь соответствия: короткое имя -> полное имя из JSON
# Например: "amoxiclav_1.jpg" -> "f522ea4b-amoxiclav_1.jpg"
file_mapping = {}
for task in tasks:
    long_name = task["file_upload"]
    # Извлекаем короткое имя (убираем префикс)
    parts = long_name.split("-", 1)
    if len(parts) > 1:
        short_name = parts[1]  # "amoxiclav_1.jpg"
        file_mapping[short_name] = long_name
        file_mapping[long_name] = long_name  # оставляем и длинное имя

print("📝 Сопоставление имён файлов:")
for short, long in list(file_mapping.items())[:5]:
    print(f"  {short} -> {long}")

# Собираем все файлы
all_files = []
missing_files = []

for task in tasks:
    json_filename = task["file_upload"]

    # Пробуем найти файл
    img_path = None

    # Вариант 1: ищем с оригинальным именем
    test_path = IMAGES_DIR / json_filename
    if test_path.exists():
        img_path = test_path
    else:
        # Вариант 2: ищем без префикса
        parts = json_filename.split("-", 1)
        if len(parts) > 1:
            short_name = parts[1]
            test_path2 = IMAGES_DIR / short_name
            if test_path2.exists():
                img_path = test_path2
                print(f"✅ Найден файл: {json_filename} -> {short_name}")

    if img_path:
        all_files.append((task, img_path, json_filename))
    else:
        missing_files.append(json_filename)
        print(f"❌ Не найден: {json_filename}")

print(f"\n📊 Найдено {len(all_files)} из {len(tasks)} изображений")

if len(all_files) == 0:
    print("\n❌ Нет ни одного изображения. Проверьте пути.")
    print(f"Содержимое папки {IMAGES_DIR}:")
    for f in list(IMAGES_DIR.glob("*.jpg"))[:10]:
        print(f"  - {f.name}")
    exit(1)

# Разделение на train/val (80/20)
split_idx = int(len(all_files) * 0.8)
train_files = all_files[:split_idx]
val_files = all_files[split_idx:]

print(f"📊 Train: {len(train_files)} изображений")
print(f"📊 Val: {len(val_files)} изображений\n")


def convert_bbox_to_yolo(x, y, width, height):
    """Конвертирует координаты Label Studio (0-100%) в формат YOLO"""
    x_center = (x + width / 2) / 100.0
    y_center = (y + height / 2) / 100.0
    w_norm = width / 100.0
    h_norm = height / 100.0

    # Ограничиваем значения
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w_norm = max(0.0, min(1.0, w_norm))
    h_norm = max(0.0, min(1.0, h_norm))

    return x_center, y_center, w_norm, h_norm


def process_split(file_list, split_name):
    """Обрабатывает одну выборку"""
    for task, img_path, original_name in file_list:
        # Копируем изображение с сохранением оригинального имени из JSON
        dest_img = OUTPUT_DIR / split_name / "images" / original_name
        shutil.copy2(img_path, dest_img)

        # Создаём label файл (с тем же именем, но .txt)
        label_name = Path(original_name).stem + ".txt"
        label_path = OUTPUT_DIR / split_name / "labels" / label_name

        annotations_count = 0
        with open(label_path, "w", encoding="utf-8") as f:
            for ann in task.get("annotations", []):
                for result in ann.get("result", []):
                    if result.get("type") != "rectanglelabels":
                        continue

                    labels = result["value"].get("rectanglelabels", [])
                    if not labels:
                        continue

                    label = labels[0]
                    if label not in CLASS_MAPPING:
                        continue

                    class_id = CLASS_MAPPING[label]

                    x = result["value"]["x"]
                    y = result["value"]["y"]
                    w = result["value"]["width"]
                    h = result["value"]["height"]

                    x_c, y_c, w_n, h_n = convert_bbox_to_yolo(x, y, w, h)

                    f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")
                    annotations_count += 1

        print(f"✅ {split_name}: {original_name} -> {annotations_count} объектов")


print("📝 Обработка тренировочной выборки:")
process_split(train_files, "train")

print("\n📝 Обработка валидационной выборки:")
process_split(val_files, "val")

# Создаём dataset.yaml для YOLO
yaml_content = f"""
# YOLO dataset configuration
path: {OUTPUT_DIR.absolute()}
train: train/images
val: val/images

nc: 5
names: ['drug_name', 'expiry_date', 'batch_number', 'barcode', 'package']
"""

yaml_path = OUTPUT_DIR / "dataset.yaml"
with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(yaml_content)

print(f"\n✅ YAML файл создан: {yaml_path}")
print("\n🎉 Готово! Теперь запустите обучение:")
print(f"\ncd {OUTPUT_DIR.parent}")
print(f"yolo detect train data={yaml_path} model=yolov8n.pt epochs=100 imgsz=640 batch=8")