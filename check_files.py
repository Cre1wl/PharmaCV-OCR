import json
import os
from pathlib import Path

# Ваш путь к JSON файлу
json_path = Path("D:/PyCharm 2025.3/PythonProject/ForLabelStudio/pharma_dataset/Annotations.json")

# Проверяем, существует ли JSON файл
if not json_path.exists():
    print(f"❌ JSON файл не найден: {json_path}")
    exit(1)

print(f"✅ JSON файл найден: {json_path}")

# Загружаем JSON
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"\n📋 В JSON найдено {len(data)} записей")

# Получаем список файлов из JSON
json_files = []
for task in data:
    img_file = task.get("file_upload", "")
    if img_file:
        json_files.append(img_file)
        print(f"  - {img_file}")

# Проверяем, какие папки существуют
print("\n📁 Проверяем возможные расположения файлов:")

# Возможные пути к папке с изображениями
possible_dirs = [
    Path("D:/PyCharm 2025.3/PythonProject/ForLabelStudio/pharma_dataset/images"),
    Path("D:/PyCharm 2025.3/PythonProject/ForLabelStudio/images"),
    Path("D:/PyCharm 2025.3/PythonProject/ForLabelStudio"),
    Path("D:/PyCharm 2025.3/PythonProject/ForLabelStudio/pharma_dataset"),
]

found_any = False
for search_dir in possible_dirs:
    if search_dir.exists():
        print(f"\n  ✅ Папка существует: {search_dir}")

        # Проверяем, какие файлы из JSON есть в этой папке
        found_files = []
        for img_file in json_files:
            full_path = search_dir / img_file
            if full_path.exists():
                found_files.append(img_file)

        if found_files:
            print(f"     Найдено {len(found_files)} из {len(json_files)} файлов:")
            for f in found_files[:5]:  # показываем первые 5
                print(f"       ✅ {f}")
            if len(found_files) > 5:
                print(f"       ... и еще {len(found_files) - 5} файлов")
            found_any = True
        else:
            print(f"     ❌ Файлы не найдены в этой папке")
    else:
        print(f"\n  ❌ Папка не существует: {search_dir}")

if not found_any:
    print("\n❌ НИ ОДНОГО ФАЙЛА НЕ НАЙДЕНО!")
    print("\n💡 Возможные решения:")
    print("1. Убедитесь, что изображения действительно скачаны/скопированы в папку")
    print("2. Проверьте имена файлов - они должны точно совпадать с именами в JSON")
    print("3. Укажите правильный путь к папке с изображениями")

    # Показываем содержимое папки pharma_dataset если она существует
    pharma_dir = Path("D:/PyCharm 2025.3/PythonProject/ForLabelStudio/pharma_dataset")
    if pharma_dir.exists():
        print(f"\n📂 Содержимое папки {pharma_dir}:")
        for item in pharma_dir.iterdir():
            if item.is_file():
                print(f"   📄 {item.name}")
            elif item.is_dir():
                print(f"   📁 {item.name}/")
                # Показываем первые 5 файлов в подпапке
                for subitem in list(item.iterdir())[:5]:
                    print(f"      📄 {subitem.name}")