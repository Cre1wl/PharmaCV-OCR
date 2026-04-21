import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from datetime import datetime, timedelta
import re

# --- ЗАГРУЖАЕМ МОДЕЛЬ ---
MODEL_PATH = "runs/detect/train/weights/best.pt"
yolo_model = YOLO(MODEL_PATH)
print(f"✅ Загружена модель: {MODEL_PATH}")

reader = easyocr.Reader(['ru', 'en'], gpu=False)

CLASS_NAMES = {
    0: "drug_name",
    1: "expiry_date",
    2: "batch_number",
    3: "barcode",
    4: "package"
}

CLASS_COLORS = {
    "drug_name": (255, 0, 0),
    "expiry_date": (0, 255, 0),
    "batch_number": (0, 255, 255),
    "barcode": (0, 165, 255),
    "package": (255, 0, 255)
}


def extract_all_fields(text):
    """
    Извлекает все поля из текста с правильным сопоставлением
    """
    result = {}

    # Разбиваем текст на строки или фрагменты по пробелам
    # Но лучше искать паттерны "КЛЮЧ: ЗНАЧЕНИЕ"

    # 1. Поиск КП (GTIN) - обычно 13-14 цифр, может быть после "КП:" или "GTIN:"
    kp_patterns = [
        r'КП[:\s]*([0-9]{13,14})',
        r'GTIN[:\s]*([0-9]{13,14})',
        r'КП[:\s]*([0-9]+)',
    ]
    for pattern in kp_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result["КП"] = match.group(1)
            break

    # 2. Поиск СН (S/N) - может быть буквенно-цифровым, после "СН:", "CH:", "S.N.:", "SN:"
    sn_patterns = [
        r'СН[:\s]*([A-Z0-9]{8,15})',
        r'CH[:\s]*([A-Z0-9]{8,15})',
        r'S\.N\.[:\s]*([A-Z0-9]{8,15})',
        r'SN[:\s]*([A-Z0-9]{8,15})',
        r'Serial[:\s]*([A-Z0-9]{8,15})',
    ]
    for pattern in sn_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result["СН"] = match.group(1)
            break

    # 3. Поиск Серии (Batch/Lot)
    batch_patterns = [
        r'Серия[:\s№]*([A-Z0-9]{5,12})',
        r'Batch[:\s]*([A-Z0-9]{5,12})',
        r'Lot[:\s]*([A-Z0-9]{5,12})',
        r'№ серии[:\s]*([A-Z0-9]{5,12})',
    ]
    for pattern in batch_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result["Серия"] = match.group(1)
            break

    # 4. Поиск срока годности
    expiry_patterns = [
        r'Годен до[:\s]*([0-9]{2}[/.-][0-9]{2,4})',
        r'Expiry[:\s]*([0-9]{2}[/.-][0-9]{2,4})',
        r'Exp[:\s]*([0-9]{2}[/.-][0-9]{2,4})',
        r'EXP[:\s]*([0-9]{2}[/.-][0-9]{2,4})',
        r'до[:\s]*([0-9]{2}[/.-][0-9]{2,4})',
    ]
    for pattern in expiry_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result["Годен до"] = match.group(1)
            break

    # Если не нашли по паттернам, ищем просто дату в тексте
    if "Годен до" not in result:
        date_pattern = r'\b([0-9]{2}[/.-][0-9]{2,4})\b'
        dates = re.findall(date_pattern, text)
        if dates:
            result["Годен до"] = dates[0]

    # 5. Поиск даты производства
    mfg_patterns = [
        r'Дата произв[:\s]*([0-9]{2}[/.-][0-9]{2,4})',
        r'Произведено[:\s]*([0-9]{2}[/.-][0-9]{2,4})',
        r'MFG[:\s]*([0-9]{2}[/.-][0-9]{2,4})',
        r'Manufactured[:\s]*([0-9]{2}[/.-][0-9]{2,4})',
    ]
    for pattern in mfg_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result["Дата производства"] = match.group(1)
            break

    return result


def parse_expiry_date(date_text):
    """Парсит срок годности"""
    if not date_text:
        return None

    date_text = str(date_text).strip()

    patterns = [
        (r'(\d{2})[/.-](\d{2,4})', 'month_year'),
        (r'(\d{2})[/.-](\d{2})[/.-](\d{4})', 'day_month_year'),
        (r'(\d{4})[/.-](\d{2})', 'year_month'),
    ]

    for pattern, date_type in patterns:
        match = re.search(pattern, date_text)
        if match:
            try:
                if date_type == 'month_year':
                    month = int(match.group(1))
                    year_str = match.group(2)
                    if len(year_str) == 2:
                        year = 2000 + int(year_str)
                    else:
                        year = int(year_str)

                    if month < 1 or month > 12:
                        continue

                    if month == 12:
                        return datetime(year + 1, 1, 1) - timedelta(days=1)
                    else:
                        return datetime(year, month + 1, 1) - timedelta(days=1)

                elif date_type == 'day_month_year':
                    day = int(match.group(1))
                    month = int(match.group(2))
                    year = int(match.group(3))
                    if 1 <= month <= 12 and 1 <= day <= 31:
                        return datetime(year, month, day)

                elif date_type == 'year_month':
                    year = int(match.group(1))
                    month = int(match.group(2))
                    if 1 <= month <= 12:
                        if month == 12:
                            return datetime(year + 1, 1, 1) - timedelta(days=1)
                        else:
                            return datetime(year, month + 1, 1) - timedelta(days=1)

            except Exception:
                continue

    return None


def check_expiry(expiry_date_obj):
    """Проверяет срок годности"""
    if expiry_date_obj is None:
        return "unknown", "❓ Срок годности не распознан", "gray"

    today = datetime.now().date()
    expiry_date = expiry_date_obj.date() if hasattr(expiry_date_obj, 'date') else expiry_date_obj

    days_left = (expiry_date - today).days

    if days_left < 0:
        return "expired", f"❌ ПРОСРОЧЕНО! Просрочено на {abs(days_left)} дней (было до {expiry_date.strftime('%d.%m.%Y')})", "red"
    elif days_left <= 30:
        return "warning", f"⚠️ Истекает через {days_left} дней (до {expiry_date.strftime('%d.%m.%Y')})", "orange"
    else:
        return "good", f"✅ ГОДЕН. Срок истекает через {days_left} дней (до {expiry_date.strftime('%d.%m.%Y')})", "green"


def process_image(image):
    """Основная функция обработки"""
    if image is None:
        return None, "❌ Ошибка: изображение не загружено"

    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # --- Детекция YOLO (для визуализации) ---
    results = yolo_model(img_bgr)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        class_name = CLASS_NAMES.get(cls_id, f"unknown_{cls_id}")
        color = CLASS_COLORS.get(class_name, (0, 255, 0))
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

    # --- ОСНОВНОЙ OCR ПО ВСЕМУ ИЗОБРАЖЕНИЮ ---
    print("🔍 Запускаем OCR...")
    ocr_results = reader.readtext(img_bgr)

    # Собираем весь текст
    full_text = " ".join([res[1] for res in ocr_results])
    print(f"Распознанный текст: {full_text}")

    # Извлекаем поля
    extracted_fields = extract_all_fields(full_text)

    # Визуализация найденных полей
    for field_name, field_value in extracted_fields.items():
        for (bbox, text, conf) in ocr_results:
            if field_value in text or text in field_value:
                pts = np.array(bbox, dtype=np.int32)
                color = (255, 255, 0)
                cv2.polylines(img_bgr, [pts], True, color, 2)
                x, y = pts[0]
                label = f"{field_name}: {field_value}"
                cv2.putText(img_bgr, label, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                break

    # --- АНАЛИЗ СРОКА ГОДНОСТИ ---
    expiry_text = extracted_fields.get("Годен до", "")
    expiry_date_obj = parse_expiry_date(expiry_text)
    expiry_status, expiry_message, expiry_color = check_expiry(expiry_date_obj)

    # --- ФОРМИРОВАНИЕ ОТЧЁТА ---
    result_text = "=" * 55 + "\n"
    result_text += "📋 РЕЗУЛЬТАТ ПРОВЕРКИ ЛЕКАРСТВЕННОГО ПРЕПАРАТА\n"
    result_text += "=" * 55 + "\n\n"

    if extracted_fields:
        for field_name, field_value in extracted_fields.items():
            result_text += f"📌 {field_name}: {field_value}\n"
    else:
        result_text += "❌ Не удалось распознать поля\n"

    result_text += "\n" + "=" * 55 + "\n"
    result_text += f"📅 {expiry_message}\n\n"

    result_text += "=" * 55 + "\n"
    result_text += "🏥 ЗАКЛЮЧЕНИЕ:\n"

    if expiry_status == "expired":
        result_text += "❌ ЛЕКАРСТВО ПРОСРОЧЕНО! Использовать НЕЛЬЗЯ.\n"
        result_text += "   Утилизируйте препарат согласно правилам.\n"
    elif expiry_status == "warning":
        result_text += "⚠️ Срок годности истекает. Препарат ещё годен.\n"
    elif expiry_status == "good":
        result_text += "✅ Препарат годен к применению.\n"
    else:
        result_text += "❓ Не удалось определить срок годности.\n"

    result_text += "=" * 55 + "\n"
    result_text += "\n💡 При сомнениях обратитесь к фармацевту."

    # Отладочная информация
    result_text += f"\n\n---\n🔍 Распознанный текст:\n{full_text}"

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    return img_rgb, result_text


# --- ИНТЕРФЕЙС ---
with gr.Blocks(title="💊 Проверка срока годности лекарств", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 💊 Система проверки срока годности лекарственных средств

    **Автоматическое распознавание полей:**
    - 📌 **КП / GTIN** - код продукции (13-14 цифр)
    - 📌 **СН / S/N** - серийный номер (буквы+цифры)
    - 📌 **Серия** - номер партии
    - 📌 **Годен до** - срок годности (с проверкой)

    > **Просто сфотографируйте упаковку** - программа сама найдёт все поля!
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="numpy", label="📸 Загрузите фото упаковки", height=450)
            submit_btn = gr.Button("🔍 ПРОВЕРИТЬ", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_image = gr.Image(label="🔍 Результат распознавания", height=450)
            output_text = gr.Textbox(label="📋 ЗАКЛЮЧЕНИЕ", lines=20, max_lines=30)

    submit_btn.click(
        fn=process_image,
        inputs=input_image,
        outputs=[output_image, output_text]
    )

    gr.Markdown("""
    ---
    ### 📌 Поддерживаемые форматы:
    | Поле | Ключевые слова | Пример |
    |------|----------------|--------|
    | КП | `КП:`, `GTIN:` | `05903060624399` |
    | СН | `СН:`, `CH:`, `S.N.:` | `0023445049724` |
    | Серия | `Серия:`, `Batch:`, `Lot:` | `0614025` |
    | Годен до | `Годен до:`, `Expiry:`, `EXP:` | `07/2027` |
    """)

if __name__ == "__main__":
    demo.launch(share=True, server_name="127.0.0.1", server_port=7860)