import torch
import cv2
import os
import time
import torchvision.models as models
from PytorchWildlife.models import detection as pw_detection
import numpy as np
from PIL import Image

# *** Для запуска на процессоре раскомментировать ***
# *** Для запуска на видеокарте закомментировать ***
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Скрывает все GPU для CUDA/ROCm
#os.environ["PYTORCH_ROCM_ARCH"] = "none"   # Отключает архитектуру ROCm

# Переменные для настройки
SOURCE_DIR = r"C:\Storage\Downloads\val_set (1)\val_set"  # Папка с датасетом
OUTPUT_DIR = "crops"  # Папка для кропов
CALIBRATING = 1  # Управляющий флаг для запуска предподготовки

BATCH_SIZE = 4  # Для обычных запусков оптимальнее 8 - 32
MODEL_VERSION = "MDV6-yolov10-c"  # Это детектор; буква c - маленькая модель, буква e - большая модель
# Список доступных моделей: MDV6-yolov9-c, MDV6-yolov9-e, MDV6-yolov10-c, MDV6-yolov10-e, MDV6-rtdetr-c

os.makedirs(OUTPUT_DIR, exist_ok=True)  # Создание папки для кропов

# Загрузка моделей
# Автоматическое определение устройства на основе настроек окружения
device = torch.device("cuda" if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" else "cpu")
print(f"Используемое устройство: {device}")

# 1. Загрузка детектора
model = pw_detection.MegaDetectorV6(version=MODEL_VERSION, device=device)  # Детектор

# 2. Загрузка классификатора (ResNet50)
weights = models.ResNet50_Weights.IMAGENET1K_V1
clf_model = models.resnet50(weights=weights)
clf_model.to(device)
clf_model.eval()

# Получаем стандартные трансформации и названия классов ImageNet
preprocess = weights.transforms()
categories = weights.meta["categories"]

# Предподготовка. При первом запуске компилируются низкоуровневые инструкции, что приводит к некорректному измерению времени работы
# Чтобы избежать некорректных замеров сначала запускается 2 тестовых запуска на пустых данных
if (CALIBRATING == 1):
    print(f"\nПредподготовка при Batch Size = {BATCH_SIZE}...")
    start_warmup = time.time()

    # Создаем пустые изображения, выделяем для них память, компилируем инструкции
    warmup_imgs = [np.zeros((1280, 1280, 3), dtype=np.uint8) for _ in range(BATCH_SIZE)]

    with torch.no_grad():
        for i in range(2):
            print("\nПрогон №" + str(i + 1))
            _ = model.batch_image_detection(warmup_imgs, batch_size=BATCH_SIZE)

            if device.type == "cuda":
                torch.cuda.synchronize()

    print(f"\nПредподготовка завершена за {time.time() - start_warmup:.2f} сек.")

# Список всех изображений
image_paths = [os.path.join(SOURCE_DIR, f) for f in os.listdir(SOURCE_DIR) if
               f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"\nКоличество изображений: {len(image_paths)} | Batch size: {BATCH_SIZE}\n")

start_time = time.time()

# Загружаем изображения
iteration = 1
for i in range(0, len(image_paths), BATCH_SIZE):
    current_paths = image_paths[i: i + BATCH_SIZE]

    # Загрузка изображений текущего батча
    batch_imgs = []
    for p in current_paths:
        img_raw = cv2.imread(p)
        if img_raw is not None:
            batch_imgs.append(img_raw)
        else:
            print(f"Не удалось загрузить изображение: {p}")

    if not batch_imgs:
        continue

    # Запуск модели детекции
    print("Batch №" + str(iteration))
    print("Время детекции. Полное время в конце работы")
    iteration += 1
    results = model.batch_image_detection(batch_imgs, batch_size=BATCH_SIZE)
    print()

    # Сохранение кропов и классификация
    for img, res, img_path in zip(batch_imgs, results, current_paths):
        h, w, _ = img.shape
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        for j, det in enumerate(res['detections']):
            bbox = det[0]

            # Приведение координат к пикселям (проверка: пиксели или доли)
            if all(c <= 1.1 for c in bbox):
                x1, y1, x2, y2 = int(bbox[0] * w), int(bbox[1] * h), int(bbox[2] * w), int(bbox[3] * h)
            else:
                x1, y1, x2, y2 = map(int, bbox)

            # Создаем кроп
            crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

            if crop.size > 0:
                # Подготовка кропа для классификации
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_pil = Image.fromarray(crop_rgb)

                with torch.no_grad():
                    # Применяем трансформации (Resize, Normalize) и перенос на устройство
                    input_tensor = preprocess(crop_pil).unsqueeze(0).to(device)

                    # Прогон через ResNet50
                    logits = clf_model(input_tensor)
                    prediction_idx = torch.argmax(logits, dim=1).item()

                # Получение текстовой метки класса ImageNet
                label = categories[prediction_idx].lower()

                # Логика сортировки
                if "fox" in label:
                    animal_class = "1_fox"
                elif "wolf" in label:
                    animal_class = "2_wolf"
                else:
                    animal_class = "3_other"

                # Сохранение результата
                class_path = os.path.join(OUTPUT_DIR, animal_class)
                os.makedirs(class_path, exist_ok=True)
                cv2.imwrite(os.path.join(class_path, f"{animal_class}_{base_name}_crop_{j}.jpg"), crop)

                print(f"  -> Обнаружен: {label} (сохранено в {animal_class})")

end_time = time.time()
total_time = end_time - start_time
print(f"Обработка завершена за {total_time:.2f} сек.")
if len(image_paths) > 0:
    print(f"Средняя скорость: {total_time / len(image_paths):.3f} сек/фото")
