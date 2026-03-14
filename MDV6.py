import torch
import cv2
import os
import time
from PytorchWildlife.models import detection as pw_detection
import numpy as np

# *** Для запуска на процессоре раскомментировать ***
# *** Для запуска на видеокарте закомментировать ***
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Скрывает все GPU для CUDA/ROCm
#os.environ["PYTORCH_ROCM_ARCH"] = "none"   # Отключает архитектуру ROCm


# Переменные для настройки
SOURCE_DIR = r"C:\Storage\Downloads\val_set\val_set"  # Папка с датасетом
OUTPUT_DIR = "crops" # Папка для кропов
CALIBRATING = 1 # Управляющий флаг для запуска предподготовки

BATCH_SIZE = 70 # Для обычных запусков оптимальнее 8 - 32
MODEL_VERSION = "MDV6-yolov10-c" # буква c - маленькая модель, буква e - большая модель
# Список доступных моделей: MDV6-yolov9-c, MDV6-yolov9-e, MDV6-yolov10-c, MDV6-yolov10-e, MDV6-rtdetr-c

os.makedirs(OUTPUT_DIR, exist_ok = True) # Создание папки для кропов


# Загрузка модели
model = pw_detection.MegaDetectorV6(version = MODEL_VERSION)


# Предподготовка. При первом запуске компилируются низкоуровневые инструкции, что приводит к некорректному измерению времени работы
# Чтобы избежать некорректных замеров сначала запускается 2 тестовых запуска на пустых данных
# При финальной подготовке нужно будет убрать из кода
if (CALIBRATING == 1):
    print(f"\nПредподготовка при Batch Size = {BATCH_SIZE}...")
    start_warmup = time.time()

    # Создаем пустые изображения, выделяем для них память, компилируем инструкции
    warmup_imgs = [np.zeros((1280, 1280, 3), dtype=np.uint8) for _ in range(BATCH_SIZE)]

    with torch.no_grad():
        for i in range(2):
            print("\nПрогон №" + str(i + 1))
            i = model.batch_image_detection(warmup_imgs, batch_size = BATCH_SIZE)

            if torch.cuda.is_available() == True:
                torch.cuda.synchronize()

    print(f"\nПредподготовка завершена за {time.time() - start_warmup:.2f} сек.")


# Список всех изображений
image_paths = [os.path.join(SOURCE_DIR, f) for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

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

    # Запуск модели
    print("Batch №" + str(iteration))
    iteration += 1
    results = model.batch_image_detection(batch_imgs, batch_size = BATCH_SIZE)
    print()

    # Сохранение кропов
    for img, res, img_path in zip(batch_imgs, results, current_paths):
        h, w, _ = img.shape
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        for j, det in enumerate(res['detections']):
            bbox = det[0]
            score = float(det[2]) if det[2] is not None else 0.0

            # Координаты (проверка: пиксели или доли)
            if any(c > 1.1 for c in bbox):
                x1, y1, x2, y2 = map(int, bbox)
            else:
                x1, y1, x2, y2 = int(bbox[0] * w), int(bbox[1] * h), int(bbox[2] * w), int(bbox[3] * h)

            crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop.size > 0:
                cv2.imwrite(f"{OUTPUT_DIR}/{base_name}_crop_{j}.jpg", crop)

end_time = time.time()
total_time = end_time - start_time
print(f"Обработка завершена за {total_time:.2f} сек.")
print(f"Средняя скорость: {total_time / len(image_paths):.3f} сек/фото")