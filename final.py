import os
import cv2
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import label_binarize
from PytorchWildlife.models import detection as pw_detection

# ==========================================
# 1. НАСТРОЙКИ ПУТЕЙ И ПАРАМЕТРОВ
# ==========================================
SOURCE_DIR = "/Users/elvsevolod/PycharmProjects/SBER_animal_classifier/val_set"
MODEL_WEIGHTS = "/Users/elvsevolod/PycharmProjects/SBER_animal_classifier/resnet50_wildlife_best.pth"
BATCH_SIZE = 32
MODEL_VERSION = "MDV6-yolov10-c"
IMG_SIZE = 256

# Определение устройства (используем MPS для мака, если доступно)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"✅ Используется устройство: {DEVICE}")

# ==========================================
# 2. ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ
# ==========================================
# --- Детектор ---
print("Загрузка MegaDetector...")
detector = pw_detection.MegaDetectorV6(version=MODEL_VERSION, device="cpu", pretrained=True)
detector.predictor.args.conf = 0.2

# --- Классификатор (ResNet50) ---
print("Загрузка классификатора ResNet50...")


def build_model(num_classes: int):
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes)
    )
    return model


classifier = build_model(num_classes=3)
classifier.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
classifier.to(DEVICE)
classifier.eval()

infer_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 3. ПОДГОТОВКА СПИСКОВ И ФУНКЦИЙ
# ==========================================
image_paths = [os.path.join(SOURCE_DIR, f) for f in os.listdir(SOURCE_DIR)
               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
print(f"\nКоличество изображений: {len(image_paths)}\n")

predictions = []  # Для result.csv

# Списки для метрик (чтобы рассчитать ROC-AUC и F1)
true_labels_metrics = []
pred_labels_metrics = []
pred_probs_metrics = []


def get_true_label_from_filename(filename: str) -> int:
    """
    Парсит имя файла для получения реального класса.
    Основано на префиксах: fox..., wolf..., other...
    Возвращает индексы ResNet: 0 (лиса), 1 (волк), 2 (другое).
    """
    name = filename.lower()
    if name.startswith("fox"):
        return 0
    elif name.startswith("wolf"):
        return 1
    elif name.startswith("other"):
        return 2
    else:
        # Если префикс неизвестен, относим к "other"
        return 2


# ==========================================
# 4. ИНФЕРЕНС (ОБРАБОТКА БАТЧАМИ)
# ==========================================
start_time = time.time()

for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Обработка батчей"):
    current_paths = image_paths[i: i + BATCH_SIZE]
    batch_imgs = []

    for p in current_paths:
        img_raw = cv2.imread(p)
        if img_raw is not None:
            # Детектор и классификатор ожидают RGB
            batch_imgs.append(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB))
        else:
            print(f"⚠️ Ошибка загрузки: {p}")

    if not batch_imgs:
        continue

    # --- ДЕТЕКЦИЯ ---
    results = detector.batch_image_detection(batch_imgs, batch_size=len(batch_imgs))

    # --- КРОППИНГ И КЛАССИФИКАЦИЯ ---
    for img, res, img_path in zip(batch_imgs, results, current_paths):
        filename = os.path.basename(img_path)
        h, w, _ = img.shape

        detections = res.get("detections")
        true_cls = get_true_label_from_filename(filename)

        # ----------------------------------------------------
        # СЛУЧАЙ А: Детектор не нашел животных
        # ----------------------------------------------------
        if detections is None or len(detections.xyxy) == 0:
            # Запись в финальный CSV строго по ТЗ: 0, 0
            predictions.append({
                "filename": filename,
                "animal_detected": 0,
                "animal_class": 0
            })

            # Логирование для МЕТРИК:
            # Записываем как уверенное предсказание класса "other" (индекс 2)
            true_labels_metrics.append(true_cls)
            pred_labels_metrics.append(2)
            pred_probs_metrics.append(np.array([0.0, 0.0, 1.0]))
            continue

        # ----------------------------------------------------
        # СЛУЧАЙ Б: Детектор нашел животных (вырезаем кропы)
        # ----------------------------------------------------
        crops_tensors = []
        for box in detections.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            pad = 10
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(w, x2 + pad), min(h, y2 + pad)

            crop_arr = img[y1:y2, x1:x2]
            if crop_arr.size > 0:
                crop_pil = Image.fromarray(crop_arr)
                crops_tensors.append(infer_transform(crop_pil))

        if not crops_tensors:
            # Если кропы получились пустыми по размеру (редкий баг)
            predictions.append({"filename": filename, "animal_detected": 0, "animal_class": 0})
            true_labels_metrics.append(true_cls)
            pred_labels_metrics.append(2)
            pred_probs_metrics.append(np.array([0.0, 0.0, 1.0]))
            continue

        # ----------------------------------------------------
        # СЛУЧАЙ В: Классификация вырезанных кропов
        # ----------------------------------------------------
        crops_batch = torch.stack(crops_tensors).to(DEVICE)
        with torch.no_grad():
            outputs = classifier(crops_batch)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()

        for prob in probs:
            pred_idx = np.argmax(prob)  # 0: fox, 1: wolf, 2: other

            # Переводим индексы ResNet в индексы CSV из ТЗ
            # В ТЗ: 1 - лиса, 2 - волк, 3 - другое
            csv_class = pred_idx + 1

            predictions.append({
                "filename": filename,
                "animal_detected": 1,
                "animal_class": csv_class
            })

            # Собираем данные для метрик
            true_labels_metrics.append(true_cls)
            pred_labels_metrics.append(pred_idx)
            pred_probs_metrics.append(prob)

total_time = time.time() - start_time
print(f"\n✅ Обработка завершена за {total_time:.2f} сек.")
print(f"🚀 Средняя скорость: {total_time / len(image_paths):.3f} сек/фото")

# ==========================================
# 5. СОХРАНЕНИЕ CSV РЕЗУЛЬТАТОВ (Формат ТЗ)
# ==========================================
df_preds = pd.DataFrame(predictions)
df_preds.to_csv("result_2.csv", index=False)
print("📄 Файл result_2.csv успешно сохранен!")

# ==========================================
# 6. РАСЧЕТ И СОХРАНЕНИЕ МЕТРИК
# ==========================================
if len(true_labels_metrics) > 0:
    true_labels = np.array(true_labels_metrics)
    pred_labels = np.array(pred_labels_metrics)
    pred_probs = np.array(pred_probs_metrics)

    classes_names = ["fox", "wolf", "other"]
    labels_bin = label_binarize(true_labels, classes=[0, 1, 2])
    metrics_rows = []

    for i, cls in enumerate(classes_names):
        tp = np.sum((pred_labels == i) & (true_labels == i))
        tn = np.sum((pred_labels != i) & (true_labels != i))
        fp = np.sum((pred_labels == i) & (true_labels != i))
        fn = np.sum((pred_labels != i) & (true_labels == i))

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        specificity = tn / (tn + fp + 1e-9)
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)

        # ROC-AUC нельзя посчитать, если в выборке только один класс
        roc_auc = roc_auc_score(labels_bin[:, i], pred_probs[:, i]) if len(np.unique(true_labels)) > 1 else 0.0

        metrics_rows.append({
            "class": cls,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "specificity": round(specificity, 4),
            "f1": round(f1, 4),
            "roc_auc": round(roc_auc, 4)
        })

    f1_macro = f1_score(true_labels, pred_labels, average="macro")
    roc_macro = roc_auc_score(labels_bin, pred_probs, multi_class="ovr", average="macro") if len(
        np.unique(true_labels)) > 1 else 0.0

    metrics_rows.append({
        "class": "MACRO",
        "precision": round(precision_score(true_labels, pred_labels, average="macro", zero_division=0), 4),
        "recall": round(recall_score(true_labels, pred_labels, average="macro", zero_division=0), 4),
        "specificity": "-",
        "f1": round(f1_macro, 4),
        "roc_auc": round(roc_macro, 4)
    })

    df_metrics = pd.DataFrame(metrics_rows)
    df_metrics.to_csv("metrics_2.csv", index=False)
    print("📈 Файл metrics_2.csv успешно сохранен!")
    print("\n" + df_metrics.to_string(index=False))