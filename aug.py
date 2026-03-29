import json
import random
from pathlib import Path
from PIL import Image, ImageEnhance


def apply_color_correction(img: Image.Image) -> Image.Image:
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.7, 1.4))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.7, 1.4))
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(random.uniform(0.6, 1.6))
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(random.uniform(0.5, 1.5))
    return img


def augment_dataset(input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = input_path / "cleaned_annotations.json"
    with open(json_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    label_map = {}
    if isinstance(annotations, list):
        for entry in annotations:
            label_map[entry["filename"]] = entry["label"]
    elif isinstance(annotations, dict):
        label_map = annotations

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [f for f in input_path.rglob("*") if f.suffix.lower() in extensions]

    print(f"Найдено изображений: {len(image_files)}")
    total_generated = 0
    new_entries = []

    for img_path in image_files:
        relative = img_path.relative_to(input_path)
        out_subdir = output_path / relative.parent
        out_subdir.mkdir(parents=True, exist_ok=True)

        stem = img_path.stem
        suffix = img_path.suffix

        label = label_map.get(img_path.name) or label_map.get(str(relative))
        if label is None:
            print(f"  ⚠️  Лейбл не найден для {img_path.name}, пропускаем")
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  Ошибка открытия {img_path}: {e}")
            continue

        img_mirror = img.transpose(Image.FLIP_LEFT_RIGHT)
        img_color = apply_color_correction(img.copy())
        img_mirror_color = apply_color_correction(img_mirror.copy())

        variants = [
            (f"{stem}_original{suffix}",    img),
            (f"{stem}_mirror{suffix}",      img_mirror),
            (f"{stem}_original_cc{suffix}", img_color),
            (f"{stem}_mirror_cc{suffix}",   img_mirror_color),
        ]

        for filename, image in variants:
            image.save(out_subdir / filename)
            new_entries.append({
                "filename": filename,
                "label": label,
                "yolo_log": ""
            })

        total_generated += 4

    out_json_path = output_path / "cleaned_annotations.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(new_entries, f, ensure_ascii=False, indent=2)

    print(f"Готово! Сгенерировано изображений: {total_generated}")
    print(f"Аннотации сохранены в: {out_json_path}")
    print(f"Результат сохранён в: {output_path.resolve()}")


if __name__ == "__main__":
    INPUT_DIR  = "/Users/elvsevolod/PycharmProjects/SBER_animal_classifier/dataset_2"
    OUTPUT_DIR = "dataset_2_augmented"
    augment_dataset(INPUT_DIR, OUTPUT_DIR)