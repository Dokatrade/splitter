"""
Скрипт для пропорционального уменьшения изображений
и (опционально) удаления фона. Результат сохраняется в формате PNG.

Источник:  d:/results/size&back/
Результат: d:/results/size&back_result/
"""

import os
os.environ["ORT_LOG_LEVEL"] = "3"  # подавить предупреждения ONNX Runtime (CUDA и т.п.)
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# =============================================================================
# НАСТРОЙКИ (менять здесь)
# =============================================================================

# Приблизительный целевой размер (px) по большей стороне.
# Скрипт сам пропорционально подгонит обе стороны.
TARGET_SIZE = 250

# Удалять ли фон: 1 - только уменьшить размер, 2 - уменьшить + удалить фон.
REMOVE_BG = 2

# Модель для удаления фона (работает только при REMOVE_BG = 2):
#   1 — u2net              (базовая, универсальная)
#   2 — isnet-general-use  (лучше на мелких деталях: ветки, волосы, кружево)
#   3 — u2net_human_seg    (оптимизирована для людей / портретов)
#   4 — silueta            (лёгкая и быстрая, но менее точная)
REMBG_MODEL = 2

# Постобработка: убрать оставшиеся белые/светлые пиксели внутри объекта.
# True — включить, False — выключить.
REMOVE_WHITE_BG = False

# Порог «белизны» (0–255). Работает только при REMOVE_WHITE_BG = True.
# Пиксель считается белым, если R, G и B все выше порога.
# 240 — только почти белые, 220 — посветлее тоже захватит, 200 — агрессивнее.
WHITE_THRESHOLD = 220


# =============================================================================

INPUT_DIR = Path(r"d:\results\size&back")
OUTPUT_DIR = Path(r"d:\results\size&back_result")

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".tif", ".webp"}

_MODEL_MAP = {
    1: "u2net",
    2: "isnet-general-use",
    3: "u2net_human_seg",
    4: "silueta",
}

_rembg_session = None  # lazy-loaded, reused for all images


def _get_rembg_session():
    """Create or return the cached rembg session."""
    global _rembg_session
    if _rembg_session is None:
        from rembg import new_session
        model_name = _MODEL_MAP.get(REMBG_MODEL, "u2net")
        print(f"  Loading model: {model_name} ...")
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        try:
            os.dup2(devnull, 2)
            _rembg_session = new_session(model_name)
        finally:
            os.dup2(old_stderr, 2)
            os.close(devnull)
            os.close(old_stderr)
    return _rembg_session


def _remove_bg(img: Image.Image) -> Image.Image:
    """Удаляет фон, подавляя нативные предупреждения ONNX Runtime."""
    from rembg import remove

    session = _get_rembg_session()

    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    try:
        os.dup2(devnull, 2)
        result = remove(img, session=session)
    finally:
        os.dup2(old_stderr, 2)
        os.close(devnull)
        os.close(old_stderr)
    return result


def _remove_white_pixels(img: Image.Image, threshold: int = 240) -> Image.Image:
    """Делает прозрачными пиксели, у которых R, G и B все выше threshold."""
    img = img.convert("RGBA")
    data = np.array(img)
    r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]
    white_mask = (r >= threshold) & (g >= threshold) & (b >= threshold)
    data[:, :, 3] = np.where(white_mask, 0, a)
    return Image.fromarray(data)


def proportional_resize(img: Image.Image, target: int) -> Image.Image:
    """
    Уменьшает изображение пропорционально так, чтобы большая сторона
    стала равна target (с округлением вниз).

    Если обе стороны уже <= target -- возвращает изображение без изменений.
    """
    w, h = img.size

    if max(w, h) <= target:
        return img

    scale = target / max(w, h)
    new_w = max(int(w * scale), 1)
    new_h = max(int(h * scale), 1)

    return img.resize((new_w, new_h), Image.LANCZOS)


def process_image(src_path: Path, dst_dir: Path) -> None:
    """Обрабатывает одно изображение."""
    print(f"  > {src_path.name} ... ", end="", flush=True)

    img = Image.open(src_path).convert("RGBA")
    original_size = img.size

    # 1. Пропорциональное уменьшение
    img = proportional_resize(img, TARGET_SIZE)

    # 2. Удаление фона (если включено)
    if REMOVE_BG == 2:
        img = _remove_bg(img)

    # 3. Постобработка: убрать оставшиеся белые пиксели
    if REMOVE_WHITE_BG:
        img = _remove_white_pixels(img, WHITE_THRESHOLD)

    # 4. Сохранение в PNG
    out_name = src_path.stem + ".png"
    out_path = dst_dir / out_name
    img.save(out_path, "PNG")

    resized = original_size != img.size
    size_info = f"{original_size[0]}x{original_size[1]} -> " if resized else ""
    print(f"OK  ({size_info}{img.size[0]}x{img.size[1]})")


def main() -> None:
    if not INPUT_DIR.exists():
        print(f"Papka-istochnik ne najdena: {INPUT_DIR}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = [
        f for f in INPUT_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        print(f"V papke {INPUT_DIR} net kartinok.")
        sys.exit(0)

    mode_label = "umenshenie + udalenie fona" if REMOVE_BG == 2 else "tolko umenshenie"
    model_name = _MODEL_MAP.get(REMBG_MODEL, "u2net") if REMOVE_BG == 2 else ""
    model_info = f", model: {model_name}" if model_name else ""
    print(f"Rezhim: {mode_label}{model_info}, razmer: ~{TARGET_SIZE} px")
    print(f"Najdeno izobrazhenij: {len(files)}")
    print(f"Rezultat -> {OUTPUT_DIR}")
    print()

    for f in sorted(files):
        try:
            process_image(f, OUTPUT_DIR)
        except Exception as exc:
            print(f"  Oshibka: {f.name}: {exc}")

    print()
    print("Gotovo!")


if __name__ == "__main__":
    main()
