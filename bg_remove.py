#!/usr/bin/env python3
"""
bg_remove.py
============

CLI-скрипт для пакетного/одиночного удаления фона из изображений через
`rembg` с:
1. авто-профилем сцены (`--profile auto`);
2. fallback на вторую модель (`--fallback-mode smart|always`);
3. постобработкой краев (edge refine + decontamination).

Что делает:
1. Принимает входной путь: файл или папка.
2. Предварительно уменьшает изображение пропорционально (настраивается в константе
   `DEFAULT_PRE_RESIZE_MAX_SIDE` вверху файла).
3. Удаляет фон у уменьшенного изображения.
4. Сохраняет результат:
   - для папки: всегда в `.png`, с сохранением структуры подпапок;
   - для файла: в указанный файл или в папку (см. правила ниже).
   - если предуменьшение включено, сохраняется уменьшенный результат.

Требования:
1. Python 3.8+ (рекомендуется 3.10+).
2. Установлен пакет `rembg` и его зависимости (включая Pillow).

Установка (пример):
    pip install rembg

Если команда `python` не найдена, используйте `python3`.

Базовый синтаксис:
    python3 bg_remove.py [input] [output] [options]

Пути по умолчанию (если аргументы не переданы):
    input  = d:\\results\\size&back
    output = d:\\results\\size&back_result

Аргументы:
1. `input` (необязательный):
   - путь к файлу или папке с изображениями.
2. `output` (необязательный):
   - путь к файлу или папке назначения.

Опции (основные):
1. `-m, --model` - первичная модель.
2. `--profile auto|general|portrait|product|graphic|animal`
   - профиль маршрутизации модели.
3. `--fallback-model` - явная резервная модель.
4. `--fallback-mode off|smart|always`
   - режим fallback:
     - `off`: без fallback;
     - `smart`: fallback при плохой маске;
     - `always`: всегда прогонять 2 модели и выбирать лучшее.
5. `--min-fg-coverage`, `--max-fg-coverage`, `--min-quality-score`
   - пороги для `smart` fallback.
6. `--max-inference-pixels`
   - лимит пикселей для инференса (большие кадры уменьшаются перед обработкой).
7. `--alpha-matting-max-pixels`
   - выше этого порога alpha matting отключается для защиты от OOM.
8. `--no-alpha-matting`, `--af`, `--ab`, `--ae`
   - настройки alpha matting.
9. `--only-mask`, `--post-process-mask`
   - маска и очистка маски.
10. `--edge-refine-radius`
   - сглаживание краев альфа-канала (0 отключает).
11. `--decontaminate off|light|medium|strong`
   - ослабление цветных ореолов по краю.
12. `--bgcolor`
   - цвет фона вместо прозрачности.
   Поддерживаемые форматы:
     - `#RRGGBB`
     - `#RRGGBBAA`
     - `R,G,B`
     - `R,G,B,A`
   Также можно: `none`, `transparent`, `trans` (прозрачный фон).

Режим одиночного файла (`input` = файл):
1. Если `output` указывает на существующую папку -> результат:
   `<output>/<input_stem>.png`.
2. Если `output` оканчивается на `/` или `\\` -> как папка:
   `<output>/<input_stem>.png`.
3. Если `output` не существует и без расширения -> как папка:
   `<output>/<input_stem>.png`.
4. Иначе `output` считается явным путем к выходному файлу.

Режим папки (`input` = папка):
1. Рекурсивно обрабатывает только расширения:
   `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tiff`, `.tif`.
2. Всегда пишет результат как `.png`.
3. Сохраняет относительную структуру подпапок.
4. Не прерывается на одной ошибке: продолжает обработку остальных файлов.
5. В конце печатает итог: сколько успешно/с ошибкой.

Примеры:
1. Обработать папку путями по умолчанию:
    python3 bg_remove.py

2. Обработать одну картинку в явный файл:
    python3 bg_remove.py "d:/in/photo.jpg" "d:/out/photo_cut.png"

3. Обработать одну картинку в папку:
    python3 bg_remove.py "d:/in/photo.jpg" "d:/out/"

4. Пакетно обработать папку:
    python3 bg_remove.py "d:/in_folder" "d:/out_folder"

5. Пакетно с белым фоном:
    python3 bg_remove.py "d:/in_folder" "d:/out_folder" --bgcolor "#FFFFFF"

6. Только маска:
    python3 bg_remove.py "d:/in_folder" "d:/out_folder" --only-mask

7. Автопрофиль + smart fallback (рекомендуемый режим):
    python3 bg_remove.py "d:/in_folder" "d:/out_folder" --profile auto --fallback-mode smart

8. Усиленная постобработка края:
    python3 bg_remove.py "d:/in_folder" "d:/out_folder" --edge-refine-radius 1.2 --decontaminate medium

Настройка предуменьшения вверху скрипта:
1. `DEFAULT_PRE_RESIZE_MAX_SIDE = 300`
   - уменьшать так, чтобы большая сторона была около 300 px (пропорционально).
2. `DEFAULT_PRE_RESIZE_MAX_SIDE = None` или `DEFAULT_PRE_RESIZE_MAX_SIDE = ""`
   - не уменьшать перед удалением фона.
3. `DEFAULT_PRE_RESIZE_ALIGN = 16`
   - округление размеров до шага (для примера 608x416 -> 304x208).

Коды выхода:
1. `0` - успешно (или в пакетном режиме без ошибок).
2. `1` - ошибка обработки файла / в пакетном режиме есть ошибки.
3. `2` - ошибка аргументов, путей, инициализации модели и т.д.
"""

import argparse
import io
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    from PIL import Image, ImageFilter, ImageStat
except Exception:
    Image = None
    ImageFilter = None
    ImageStat = None

from rembg import new_session, remove

# =========================
# ПУТИ ПО УМОЛЧАНИЮ (Windows)
# =========================
DEFAULT_INPUT_PATH = Path(r"d:\results\size&back")
DEFAULT_OUTPUT_PATH = Path(r"d:\results\size&back_result")

# =========================
# НАСТРОЙКИ ПО УМОЛЧАНИЮ
# =========================
DEFAULT_MODEL = "birefnet-general"   # можно заменить на birefnet-portrait для людей
DEFAULT_ALPHA_MATTING = False         # как remove.bg — лучше включить
DEFAULT_AF = 240                     # foreground threshold (0..255)
DEFAULT_AB = 10                      # background threshold (0..255)
DEFAULT_AE = 10                      # erode size (>=0)
DEFAULT_PROFILE = "auto"
DEFAULT_FALLBACK_MODE = "off"
DEFAULT_MIN_FG_COVERAGE = 0.005
DEFAULT_MAX_FG_COVERAGE = 0.98
DEFAULT_MIN_QUALITY_SCORE = 0.55
DEFAULT_EDGE_REFINE_RADIUS = 1.0
DEFAULT_DECONTAMINATE = "light"
DEFAULT_MAX_INFERENCE_PIXELS = 16_000_000
DEFAULT_ALPHA_MATTING_MAX_PIXELS = 12_000_000
# Предварительное уменьшение всех входных изображений перед удалением фона.
# Варианты:
#   - число (например 300): уменьшать так, чтобы большая сторона была ~300 px
#   - None или ""           : не уменьшать
DEFAULT_PRE_RESIZE_MAX_SIDE = 300
# Для "красивых" размеров можно выравнивать стороны до шага (например 16):
# пример 608x416 при цели 300 -> 304x208.
DEFAULT_PRE_RESIZE_ALIGN = 16
DECONTAMINATE_STRENGTH = {
    "off": 0.0,
    "light": 0.35,
    "medium": 0.60,
    "strong": 0.85,
}

# Профили -> приоритетные модели
PROFILE_PRIMARY_MODEL = {
    "general": "birefnet-general",
    "portrait": "birefnet-portrait",
    "product": "bria-rmbg",
    "graphic": "birefnet-general",
    "animal": "birefnet-general",
}
PROFILE_FALLBACK_MODEL = {
    "general": "bria-rmbg",
    "portrait": "birefnet-general",
    "product": "birefnet-general",
    "graphic": "bria-rmbg",
    "animal": "bria-rmbg",
}


def parse_bgcolor(value: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    """
    Accepts:
      - None -> None (transparent)
      - "#RRGGBB" -> (R,G,B,255)
      - "#RRGGBBAA" -> (R,G,B,A)
      - "R,G,B" -> (R,G,B,255)
      - "R,G,B,A" -> (R,G,B,A)
    """
    if value is None:
        return None
    v = value.strip()
    if v.lower() in ("none", "transparent", "trans"):
        return None

    if v.startswith("#"):
        hexv = v[1:]
        if len(hexv) == 6:
            r = int(hexv[0:2], 16)
            g = int(hexv[2:4], 16)
            b = int(hexv[4:6], 16)
            return (r, g, b, 255)
        if len(hexv) == 8:
            r = int(hexv[0:2], 16)
            g = int(hexv[2:4], 16)
            b = int(hexv[4:6], 16)
            a = int(hexv[6:8], 16)
            return (r, g, b, a)
        raise ValueError("Hex background color must be #RRGGBB or #RRGGBBAA")

    parts = [p.strip() for p in v.split(",")]
    if len(parts) not in (3, 4):
        raise ValueError("bgcolor must be '#RRGGBB', '#RRGGBBAA', 'R,G,B' or 'R,G,B,A'")
    nums = [int(p) for p in parts]
    if any(n < 0 or n > 255 for n in nums):
        raise ValueError("bgcolor channel values must be in 0..255")
    if len(nums) == 3:
        return (nums[0], nums[1], nums[2], 255)
    return (nums[0], nums[1], nums[2], nums[3])


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif")


def choose_auto_profile(inp: bytes) -> str:
    """
    Lightweight routing heuristic.
    Falls back to "general" when Pillow is unavailable.
    """
    if Image is None:
        return "general"
    try:
        with Image.open(io.BytesIO(inp)) as img:
            w, h = img.size
            if w == 0 or h == 0:
                return "general"

            small = img.convert("RGB")
            small.thumbnail((256, 256))
            s_mean = ImageStat.Stat(small.convert("HSV")).mean[1]

            quant = small.convert("P", palette=Image.ADAPTIVE, colors=32)
            used_colors = sum(1 for c in quant.histogram() if c > 0)

            # Графика: мало цветов / логотипы / скриншоты.
            if used_colors <= 14:
                return "graphic"
            # Портрет: чаще вертикальный кадр.
            if h / w >= 1.18:
                return "portrait"
            # Товар: чаще "каталожные" кадры с умеренной насыщенностью.
            if (0.8 <= (w / h) <= 1.25) and s_mean < 70:
                return "product"
    except Exception:
        return "general"
    return "general"


def resolve_profile(inp: bytes, profile: str) -> str:
    if profile == "auto":
        return choose_auto_profile(inp)
    return profile


def build_model_chain(
    profile: str,
    model_override: str,
    fallback_model_override: Optional[str],
) -> Tuple[str, ...]:
    """
    Returns ordered unique model chain:
      1) primary model
      2) fallback model
    """
    if profile not in PROFILE_PRIMARY_MODEL:
        profile = "general"

    primary = model_override if model_override != DEFAULT_MODEL else PROFILE_PRIMARY_MODEL[profile]
    fallback = fallback_model_override or PROFILE_FALLBACK_MODEL[profile]

    chain = []
    for m in (primary, fallback):
        if m and m not in chain:
            chain.append(m)
    return tuple(chain)


def get_session(session_cache: Dict[str, object], model_name: str):
    session = session_cache.get(model_name)
    if session is None:
        session = new_session(model_name)
        session_cache[model_name] = session
    return session


def run_remove(
    inp: bytes,
    session_cache: Dict[str, object],
    model_name: str,
    alpha_matting: bool,
    af: int,
    ab: int,
    ae: int,
    only_mask: bool,
    post_process_mask: bool,
    bgcolor: Optional[Tuple[int, int, int, int]],
) -> bytes:
    session = get_session(session_cache, model_name)
    return remove(
        inp,
        session=session,
        alpha_matting=alpha_matting,
        alpha_matting_foreground_threshold=af,
        alpha_matting_background_threshold=ab,
        alpha_matting_erode_size=ae,
        only_mask=only_mask,
        post_process_mask=post_process_mask,
        bgcolor=bgcolor,
        force_return_bytes=True,
    )


def _is_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    if isinstance(exc, MemoryError):
        return True
    return ("unable to allocate" in msg) or ("out of memory" in msg)


def run_remove_safe(
    inp: bytes,
    session_cache: Dict[str, object],
    model_name: str,
    alpha_matting: bool,
    af: int,
    ab: int,
    ae: int,
    only_mask: bool,
    post_process_mask: bool,
    bgcolor: Optional[Tuple[int, int, int, int]],
) -> bytes:
    """
    Runs rembg remove with automatic OOM fallback:
      1) try requested alpha_matting mode
      2) on OOM, retry once with alpha_matting=False
    """
    try:
        return run_remove(
            inp=inp,
            session_cache=session_cache,
            model_name=model_name,
            alpha_matting=alpha_matting,
            af=af,
            ab=ab,
            ae=ae,
            only_mask=only_mask,
            post_process_mask=post_process_mask,
            bgcolor=bgcolor,
        )
    except Exception as e:
        if alpha_matting and _is_oom_error(e):
            return run_remove(
                inp=inp,
                session_cache=session_cache,
                model_name=model_name,
                alpha_matting=False,
                af=af,
                ab=ab,
                ae=ae,
                only_mask=only_mask,
                post_process_mask=post_process_mask,
                bgcolor=bgcolor,
            )
        raise


def evaluate_quality(out_bytes: bytes, only_mask: bool) -> Optional[Dict[str, float]]:
    """
    Evaluates output mask quality from alpha channel.
    Returns None when image has no alpha and not in only-mask mode.
    """
    if Image is None:
        return None
    try:
        with Image.open(io.BytesIO(out_bytes)) as im:
            if only_mask:
                alpha = im.convert("L")
            else:
                rgba = im.convert("RGBA")
                if "A" not in rgba.getbands():
                    return None
                alpha = rgba.getchannel("A")
    except Exception:
        return None

    hist = alpha.histogram()
    total = sum(hist)
    if total == 0:
        return None
    fg = sum(hist[9:])
    edge = sum(hist[9:245])

    fg_coverage = fg / total
    edge_ratio = edge / max(fg, 1)
    score = 1.0

    if fg_coverage < 0.005:
        score -= 0.8
    elif fg_coverage < 0.02:
        score -= 0.3
    if fg_coverage > 0.985:
        score -= 0.8
    elif fg_coverage > 0.95:
        score -= 0.3

    if 0.02 < fg_coverage < 0.95 and edge_ratio < 0.01:
        score -= 0.15
    if edge_ratio > 0.75:
        score -= 0.1

    score = max(0.0, min(1.2, score))
    return {"score": score, "fg_coverage": fg_coverage, "edge_ratio": edge_ratio}


def should_try_fallback(
    metrics: Optional[Dict[str, float]],
    min_fg_coverage: float,
    max_fg_coverage: float,
    min_quality_score: float,
) -> bool:
    if metrics is None:
        return False
    fg = metrics["fg_coverage"]
    score = metrics["score"]
    return fg < min_fg_coverage or fg > max_fg_coverage or score < min_quality_score


def pick_best_result(
    primary_bytes: bytes,
    primary_metrics: Optional[Dict[str, float]],
    fallback_bytes: bytes,
    fallback_metrics: Optional[Dict[str, float]],
) -> Tuple[bytes, bool]:
    """
    Returns selected bytes and whether fallback won.
    """
    if primary_metrics is not None and fallback_metrics is not None:
        if fallback_metrics["score"] > primary_metrics["score"]:
            return fallback_bytes, True
        return primary_bytes, False
    if fallback_metrics is not None and primary_metrics is None:
        return fallback_bytes, True
    return primary_bytes, False


def _lanczos_resample():
    if Image is None:
        return None
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return Image.LANCZOS


def _normalize_optional_positive_int(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip()
        if v == "":
            return None
        iv = int(v)
    else:
        iv = int(value)
    if iv <= 0:
        return None
    return iv


def _scaled_size_max_side(w: int, h: int, max_side: int, align: int) -> Tuple[int, int]:
    if w <= 0 or h <= 0 or max_side <= 0:
        return w, h
    long_side = max(w, h)
    if long_side <= max_side:
        return w, h

    scale = max_side / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    if align > 1:
        new_w = max(1, int(round(new_w / align) * align))
        new_h = max(1, int(round(new_h / align) * align))
    return new_w, new_h


def pre_resize_input(
    inp_bytes: bytes,
    max_side_value,
    align_value: int,
) -> Tuple[bytes, Optional[Tuple[int, int]]]:
    """
    Pre-resize source image proportionally before background removal.
    Returns resized bytes and target working size (None when unchanged/unavailable).
    """
    if Image is None:
        return inp_bytes, None
    max_side = _normalize_optional_positive_int(max_side_value)
    if max_side is None:
        return inp_bytes, None
    align = max(1, int(align_value))

    try:
        with Image.open(io.BytesIO(inp_bytes)) as im:
            w, h = im.size
            new_w, new_h = _scaled_size_max_side(w, h, max_side, align)
            if (new_w, new_h) == (w, h):
                return inp_bytes, (w, h)
            lanczos = _lanczos_resample()
            if lanczos is None:
                return inp_bytes, (w, h)
            resized = im.convert("RGB").resize((new_w, new_h), lanczos)
            buf = io.BytesIO()
            resized.save(buf, format="PNG")
            return buf.getvalue(), (new_w, new_h)
    except Exception:
        return inp_bytes, None


def _get_image_size_from_bytes(image_bytes: bytes) -> Optional[Tuple[int, int]]:
    if Image is None:
        return None
    try:
        with Image.open(io.BytesIO(image_bytes)) as im:
            return im.size
    except Exception:
        return None


def resize_input_for_inference(
    inp_bytes: bytes,
    max_pixels: int,
) -> Tuple[bytes, Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """
    Returns:
      resized_input_bytes,
      original_size,
      inference_size
    If resize is not needed or Pillow unavailable, original bytes are returned.
    """
    if Image is None or max_pixels <= 0:
        return inp_bytes, None, None

    try:
        with Image.open(io.BytesIO(inp_bytes)) as im:
            w, h = im.size
            if w <= 0 or h <= 0:
                return inp_bytes, None, None
            pixels = w * h
            if pixels <= max_pixels:
                return inp_bytes, (w, h), (w, h)

            scale = (max_pixels / float(pixels)) ** 0.5
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            lanczos = _lanczos_resample()
            if lanczos is None:
                return inp_bytes, (w, h), (w, h)

            resized = im.convert("RGB").resize((new_w, new_h), lanczos)
            buf = io.BytesIO()
            resized.save(buf, format="PNG")
            return buf.getvalue(), (w, h), (new_w, new_h)
    except Exception:
        return inp_bytes, None, None


def resize_output_to_size(
    out_bytes: bytes,
    target_size: Tuple[int, int],
    only_mask: bool,
) -> bytes:
    if Image is None:
        return out_bytes
    try:
        with Image.open(io.BytesIO(out_bytes)) as im:
            lanczos = _lanczos_resample()
            if lanczos is None:
                return out_bytes
            if only_mask:
                up = im.convert("L").resize(target_size, lanczos)
                buf = io.BytesIO()
                up.save(buf, format="PNG")
                return buf.getvalue()
            up = im.convert("RGBA").resize(target_size, lanczos)
            buf = io.BytesIO()
            up.save(buf, format="PNG")
            return buf.getvalue()
    except Exception:
        return out_bytes


def _estimate_bg_color(src_rgb, alpha) -> Tuple[int, int, int]:
    src = src_rgb.load()
    a_im = alpha.load()
    w, h = src_rgb.size
    bg_samples = 0
    sr = 0
    sg = 0
    sb = 0
    for y in range(h):
        for x in range(w):
            if a_im[x, y] <= 6:
                r, g, b = src[x, y]
                sr += r
                sg += g
                sb += b
                bg_samples += 1
                if bg_samples >= 30000:
                    break
        if bg_samples >= 30000:
            break
    if not bg_samples:
        stat = ImageStat.Stat(src_rgb)
        return (int(stat.mean[0]), int(stat.mean[1]), int(stat.mean[2]))
    return (int(sr / bg_samples), int(sg / bg_samples), int(sb / bg_samples))


def postprocess_rgba(
    out_bytes: bytes,
    inp_bytes: bytes,
    only_mask: bool,
    edge_refine_radius: float,
    decontaminate: str,
) -> bytes:
    if only_mask:
        return out_bytes
    if Image is None:
        return out_bytes

    strength = DECONTAMINATE_STRENGTH.get(decontaminate, 0.0)
    if edge_refine_radius <= 0 and strength <= 0:
        return out_bytes

    try:
        with Image.open(io.BytesIO(out_bytes)) as out_im:
            rgba = out_im.convert("RGBA")
        with Image.open(io.BytesIO(inp_bytes)) as src_im:
            src_rgb = src_im.convert("RGB")
    except Exception:
        return out_bytes

    if rgba.size != src_rgb.size:
        lanczos = _lanczos_resample()
        if lanczos is None:
            return out_bytes
        src_rgb = src_rgb.resize(rgba.size, lanczos)

    alpha = rgba.getchannel("A")
    if edge_refine_radius > 0 and ImageFilter is not None:
        smoothed = alpha.filter(ImageFilter.GaussianBlur(radius=edge_refine_radius))
        strong_fg = alpha.point(lambda a: 255 if a >= 250 else 0)
        strong_bg = alpha.point(lambda a: 255 if a <= 5 else 0)
        smoothed.paste(255, mask=strong_fg)
        smoothed.paste(0, mask=strong_bg)
        alpha = smoothed
        rgba.putalpha(alpha)

    if strength > 0:
        bg_r, bg_g, bg_b = _estimate_bg_color(src_rgb, alpha)
        px = rgba.load()
        w, h = rgba.size
        for y in range(h):
            for x in range(w):
                r, g, b, a = px[x, y]
                if 6 < a < 235:
                    wf = strength * (1.0 - (a / 255.0))
                    nr = int(max(0, min(255, r + (r - bg_r) * wf)))
                    ng = int(max(0, min(255, g + (g - bg_g) * wf)))
                    nb = int(max(0, min(255, b + (b - bg_b) * wf)))
                    px[x, y] = (nr, ng, nb, a)

    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    return buf.getvalue()


def process_one(
    in_path: Path,
    out_path: Path,
    session_cache: Dict[str, object],
    alpha_matting: bool,
    af: int,
    ab: int,
    ae: int,
    only_mask: bool,
    post_process_mask: bool,
    bgcolor: Optional[Tuple[int, int, int, int]],
    profile: str,
    model_override: str,
    fallback_model_override: Optional[str],
    fallback_mode: str,
    min_fg_coverage: float,
    max_fg_coverage: float,
    min_quality_score: float,
    edge_refine_radius: float,
    decontaminate: str,
    max_inference_pixels: int,
    alpha_matting_max_pixels: int,
    pre_resize_max_side,
    pre_resize_align: int,
) -> Tuple[str, str, bool]:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("rb") as f:
        inp = f.read()

    # 1) Предуменьшаем исходник (по настройке вверху файла).
    inp_working, working_size = pre_resize_input(
        inp_bytes=inp,
        max_side_value=pre_resize_max_side,
        align_value=pre_resize_align,
    )
    if working_size is None:
        working_size = _get_image_size_from_bytes(inp_working)

    # 2) При необходимости уменьшаем еще для безопасности по памяти (только для инференса).
    inp_for_inference, _, inference_size = resize_input_for_inference(
        inp_bytes=inp_working,
        max_pixels=max_inference_pixels,
    )
    # Для очень больших изображений alpha matting может вызвать OOM.
    effective_alpha_matting = alpha_matting
    if alpha_matting_max_pixels > 0:
        if inference_size is None:
            # Если не удалось безопасно определить размер, лучше отключить matting.
            effective_alpha_matting = False
        elif inference_size[0] * inference_size[1] > alpha_matting_max_pixels:
            effective_alpha_matting = False

    resolved_profile = resolve_profile(inp_for_inference, profile)
    model_chain = build_model_chain(resolved_profile, model_override, fallback_model_override)
    primary_model = model_chain[0]

    primary_bytes = run_remove_safe(
        inp=inp_for_inference,
        session_cache=session_cache,
        model_name=primary_model,
        alpha_matting=effective_alpha_matting,
        af=af,
        ab=ab,
        ae=ae,
        only_mask=only_mask,
        post_process_mask=post_process_mask,
        bgcolor=bgcolor,
    )
    primary_metrics = evaluate_quality(primary_bytes, only_mask=only_mask)
    final_bytes = primary_bytes
    selected_model = primary_model
    fallback_used = False

    has_fallback = len(model_chain) > 1 and fallback_mode != "off"
    run_fallback = False
    if has_fallback:
        if fallback_mode == "always":
            run_fallback = True
        elif fallback_mode == "smart":
            run_fallback = should_try_fallback(
                primary_metrics,
                min_fg_coverage=min_fg_coverage,
                max_fg_coverage=max_fg_coverage,
                min_quality_score=min_quality_score,
            )

    if run_fallback:
        fallback_model = model_chain[1]
        fallback_bytes = run_remove_safe(
            inp=inp_for_inference,
            session_cache=session_cache,
            model_name=fallback_model,
            alpha_matting=effective_alpha_matting,
            af=af,
            ab=ab,
            ae=ae,
            only_mask=only_mask,
            post_process_mask=post_process_mask,
            bgcolor=bgcolor,
        )
        fallback_metrics = evaluate_quality(fallback_bytes, only_mask=only_mask)
        final_bytes, fallback_won = pick_best_result(
            primary_bytes=primary_bytes,
            primary_metrics=primary_metrics,
            fallback_bytes=fallback_bytes,
            fallback_metrics=fallback_metrics,
        )
        if fallback_won:
            selected_model = fallback_model
            fallback_used = True

    final_bytes = postprocess_rgba(
        out_bytes=final_bytes,
        inp_bytes=inp_for_inference,
        only_mask=only_mask,
        edge_refine_radius=edge_refine_radius,
        decontaminate=decontaminate,
    )

    if working_size is not None and inference_size is not None and working_size != inference_size:
        final_bytes = resize_output_to_size(
            out_bytes=final_bytes,
            target_size=working_size,
            only_mask=only_mask,
        )

    with out_path.open("wb") as f:
        f.write(final_bytes)
    return resolved_profile, selected_model, fallback_used


def resolve_single_output(in_path: Path, out_path: Path, raw_output_arg: str) -> Path:
    """
    Resolve output path for single input file.
    Rules:
      - existing directory -> write <dir>/<input_stem>.png
      - path ending with slash -> write <dir>/<input_stem>.png
      - non-existing path without suffix -> treat as directory
      - otherwise -> treat as explicit output file path
    """
    if out_path.exists() and out_path.is_dir():
        return out_path / f"{in_path.stem}.png"
    if raw_output_arg.endswith(("\\", "/")):
        return out_path / f"{in_path.stem}.png"
    if (not out_path.exists()) and (out_path.suffix == ""):
        return out_path / f"{in_path.stem}.png"
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="High-quality background removal (rembg + BiRefNet + alpha matting)."
    )

    # Теперь пути не обязательные — берутся из констант сверху
    parser.add_argument(
        "input",
        nargs="?",
        default=str(DEFAULT_INPUT_PATH),
        help=f"Input file or folder (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Output file or folder (default: {DEFAULT_OUTPUT_PATH})",
    )

    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help="Model name (e.g. birefnet-general, birefnet-portrait, birefnet-massive, bria-rmbg)",
    )
    parser.add_argument(
        "--profile",
        choices=("auto", "general", "portrait", "product", "graphic", "animal"),
        default=DEFAULT_PROFILE,
        help="Routing profile: auto/general/portrait/product/graphic/animal",
    )
    parser.add_argument(
        "--fallback-model",
        default=None,
        help="Explicit fallback model (if omitted, profile default is used)",
    )
    parser.add_argument(
        "--fallback-mode",
        choices=("off", "smart", "always"),
        default=DEFAULT_FALLBACK_MODE,
        help="Fallback strategy: off/smart/always",
    )
    parser.add_argument(
        "--min-fg-coverage",
        type=float,
        default=DEFAULT_MIN_FG_COVERAGE,
        help="Smart fallback lower bound for foreground coverage (0..1)",
    )
    parser.add_argument(
        "--max-fg-coverage",
        type=float,
        default=DEFAULT_MAX_FG_COVERAGE,
        help="Smart fallback upper bound for foreground coverage (0..1)",
    )
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=DEFAULT_MIN_QUALITY_SCORE,
        help="Smart fallback minimum quality score (0..1.2)",
    )
    parser.add_argument(
        "--max-inference-pixels",
        type=int,
        default=DEFAULT_MAX_INFERENCE_PIXELS,
        help="Auto-downscale large images before inference (0 disables)",
    )
    parser.add_argument(
        "--alpha-matting-max-pixels",
        type=int,
        default=DEFAULT_ALPHA_MATTING_MAX_PIXELS,
        help="Disable alpha matting above this pixel count (0 disables guard)",
    )

    # По умолчанию alpha matting включён, но можно выключить флагом
    parser.add_argument(
        "--no-alpha-matting",
        action="store_true",
        help="Disable alpha matting (enabled by default)",
    )

    parser.add_argument("--af", type=int, default=DEFAULT_AF, help="Alpha matting foreground threshold (0..255)")
    parser.add_argument("--ab", type=int, default=DEFAULT_AB, help="Alpha matting background threshold (0..255)")
    parser.add_argument("--ae", type=int, default=DEFAULT_AE, help="Alpha matting erode size (>=0)")
    parser.add_argument("--only-mask", action="store_true", help="Output only the mask")
    parser.add_argument("--post-process-mask", action="store_true", help="Post-process mask (cleanup)")
    parser.add_argument(
        "--edge-refine-radius",
        type=float,
        default=DEFAULT_EDGE_REFINE_RADIUS,
        help="Edge smoothing radius in pixels (0 disables)",
    )
    parser.add_argument(
        "--decontaminate",
        choices=("off", "light", "medium", "strong"),
        default=DEFAULT_DECONTAMINATE,
        help="Foreground edge decontamination strength",
    )
    parser.add_argument(
        "--bgcolor",
        default=None,
        help="Background color if you want non-transparent output: '#RRGGBB', '#RRGGBBAA', 'R,G,B' or 'R,G,B,A'",
    )

    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    alpha_matting = DEFAULT_ALPHA_MATTING and (not args.no_alpha_matting)

    if args.af < 0 or args.af > 255 or args.ab < 0 or args.ab > 255:
        print("Error: --af and --ab must be in 0..255", file=sys.stderr)
        return 2
    if args.ae < 0:
        print("Error: --ae must be >= 0", file=sys.stderr)
        return 2
    if args.edge_refine_radius < 0:
        print("Error: --edge-refine-radius must be >= 0", file=sys.stderr)
        return 2
    if not (0 <= args.min_fg_coverage <= 1 and 0 <= args.max_fg_coverage <= 1):
        print("Error: --min-fg-coverage and --max-fg-coverage must be in 0..1", file=sys.stderr)
        return 2
    if args.min_fg_coverage > args.max_fg_coverage:
        print("Error: --min-fg-coverage cannot be greater than --max-fg-coverage", file=sys.stderr)
        return 2
    if not (0 <= args.min_quality_score <= 1.2):
        print("Error: --min-quality-score must be in 0..1.2", file=sys.stderr)
        return 2
    if args.max_inference_pixels < 0:
        print("Error: --max-inference-pixels must be >= 0", file=sys.stderr)
        return 2
    if args.alpha_matting_max_pixels < 0:
        print("Error: --alpha-matting-max-pixels must be >= 0", file=sys.stderr)
        return 2

    try:
        bgcolor = parse_bgcolor(args.bgcolor)
    except Exception as e:
        print(f"Error parsing --bgcolor: {e}", file=sys.stderr)
        return 2

    if Image is None and (args.profile == "auto" or args.fallback_mode == "smart" or args.edge_refine_radius > 0 or args.decontaminate != "off"):
        print("Warning: Pillow is not available; auto-profile, smart-fallback and edge postprocessing will be limited.", file=sys.stderr)

    try:
        pre_resize_max_side = _normalize_optional_positive_int(DEFAULT_PRE_RESIZE_MAX_SIDE)
    except Exception as e:
        print(f"Error: invalid DEFAULT_PRE_RESIZE_MAX_SIDE: {e}", file=sys.stderr)
        return 2
    try:
        pre_resize_align = int(DEFAULT_PRE_RESIZE_ALIGN)
    except Exception as e:
        print(f"Error: invalid DEFAULT_PRE_RESIZE_ALIGN: {e}", file=sys.stderr)
        return 2
    if pre_resize_align < 1:
        print("Error: DEFAULT_PRE_RESIZE_ALIGN must be >= 1", file=sys.stderr)
        return 2

    if not in_path.exists():
        print(f"Error: input path does not exist: {in_path}", file=sys.stderr)
        return 2

    session_cache: Dict[str, object] = {}

    if in_path.is_file():
        out_file = resolve_single_output(in_path, out_path, str(args.output))
        try:
            resolved_profile, selected_model, fallback_used = process_one(
                in_path=in_path,
                out_path=out_file,
                session_cache=session_cache,
                alpha_matting=alpha_matting,
                af=args.af,
                ab=args.ab,
                ae=args.ae,
                only_mask=args.only_mask,
                post_process_mask=args.post_process_mask,
                bgcolor=bgcolor,
                profile=args.profile,
                model_override=args.model,
                fallback_model_override=args.fallback_model,
                fallback_mode=args.fallback_mode,
                min_fg_coverage=args.min_fg_coverage,
                max_fg_coverage=args.max_fg_coverage,
                min_quality_score=args.min_quality_score,
                edge_refine_radius=args.edge_refine_radius,
                decontaminate=args.decontaminate,
                max_inference_pixels=args.max_inference_pixels,
                alpha_matting_max_pixels=args.alpha_matting_max_pixels,
                pre_resize_max_side=pre_resize_max_side,
                pre_resize_align=pre_resize_align,
            )
        except Exception as e:
            print(f"Error processing file '{in_path}': {e}", file=sys.stderr)
            return 1
        print(
            f"Done: {in_path} -> {out_file} "
            f"[profile={resolved_profile}, model={selected_model}, fallback_used={fallback_used}]"
        )
        return 0

    if in_path.is_dir():
        out_dir = out_path
        out_dir.mkdir(parents=True, exist_ok=True)

        files = [p for p in in_path.rglob("*") if p.is_file() and is_image_file(p)]
        if not files:
            print(f"No images found in input folder: {in_path}", file=sys.stderr)
            return 1

        ok_count = 0
        fail_count = 0
        for p in files:
            rel = p.relative_to(in_path)
            # Всегда пишем PNG, чтобы сохранить alpha-канал
            target = out_dir / rel.with_suffix(".png")
            try:
                resolved_profile, selected_model, fallback_used = process_one(
                    in_path=p,
                    out_path=target,
                    session_cache=session_cache,
                    alpha_matting=alpha_matting,
                    af=args.af,
                    ab=args.ab,
                    ae=args.ae,
                    only_mask=args.only_mask,
                    post_process_mask=args.post_process_mask,
                    bgcolor=bgcolor,
                    profile=args.profile,
                    model_override=args.model,
                    fallback_model_override=args.fallback_model,
                    fallback_mode=args.fallback_mode,
                    min_fg_coverage=args.min_fg_coverage,
                    max_fg_coverage=args.max_fg_coverage,
                    min_quality_score=args.min_quality_score,
                    edge_refine_radius=args.edge_refine_radius,
                    decontaminate=args.decontaminate,
                    max_inference_pixels=args.max_inference_pixels,
                    alpha_matting_max_pixels=args.alpha_matting_max_pixels,
                    pre_resize_max_side=pre_resize_max_side,
                    pre_resize_align=pre_resize_align,
                )
                ok_count += 1
                print(
                    f"Done: {p} -> {target} "
                    f"[profile={resolved_profile}, model={selected_model}, fallback_used={fallback_used}]"
                )
            except Exception as e:
                fail_count += 1
                print(f"Error: {p} -> {target}: {e}", file=sys.stderr)

        print(f"\nProcessed {len(files)} file(s). Success: {ok_count}, Failed: {fail_count}. Output folder: {out_dir}")
        return 0 if fail_count == 0 else 1

    print(f"Error: unsupported input path type: {in_path}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
