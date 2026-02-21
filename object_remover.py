"""
Object Remover — remove objects from images using inpainting.

Controls:
  LMB (hold)      — draw mask (brush) or drag rectangle
  RMB (hold)      — pan the view (when zoomed in)
  Scroll           — brush size
  Arrows           — pan the view (when zoomed in)
  Ctrl+Scroll      — zoom in / out
  M                — toggle mode: Brush / Rectangle
  P                — clone stamp: pick source on background, then clone over objects
  T                — text overlay: add text with font / color / size selection
  E                — eyedropper: pick color from image for text
  F                — fit to screen (reset zoom)
  Enter            — run inpainting (remove object)
  Tab              — switch algorithm (TELEA / Navier-Stokes / LaMa)
  Z                — undo last stroke
  R                — reset mask
  S                — save result
  Esc              — exit

Algorithms:
  TELEA           — fast, good for simple/smooth backgrounds
  Navier-Stokes   — better for lines and edges
  LaMa (neural)   — best quality, reconstructs complex textures (grass, brick, etc.)
"""

import cv2
import numpy as np
import os
import sys
import tkinter as tk
from tkinter import filedialog, ttk, colorchooser
from urllib.request import urlretrieve
from PIL import Image, ImageDraw, ImageFont


# ──────────────────── LaMa ONNX model ────────────────────

LAMA_MODEL_URL = (
    "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"
)
LAMA_MODEL_DIR = os.path.join(os.path.expanduser("~"), ".cache", "lama_onnx")
LAMA_MODEL_PATH = os.path.join(LAMA_MODEL_DIR, "lama_fp32.onnx")
LAMA_INPUT_SIZE = 512  # LaMa ONNX model expects 512x512

_lama_session = None  # lazy-loaded ONNX session


def _download_progress(block_num, block_size, total_size):
    """Progress callback for urlretrieve."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb_done = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\rDownloading LaMa model: {mb_done:.1f} / {mb_total:.1f} MB ({pct}%)", end="", flush=True)


def ensure_lama_model():
    """Download the LaMa ONNX model if not present."""
    if os.path.exists(LAMA_MODEL_PATH):
        return
    os.makedirs(LAMA_MODEL_DIR, exist_ok=True)
    print(f"Downloading LaMa model to {LAMA_MODEL_PATH} ...")
    urlretrieve(LAMA_MODEL_URL, LAMA_MODEL_PATH, _download_progress)
    print("\nDownload complete!")


def get_lama_session():
    """Lazy-load the ONNX inference session."""
    global _lama_session
    if _lama_session is not None:
        return _lama_session

    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime is not installed. Run: pip install onnxruntime")
        return None

    ensure_lama_model()
    print("Loading LaMa model (first time may take a few seconds)...")
    _lama_session = ort.InferenceSession(
        LAMA_MODEL_PATH,
        providers=["CPUExecutionProvider"],
    )
    print("LaMa model loaded!")
    return _lama_session


def run_lama_inpainting(image_bgr, mask_gray):
    """
    Run LaMa ONNX inpainting.
    image_bgr: np.ndarray (H, W, 3) uint8 BGR
    mask_gray: np.ndarray (H, W) uint8, 255 = area to inpaint
    Returns: np.ndarray (H, W, 3) uint8 BGR
    """
    session = get_lama_session()
    if session is None:
        return image_bgr  # fallback: return original

    orig_h, orig_w = image_bgr.shape[:2]

    # Convert BGR -> RGB, normalize to [0, 1] (model expects 0-1 input, outputs 0-255)
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Resize to 512x512 for the model
    img_resized = cv2.resize(img_rgb, (LAMA_INPUT_SIZE, LAMA_INPUT_SIZE))
    mask_resized = cv2.resize(mask_gray, (LAMA_INPUT_SIZE, LAMA_INPUT_SIZE))

    # Binarize mask after resize
    mask_resized = (mask_resized > 127).astype(np.float32)

    # Prepare tensors: NCHW format
    img_tensor = np.transpose(img_resized, (2, 0, 1))[np.newaxis, ...]  # (1,3,512,512)
    mask_tensor = mask_resized[np.newaxis, np.newaxis, ...]              # (1,1,512,512)

    # Run inference
    input_name_img = session.get_inputs()[0].name
    input_name_mask = session.get_inputs()[1].name
    output_name = session.get_outputs()[0].name

    output = session.run(
        [output_name],
        {input_name_img: img_tensor, input_name_mask: mask_tensor},
    )[0]  # (1, 3, 512, 512)

    # Post-process: NCHW -> HWC, clip to 0-255 (model already outputs 0-255)
    result = output[0]  # (3, 512, 512)
    result = np.transpose(result, (1, 2, 0))  # (512, 512, 3)
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Resize back to original dimensions
    result = cv2.resize(result, (orig_w, orig_h))

    # Convert RGB -> BGR
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    # Blend: use LaMa result only inside the mask, keep original outside
    mask_full = cv2.resize(mask_gray, (orig_w, orig_h))
    mask_3ch = np.stack([mask_full, mask_full, mask_full], axis=2).astype(np.float32) / 255.0

    # Feather the mask edges for smooth blending
    mask_3ch = cv2.GaussianBlur(mask_3ch, (21, 21), 0)

    blended = (result_bgr.astype(np.float32) * mask_3ch +
               image_bgr.astype(np.float32) * (1.0 - mask_3ch))
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return blended


# ──────────────────── Global state ────────────────────

drawing = False           # currently drawing
panning = False           # currently panning (RMB)
pan_start_x = 0           # pan start mouse position
pan_start_y = 0
brush_radius = 20         # brush radius
mouse_x, mouse_y = 0, 0  # mouse position (in viewport coords)

# Drawing mode: 'brush' or 'rect'
draw_mode = 'brush'
rect_start = None         # (sx, sy) screen coords of rectangle start
rect_end = None           # (sx, sy) screen coords of rectangle end

# Clone stamp mode (replaces flat-color eyedropper)
clone_active = False            # True when in clone stamp mode
clone_phase = 'pick'            # 'pick' = set source point, 'paint' = clone-paint
clone_source = None             # (ox, oy) anchor point on original image
clone_offset = (0, 0)           # (dx, dy) offset from paint cursor to source
clone_src_snapshot = None       # snapshot of original at the moment source was picked

# Zoom & pan
zoom_level = 1.0          # zoom multiplier (1.0 = fit-to-screen)
view_offset_x = 0.0       # pan offset in original image pixels
view_offset_y = 0.0
base_scale = 1.0          # base scale to fit image on screen

ZOOM_MIN = 1.0
ZOOM_MAX = 10.0
ZOOM_STEP = 1.25          # multiply/divide per scroll step

# Viewport dimensions (fixed window size)
VIEWPORT_W = 1400
VIEWPORT_H = 850

# Inpainting algorithms
ALGORITHMS = [
    ("TELEA",          "opencv"),
    ("Navier-Stokes",  "opencv"),
    ("LaMa (neural)",  "lama"),
]
current_algo_idx = 0

# Image data
original = None       # original (full-size)
display_img = None    # display copy (viewport-sized)
mask = None           # mask (full-size)
result = None         # inpainting result

# Stroke history for undo
stroke_history = []   # list of mask / image states

# Text overlay mode
text_active = False           # True when in text placement mode
text_string = ''              # text to render
text_font_name = 'Arial'      # selected font name
text_font_size = 40           # font size in pixels
text_bold = False             # bold style
text_italic = False           # italic style
text_shadow = True            # draw drop shadow behind text
text_color_rgb = (255, 255, 255)  # RGB color for text
text_color_bgr = (255, 255, 255)  # BGR for OpenCV display
text_rendered = None          # pre-rendered text as BGRA numpy array
text_place_x = 0              # placement position in screen coords
text_place_y = 0

# Eyedropper
eyedropper_active = False     # True when sampling color from image

# Available fonts with variants: {name: {style: filename}}
# Styles: 'r' = regular, 'b' = bold, 'i' = italic, 'bi' = bold-italic
FONT_VARIANTS = {
    # ── Inter ──
    'Inter':           {'r': 'Inter_28pt-Regular.ttf',       'b': 'Inter_28pt-Bold.ttf',           'i': 'Inter_28pt-Italic.ttf',           'bi': 'Inter_28pt-BoldItalic.ttf'},
    'Inter Medium':    {'r': 'Inter_28pt-Medium.ttf',        'b': 'Inter_28pt-SemiBold.ttf',        'i': 'Inter_28pt-MediumItalic.ttf',     'bi': 'Inter_28pt-SemiBoldItalic.ttf'},
    'Inter Light':     {'r': 'Inter_28pt-Light.ttf',         'b': 'Inter_28pt-Medium.ttf',          'i': 'Inter_28pt-LightItalic.ttf',      'bi': 'Inter_28pt-MediumItalic.ttf'},
    # ── DM Sans ──
    'DM Sans':         {'r': 'DMSans-Regular.ttf',           'b': 'DMSans-Bold.ttf',                'i': 'DMSans-Italic.ttf',               'bi': 'DMSans-BoldItalic.ttf'},
    'DM Sans Medium':  {'r': 'DMSans-Medium.ttf',            'b': 'DMSans-SemiBold.ttf',            'i': 'DMSans-MediumItalic.ttf',         'bi': 'DMSans-SemiBoldItalic.ttf'},
    'DM Sans Light':   {'r': 'DMSans-Light.ttf',             'b': 'DMSans-Medium.ttf',              'i': 'DMSans-LightItalic.ttf',          'bi': 'DMSans-MediumItalic.ttf'},
    # ── Poppins ──
    'Poppins':         {'r': 'Poppins-Regular.ttf',          'b': 'Poppins-Bold.ttf',               'i': 'Poppins-Italic.ttf',              'bi': 'Poppins-BoldItalic.ttf'},
    'Poppins Medium':  {'r': 'Poppins-Medium.ttf',           'b': 'Poppins-SemiBold.ttf',           'i': 'Poppins-MediumItalic.ttf',        'bi': 'Poppins-SemiBoldItalic.ttf'},
    'Poppins Light':   {'r': 'Poppins-Light.ttf',            'b': 'Poppins-Medium.ttf',             'i': 'Poppins-LightItalic.ttf',         'bi': 'Poppins-MediumItalic.ttf'},
    # ── Outfit (no italic variant exists) ──
    'Outfit':          {'r': 'Outfit-Regular.ttf',           'b': 'Outfit-Bold.ttf',                'i': 'Outfit-Regular.ttf',              'bi': 'Outfit-Bold.ttf'},
    'Outfit Medium':   {'r': 'Outfit-Medium.ttf',            'b': 'Outfit-SemiBold.ttf',            'i': 'Outfit-Medium.ttf',               'bi': 'Outfit-SemiBold.ttf'},
    'Outfit Light':    {'r': 'Outfit-Light.ttf',             'b': 'Outfit-Medium.ttf',              'i': 'Outfit-Light.ttf',                'bi': 'Outfit-Medium.ttf'},
    # ── Bebas Neue (single weight, no italic) ──
    'Bebas Neue':      {'r': 'BebasNeue-Regular.ttf',        'b': 'BebasNeue-Regular.ttf',          'i': 'BebasNeue-Regular.ttf',           'bi': 'BebasNeue-Regular.ttf'},
    # ── Barlow Condensed ──
    'Barlow Cond':         {'r': 'BarlowCondensed-Regular.ttf',  'b': 'BarlowCondensed-Bold.ttf',       'i': 'BarlowCondensed-Italic.ttf',      'bi': 'BarlowCondensed-BoldItalic.ttf'},
    'Barlow Cond Medium':  {'r': 'BarlowCondensed-Medium.ttf',   'b': 'BarlowCondensed-SemiBold.ttf',   'i': 'BarlowCondensed-MediumItalic.ttf','bi': 'BarlowCondensed-SemiBoldItalic.ttf'},
    'Barlow Cond Light':   {'r': 'BarlowCondensed-Light.ttf',    'b': 'BarlowCondensed-Medium.ttf',     'i': 'BarlowCondensed-LightItalic.ttf', 'bi': 'BarlowCondensed-MediumItalic.ttf'},
    # ── Classic Windows fonts ──
    'Arial':           {'r': 'arial.ttf',    'b': 'arialbd.ttf',  'i': 'ariali.ttf',   'bi': 'arialbi.ttf'},
    'Times New Roman': {'r': 'times.ttf',    'b': 'timesbd.ttf',  'i': 'timesi.ttf',   'bi': 'timesbi.ttf'},
    'Calibri':         {'r': 'calibri.ttf',  'b': 'calibrib.ttf', 'i': 'calibrii.ttf', 'bi': 'calibriz.ttf'},
    'Verdana':         {'r': 'verdana.ttf',  'b': 'verdanab.ttf', 'i': 'verdanai.ttf', 'bi': 'verdanaz.ttf'},
    'Georgia':         {'r': 'georgia.ttf',  'b': 'georgiab.ttf', 'i': 'georgiai.ttf', 'bi': 'georgiaz.ttf'},
    'Trebuchet MS':    {'r': 'trebuc.ttf',   'b': 'trebucbd.ttf', 'i': 'trebucit.ttf', 'bi': 'trebucbi.ttf'},
    'Consolas':        {'r': 'consola.ttf',  'b': 'consolab.ttf', 'i': 'consolai.ttf', 'bi': 'consolaz.ttf'},
    'Comic Sans MS':   {'r': 'comic.ttf',    'b': 'comicbd.ttf',  'i': 'comici.ttf',   'bi': 'comicz.ttf'},
    'Tahoma':          {'r': 'tahoma.ttf',   'b': 'tahomabd.ttf', 'i': 'tahoma.ttf',   'bi': 'tahomabd.ttf'},
    'Segoe UI':        {'r': 'segoeui.ttf',  'b': 'segoeuib.ttf', 'i': 'segoeuii.ttf', 'bi': 'segoeuiz.ttf'},
    'Courier New':     {'r': 'cour.ttf',     'b': 'courbd.ttf',   'i': 'couri.ttf',    'bi': 'courbi.ttf'},
    'Impact':          {'r': 'impact.ttf',   'b': 'impact.ttf',   'i': 'impact.ttf',   'bi': 'impact.ttf'},
}


# ──────────────────── Utilities ────────────────────

def pick_file() -> str:
    """Open file dialog to select an image."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    filetypes = [
        ("Images", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp"),
        ("All files", "*.*"),
    ]
    path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=filetypes,
    )
    root.destroy()
    return path


def compute_base_scale(img):
    """Compute base scale so the image fits the viewport."""
    h, w = img.shape[:2]
    s = 1.0
    if w > VIEWPORT_W:
        s = min(s, VIEWPORT_W / w)
    if h > VIEWPORT_H:
        s = min(s, VIEWPORT_H / h)
    return s


def effective_scale():
    """Current total scale = base_scale * zoom_level."""
    return base_scale * zoom_level


def clamp_offset():
    """Clamp view_offset so we don't scroll past image edges."""
    global view_offset_x, view_offset_y

    es = effective_scale()
    orig_h, orig_w = original.shape[:2]

    # Visible area in original pixels
    vis_w = VIEWPORT_W / es
    vis_h = VIEWPORT_H / es

    max_ox = max(0, orig_w - vis_w)
    max_oy = max(0, orig_h - vis_h)

    view_offset_x = max(0.0, min(view_offset_x, max_ox))
    view_offset_y = max(0.0, min(view_offset_y, max_oy))


def screen_to_original(sx, sy):
    """Convert viewport screen coords to original image coords."""
    es = effective_scale()
    ox = sx / es + view_offset_x
    oy = sy / es + view_offset_y
    return int(ox), int(oy)


def build_overlay():
    """Build the display image: zoomed/panned crop with mask overlay."""
    global display_img

    es = effective_scale()
    orig_h, orig_w = original.shape[:2]

    # Determine crop region in original coords
    vis_w = VIEWPORT_W / es
    vis_h = VIEWPORT_H / es

    x1 = int(view_offset_x)
    y1 = int(view_offset_y)
    x2 = min(orig_w, int(view_offset_x + vis_w + 1))
    y2 = min(orig_h, int(view_offset_y + vis_h + 1))

    # Crop original and mask
    crop_img = original[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]

    # Red semi-transparent overlay on masked area
    red_overlay = crop_img.copy()
    red_overlay[crop_mask > 0] = [0, 0, 255]
    cv2.addWeighted(red_overlay, 0.45, crop_img, 0.55, 0, crop_img)

    # Scale crop to viewport size
    out_w = min(VIEWPORT_W, int((x2 - x1) * es))
    out_h = min(VIEWPORT_H, int((y2 - y1) * es))

    if out_w > 0 and out_h > 0:
        interp = cv2.INTER_LINEAR if es > 1.0 else cv2.INTER_AREA
        display_img = cv2.resize(crop_img, (out_w, out_h), interpolation=interp)
    else:
        display_img = np.zeros((VIEWPORT_H, VIEWPORT_W, 3), dtype=np.uint8)

    # Pad to viewport if image is smaller
    dh, dw = display_img.shape[:2]
    if dw < VIEWPORT_W or dh < VIEWPORT_H:
        padded = np.full((VIEWPORT_H, VIEWPORT_W, 3), 40, dtype=np.uint8)
        padded[:dh, :dw] = display_img
        display_img = padded


def draw_dashed_rect(img, pt1, pt2, color, thickness=1, dash_len=10, gap_len=6):
    """Draw a dashed rectangle on img."""
    x1, y1 = pt1
    x2, y2 = pt2
    # Four edges as line segments
    edges = [
        ((x1, y1), (x2, y1)),  # top
        ((x2, y1), (x2, y2)),  # right
        ((x2, y2), (x1, y2)),  # bottom
        ((x1, y2), (x1, y1)),  # left
    ]
    for (ex1, ey1), (ex2, ey2) in edges:
        dx = ex2 - ex1
        dy = ey2 - ey1
        length = int(np.hypot(dx, dy))
        if length == 0:
            continue
        ux, uy = dx / length, dy / length
        drawn = 0
        is_dash = True
        while drawn < length:
            seg = dash_len if is_dash else gap_len
            seg = min(seg, length - drawn)
            px1 = int(ex1 + ux * drawn)
            py1 = int(ey1 + uy * drawn)
            px2 = int(ex1 + ux * (drawn + seg))
            py2 = int(ey1 + uy * (drawn + seg))
            if is_dash:
                cv2.line(img, (px1, py1), (px2, py2), color, thickness, cv2.LINE_AA)
            drawn += seg
            is_dash = not is_dash


def draw_ui(img):
    """Draw brush cursor / rect preview / text preview and status bar below it."""
    vis = img.copy()
    h, w = vis.shape[:2]

    if text_active and text_rendered is not None:
        # Live text preview following the mouse cursor
        _draw_text_preview(vis, mouse_x, mouse_y)
    elif eyedropper_active:
        # Eyedropper crosshair cursor with color swatch
        cx, cy = mouse_x, mouse_y
        size = 14
        cv2.line(vis, (cx - size, cy), (cx + size, cy), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(vis, (cx, cy - size), (cx, cy + size), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(vis, (cx, cy), 4, (255, 255, 255), 1, cv2.LINE_AA)
        # Show current color swatch next to cursor
        cv2.rectangle(vis, (cx + 16, cy - 10), (cx + 36, cy + 10), text_color_bgr, -1)
        cv2.rectangle(vis, (cx + 16, cy - 10), (cx + 36, cy + 10), (200, 200, 200), 1)
    elif clone_active:
        disp_r = max(1, int(brush_radius * effective_scale()))
        if clone_phase == 'pick':
            # Crosshair cursor for source picking
            cx, cy = mouse_x, mouse_y
            size = 14
            cv2.line(vis, (cx - size, cy), (cx + size, cy), (0, 255, 255), 1, cv2.LINE_AA)
            cv2.line(vis, (cx, cy - size), (cx, cy + size), (0, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(vis, (cx, cy), 6, (0, 255, 255), 1, cv2.LINE_AA)
        else:
            # Clone-paint mode — show brush outline + source marker
            cv2.circle(vis, (mouse_x, mouse_y), disp_r, (0, 255, 255), 1, cv2.LINE_AA)
            # Show where source pixels are being read from
            es = effective_scale()
            ox, oy = screen_to_original(mouse_x, mouse_y)
            src_ox = ox + clone_offset[0]
            src_oy = oy + clone_offset[1]
            # Convert source back to screen coords
            src_sx = int((src_ox - view_offset_x) * es)
            src_sy = int((src_oy - view_offset_y) * es)
            # Source crosshair (cyan, smaller)
            s = 10
            cv2.line(vis, (src_sx - s, src_sy), (src_sx + s, src_sy), (255, 255, 0), 1, cv2.LINE_AA)
            cv2.line(vis, (src_sx, src_sy - s), (src_sx, src_sy + s), (255, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(vis, (src_sx, src_sy), disp_r, (255, 255, 0), 1, cv2.LINE_AA)
    elif draw_mode == 'brush':
        # Brush cursor (scaled to current zoom)
        disp_r = max(1, int(brush_radius * effective_scale()))
        cv2.circle(vis, (mouse_x, mouse_y), disp_r, (0, 255, 0), 1)
    elif draw_mode == 'rect' and rect_start is not None and rect_end is not None:
        # Dashed rectangle preview
        draw_dashed_rect(vis, rect_start, rect_end, (0, 255, 0), thickness=2)

    # Status bar — appended BELOW the image
    algo_name = ALGORITHMS[current_algo_idx][0]
    zoom_pct = int(zoom_level * 100)
    bar_h = 36
    bar = np.full((bar_h, w, 3), 30, dtype=np.uint8)

    if text_active:
        # Color swatch
        cv2.rectangle(bar, (w - 50, 4), (w - 10, bar_h - 4), text_color_bgr, -1)
        cv2.rectangle(bar, (w - 50, 4), (w - 10, bar_h - 4), (200, 200, 200), 1)
        info = (
            f"TEXT: click to place  |  "
            f"{text_font_name} {text_font_size}px  |  "
            f"Scroll=size  T=cancel  Z=undo"
        )
    elif clone_active:
        phase_label = 'SET SOURCE (click on clean background)' if clone_phase == 'pick' else 'CLONE PAINT (draw to clone texture)'
        info = (
            f"CLONE STAMP: {phase_label}  |  "
            f"Brush: {brush_radius}px  |  "
            f"Zoom: {zoom_pct}%  |  "
            f"P=exit clone  Z=undo"
        )
    else:
        mode_label = 'Brush' if draw_mode == 'brush' else 'Rect'
        info = (
            f"Mode: {mode_label}  |  "
            f"Brush: {brush_radius}px  |  "
            f"Algo: {algo_name}  |  "
            f"Zoom: {zoom_pct}%  |  "
            f"M=mode  P=clone  T=text  E=eyedropper  Enter=inpaint  Tab=algo  F=fit  Z=undo  R=reset  S=save"
        )

    cv2.putText(
        bar, info, (10, bar_h - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA,
    )

    vis = np.vstack([vis, bar])
    return vis


# ──────────────────── Text overlay helpers ────────────────────

def _find_font(font_filename):
    """Locate a TrueType font file on Windows."""
    win_fonts = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts')
    path = os.path.join(win_fonts, font_filename)
    if os.path.exists(path):
        return path
    # Fallback: try user fonts
    user_fonts = os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Microsoft', 'Windows', 'Fonts')
    path = os.path.join(user_fonts, font_filename)
    if os.path.exists(path):
        return path
    return None


def _resolve_font_file():
    """Resolve the correct .ttf file based on font name, bold, italic."""
    variants = FONT_VARIANTS.get(text_font_name)
    if variants is None:
        # Fallback to Arial
        variants = FONT_VARIANTS['Arial']

    # Pick variant key
    if text_bold and text_italic:
        key = 'bi'
    elif text_bold:
        key = 'b'
    elif text_italic:
        key = 'i'
    else:
        key = 'r'

    fname = variants.get(key, variants['r'])
    path = _find_font(fname)
    # Fallback chain: try regular, then arial
    if path is None:
        path = _find_font(variants['r'])
    if path is None:
        path = _find_font('arial.ttf')
    return path


def _render_text_image():
    """Render the current text string as a BGRA numpy array using PIL."""
    global text_rendered
    if not text_string:
        text_rendered = None
        return

    font_file = _resolve_font_file()

    try:
        pil_font = ImageFont.truetype(font_file, text_font_size)
    except Exception:
        pil_font = ImageFont.load_default()

    # Measure text size
    dummy_img = Image.new('RGBA', (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text_string, font=pil_font)
    # Extra padding for shadow
    shadow_off = max(2, text_font_size // 20) if text_shadow else 0
    tw = bbox[2] - bbox[0] + 4 + shadow_off
    th = bbox[3] - bbox[1] + 4 + shadow_off

    # Render text onto a transparent RGBA image
    txt_img = Image.new('RGBA', (tw, th), (0, 0, 0, 0))
    draw = ImageDraw.Draw(txt_img)
    text_x = -bbox[0] + 2
    text_y = -bbox[1] + 2

    # Draw shadow first (dark, slightly offset, semi-transparent)
    if text_shadow:
        shadow_img = Image.new('RGBA', (tw, th), (0, 0, 0, 0))
        sdraw = ImageDraw.Draw(shadow_img)
        sdraw.text((text_x + shadow_off, text_y + shadow_off),
                   text_string, font=pil_font, fill=(0, 0, 0, 160))
        # Blur the shadow for softness
        shadow_arr = np.array(shadow_img)
        blur_k = max(3, shadow_off * 2 + 1) | 1
        shadow_arr = cv2.GaussianBlur(shadow_arr, (blur_k, blur_k), 0)
        shadow_img = Image.fromarray(shadow_arr)
        txt_img = Image.alpha_composite(txt_img, shadow_img)
        draw = ImageDraw.Draw(txt_img)

    # Draw main text
    r, g, b = text_color_rgb
    draw.text((text_x, text_y), text_string, font=pil_font, fill=(r, g, b, 255))

    # Convert RGBA PIL -> BGRA numpy
    arr = np.array(txt_img)
    # RGBA -> BGRA
    text_rendered = arr[:, :, [2, 1, 0, 3]]


def _draw_text_preview(vis, sx, sy):
    """Composite pre-rendered text onto the viewport at (sx, sy)."""
    if text_rendered is None:
        return
    th, tw = text_rendered.shape[:2]
    vh, vw = vis.shape[:2]

    # Clip to viewport
    dx1 = sx
    dy1 = sy
    dx2 = sx + tw
    dy2 = sy + th

    src_x1 = max(0, -dx1)
    src_y1 = max(0, -dy1)
    dx1 = max(0, dx1)
    dy1 = max(0, dy1)
    dx2 = min(vw, dx2)
    dy2 = min(vh, dy2)
    src_x2 = src_x1 + (dx2 - dx1)
    src_y2 = src_y1 + (dy2 - dy1)

    if dx1 >= dx2 or dy1 >= dy2:
        return

    patch = text_rendered[src_y1:src_y2, src_x1:src_x2]
    alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
    bgr = patch[:, :, :3].astype(np.float32)
    dst = vis[dy1:dy2, dx1:dx2].astype(np.float32)
    vis[dy1:dy2, dx1:dx2] = np.clip(dst * (1.0 - alpha) + bgr * alpha, 0, 255).astype(np.uint8)


def _stamp_text_on_original(sx, sy):
    """Permanently stamp the text onto the original image."""
    global original, result
    if text_rendered is None:
        return

    # Convert screen coords to original image coords
    es = effective_scale()
    ox = sx / es + view_offset_x
    oy = sy / es + view_offset_y

    th, tw = text_rendered.shape[:2]
    # Scale text to original image scale (text was rendered at screen size)
    # We need to place it at original resolution
    orig_tw = int(tw / es)
    orig_th = int(th / es)

    if orig_tw < 1 or orig_th < 1:
        return

    # Resize text_rendered to original scale
    resized = cv2.resize(text_rendered, (orig_tw, orig_th), interpolation=cv2.INTER_AREA)

    oh, ow = original.shape[:2]
    dx1 = int(ox)
    dy1 = int(oy)
    dx2 = dx1 + orig_tw
    dy2 = dy1 + orig_th

    src_x1 = max(0, -dx1)
    src_y1 = max(0, -dy1)
    dx1 = max(0, dx1)
    dy1 = max(0, dy1)
    dx2 = min(ow, dx2)
    dy2 = min(oh, dy2)
    src_x2 = src_x1 + (dx2 - dx1)
    src_y2 = src_y1 + (dy2 - dy1)

    if dx1 >= dx2 or dy1 >= dy2:
        return

    patch = resized[src_y1:src_y2, src_x1:src_x2]
    alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
    bgr = patch[:, :, :3].astype(np.float32)
    dst = original[dy1:dy2, dx1:dx2].astype(np.float32)
    original[dy1:dy2, dx1:dx2] = np.clip(dst * (1.0 - alpha) + bgr * alpha, 0, 255).astype(np.uint8)
    result = original.copy()


def open_text_dialog():
    """Open Tkinter dialog for text input, font, size, and color selection."""
    global text_string, text_font_name, text_font_size, text_color_rgb, text_color_bgr
    global text_active, text_rendered, text_bold, text_italic, text_shadow

    root = tk.Tk()
    root.title("Add Text")
    root.attributes("-topmost", True)
    root.resizable(False, False)

    # Center on screen
    root.update_idletasks()
    rw, rh = 420, 330
    sx = (root.winfo_screenwidth() - rw) // 2
    sy = (root.winfo_screenheight() - rh) // 2
    root.geometry(f"{rw}x{rh}+{sx}+{sy}")

    result_data = {'ok': False}

    # === Text input ===
    tk.Label(root, text="Text:", font=('Segoe UI', 10)).place(x=15, y=15)
    text_var = tk.StringVar(value=text_string or 'Hello')
    text_entry = tk.Entry(root, textvariable=text_var, font=('Segoe UI', 12), width=32)
    text_entry.place(x=15, y=40)
    text_entry.focus_set()
    text_entry.select_range(0, 'end')

    # === Font selector ===
    tk.Label(root, text="Font:", font=('Segoe UI', 10)).place(x=15, y=80)
    # Filter to only available fonts
    available_fonts = []
    for name, variants in FONT_VARIANTS.items():
        if _find_font(variants['r']):
            available_fonts.append(name)
    if not available_fonts:
        available_fonts = ['Arial']

    font_var = tk.StringVar(value=text_font_name if text_font_name in available_fonts else available_fonts[0])
    font_combo = ttk.Combobox(root, textvariable=font_var, values=available_fonts,
                               state='readonly', width=25, font=('Segoe UI', 10))
    font_combo.place(x=15, y=105)

    # === Bold / Italic / Shadow checkboxes ===
    bold_var = tk.BooleanVar(value=text_bold)
    italic_var = tk.BooleanVar(value=text_italic)
    shadow_var = tk.BooleanVar(value=text_shadow)
    bold_cb = tk.Checkbutton(root, text="Bold", variable=bold_var, font=('Segoe UI', 10))
    bold_cb.place(x=15, y=138)
    italic_cb = tk.Checkbutton(root, text="Italic", variable=italic_var, font=('Segoe UI', 10))
    italic_cb.place(x=90, y=138)
    shadow_cb = tk.Checkbutton(root, text="Shadow", variable=shadow_var, font=('Segoe UI', 10))
    shadow_cb.place(x=175, y=138)

    # Font preview label
    preview_label = tk.Label(root, text="AaBbCc 123", font=('Arial', 16))
    preview_label.place(x=15, y=168)

    def update_preview(*_):
        try:
            fn = font_var.get()
            weight = 'bold' if bold_var.get() else 'normal'
            slant = 'italic' if italic_var.get() else 'roman'
            preview_label.config(font=(fn, 16, weight, slant))
        except Exception:
            pass
    font_combo.bind('<<ComboboxSelected>>', update_preview)
    bold_cb.config(command=update_preview)
    italic_cb.config(command=update_preview)
    update_preview()

    # === Size ===
    tk.Label(root, text="Size:", font=('Segoe UI', 10)).place(x=280, y=80)
    size_var = tk.IntVar(value=text_font_size)
    size_spin = tk.Spinbox(root, from_=8, to=300, textvariable=size_var, width=6,
                           font=('Segoe UI', 11))
    size_spin.place(x=280, y=105)

    # === Color picker ===
    color_hex = ['#%02x%02x%02x' % text_color_rgb]
    color_btn_frame = tk.Frame(root, bg=color_hex[0], width=80, height=30,
                               highlightbackground='gray', highlightthickness=1)
    color_btn_frame.place(x=15, y=210)

    tk.Label(root, text="Color:", font=('Segoe UI', 10)).place(x=110, y=213)

    def pick_color():
        c = colorchooser.askcolor(color=color_hex[0], title="Text Color", parent=root)
        if c and c[1]:
            color_hex[0] = c[1]
            color_btn_frame.config(bg=c[1])

    color_btn = tk.Button(root, text="Choose Color...", command=pick_color,
                          font=('Segoe UI', 9))
    color_btn.place(x=155, y=210)

    # === OK / Cancel ===
    def on_ok():
        result_data['ok'] = True
        root.destroy()

    def on_cancel():
        root.destroy()

    ok_btn = tk.Button(root, text="  OK  ", command=on_ok, font=('Segoe UI', 10),
                       width=10)
    ok_btn.place(x=100, y=290)
    cancel_btn = tk.Button(root, text="Cancel", command=on_cancel, font=('Segoe UI', 10),
                           width=10)
    cancel_btn.place(x=230, y=290)

    root.bind('<Return>', lambda e: on_ok())
    root.bind('<Escape>', lambda e: on_cancel())

    root.mainloop()

    if not result_data['ok']:
        return False

    text_string = text_var.get().strip()
    if not text_string:
        return False

    text_font_name = font_var.get()
    text_font_size = max(8, min(300, size_var.get()))
    text_bold = bold_var.get()
    text_italic = italic_var.get()
    text_shadow = shadow_var.get()

    # Parse color hex
    hex_c = color_hex[0].lstrip('#')
    r, g, b = int(hex_c[0:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16)
    text_color_rgb = (r, g, b)
    text_color_bgr = (b, g, r)

    # Pre-render the text
    _render_text_image()
    if text_rendered is None:
        return False

    text_active = True
    style_parts = []
    if text_bold: style_parts.append('Bold')
    if text_italic: style_parts.append('Italic')
    style_str = '+'.join(style_parts) if style_parts else 'Regular'
    print(f"Text mode: \"{text_string}\" ({text_font_name} {style_str}, {text_font_size}px) \u2014 click to place")
    return True


# ──────────────────── Mouse callback ────────────────────

def clone_stamp_paint(ox, oy):
    """Clone-paint pixels from source region onto (ox, oy) with soft edges."""
    global original, result
    if clone_src_snapshot is None:
        return

    h, w = original.shape[:2]
    r = brush_radius

    # Source center for this brush stamp
    src_cx = ox + clone_offset[0]
    src_cy = oy + clone_offset[1]

    # Create a soft-edge brush mask
    stamp_size = r * 2 + 1
    stamp_mask = np.zeros((stamp_size, stamp_size), dtype=np.float32)
    cv2.circle(stamp_mask, (r, r), r, 1.0, -1)
    ksize = max(3, r // 2) | 1
    stamp_mask = cv2.GaussianBlur(stamp_mask, (ksize, ksize), 0)

    # Bounding box in destination image
    dx1, dy1 = ox - r, oy - r
    dx2, dy2 = ox + r + 1, oy + r + 1
    # Bounding box in source snapshot
    sx1, sy1 = src_cx - r, src_cy - r
    sx2, sy2 = src_cx + r + 1, src_cy + r + 1

    # Clip all boxes to image bounds
    # Left/top clip
    clip_l = max(0, -dx1, -sx1)
    clip_t = max(0, -dy1, -sy1)
    # Right/bottom clip
    clip_r = max(0, dx2 - w, sx2 - w)
    clip_b = max(0, dy2 - h, sy2 - h)

    # Apply clips
    d_x1 = dx1 + clip_l;  d_y1 = dy1 + clip_t
    d_x2 = dx2 - clip_r;  d_y2 = dy2 - clip_b
    s_x1 = sx1 + clip_l;  s_y1 = sy1 + clip_t
    s_x2 = sx2 - clip_r;  s_y2 = sy2 - clip_b
    m_x1 = clip_l;        m_y1 = clip_t
    m_x2 = stamp_size - clip_r;  m_y2 = stamp_size - clip_b

    if d_x1 >= d_x2 or d_y1 >= d_y2:
        return

    # Read source pixels from the frozen snapshot (not the modified original)
    src_pixels = clone_src_snapshot[s_y1:s_y2, s_x1:s_x2].astype(np.float32)
    dst_pixels = original[d_y1:d_y2, d_x1:d_x2].astype(np.float32)
    alpha = stamp_mask[m_y1:m_y2, m_x1:m_x2, np.newaxis]

    blended = dst_pixels * (1.0 - alpha) + src_pixels * alpha
    original[d_y1:d_y2, d_x1:d_x2] = np.clip(blended, 0, 255).astype(np.uint8)
    result = original.copy()


def mouse_callback(event, x, y, flags, param):
    global drawing, panning, mouse_x, mouse_y, mask
    global pan_start_x, pan_start_y, view_offset_x, view_offset_y
    global rect_start, rect_end
    global clone_phase, clone_source, clone_offset, clone_src_snapshot, original
    global text_font_size
    global eyedropper_active, text_color_rgb, text_color_bgr

    mouse_x, mouse_y = x, y

    # ── Eyedropper mode ──
    if eyedropper_active:
        if event == cv2.EVENT_LBUTTONDOWN:
            ox, oy = screen_to_original(x, y)
            h, w = original.shape[:2]
            ox = max(0, min(w - 1, ox))
            oy = max(0, min(h - 1, oy))
            b, g, r = original[oy, ox]
            text_color_rgb = (int(r), int(g), int(b))
            text_color_bgr = (int(b), int(g), int(r))
            eyedropper_active = False
            print(f"Color picked: RGB({r}, {g}, {b}) \u2014 press T to add text with this color")
        return

    # ── Text placement mode ──
    if text_active:
        if event == cv2.EVENT_LBUTTONDOWN:
            # Stamp text onto the image
            stroke_history.append(original.copy())
            _stamp_text_on_original(x, y)
            build_overlay()
            print("Text placed! Press T for new text or Z to undo.")
            return
        # Scroll to adjust font size
        if event == cv2.EVENT_MOUSEWHEEL:
            ctrl_pressed = (flags & cv2.EVENT_FLAG_CTRLKEY) != 0
            if ctrl_pressed:
                zoom_at(x, y, up=(flags > 0))
            else:
                delta = 2 if flags > 0 else -2
                text_font_size = max(8, min(300, text_font_size + delta))
                _render_text_image()
        return

    # ── Clone stamp mode ──
    if clone_active:
        if event == cv2.EVENT_LBUTTONDOWN:
            if clone_phase == 'pick':
                # Set source anchor point on the background
                ox, oy = screen_to_original(x, y)
                clone_source = (ox, oy)
                clone_phase = 'paint'
                # Snapshot the image at this moment for clean source reads
                clone_src_snapshot = original.copy()
                print(f"Clone source set at ({ox}, {oy}) \u2014 now paint over the object")
                build_overlay()
            else:
                # Start clone-painting
                drawing = True
                stroke_history.append(original.copy())
                ox, oy = screen_to_original(x, y)
                # Offset: from paint point to source point
                clone_offset = (clone_source[0] - ox, clone_source[1] - oy)
                clone_stamp_paint(ox, oy)
                build_overlay()

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            ox, oy = screen_to_original(x, y)
            clone_stamp_paint(ox, oy)
            build_overlay()

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

        # RMB pan still works in clone mode
        elif event == cv2.EVENT_RBUTTONDOWN:
            panning = True
            pan_start_x = x
            pan_start_y = y
        elif event == cv2.EVENT_MOUSEMOVE and panning:
            es = effective_scale()
            dx = (pan_start_x - x) / es
            dy = (pan_start_y - y) / es
            view_offset_x += dx
            view_offset_y += dy
            pan_start_x = x
            pan_start_y = y
            clamp_offset()
            build_overlay()
        elif event == cv2.EVENT_RBUTTONUP:
            panning = False

        # Scroll for brush size or zoom
        elif event == cv2.EVENT_MOUSEWHEEL:
            ctrl_pressed = (flags & cv2.EVENT_FLAG_CTRLKEY) != 0
            if ctrl_pressed:
                zoom_at(x, y, up=(flags > 0))
            else:
                delta = 3 if flags > 0 else -3
                adjust_brush(delta)
        return

    # ── LMB: draw mask (brush) or rect selection ──
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        stroke_history.append(mask.copy())
        if draw_mode == 'brush':
            ox, oy = screen_to_original(x, y)
            cv2.circle(mask, (ox, oy), brush_radius, 255, -1)
            build_overlay()
        elif draw_mode == 'rect':
            rect_start = (x, y)
            rect_end = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        if draw_mode == 'brush':
            ox, oy = screen_to_original(x, y)
            cv2.circle(mask, (ox, oy), brush_radius, 255, -1)
            build_overlay()
        elif draw_mode == 'rect':
            rect_end = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if draw_mode == 'rect' and rect_start is not None:
            # Fill rectangle area into mask
            ox1, oy1 = screen_to_original(*rect_start)
            ox2, oy2 = screen_to_original(x, y)
            # Normalize corners
            rx1, ry1 = min(ox1, ox2), min(oy1, oy2)
            rx2, ry2 = max(ox1, ox2), max(oy1, oy2)
            cv2.rectangle(mask, (rx1, ry1), (rx2, ry2), 255, -1)
            rect_start = None
            rect_end = None
            build_overlay()

    # ── RMB: pan view ──
    elif event == cv2.EVENT_RBUTTONDOWN:
        panning = True
        pan_start_x = x
        pan_start_y = y

    elif event == cv2.EVENT_MOUSEMOVE and panning:
        es = effective_scale()
        dx = (pan_start_x - x) / es
        dy = (pan_start_y - y) / es
        view_offset_x += dx
        view_offset_y += dy
        pan_start_x = x
        pan_start_y = y
        clamp_offset()
        build_overlay()

    elif event == cv2.EVENT_RBUTTONUP:
        panning = False

    # ── Scroll: brush size or zoom ──
    elif event == cv2.EVENT_MOUSEWHEEL:
        ctrl_pressed = (flags & cv2.EVENT_FLAG_CTRLKEY) != 0

        if ctrl_pressed:
            # Ctrl+Scroll → zoom
            zoom_at(x, y, up=(flags > 0))
        else:
            # Scroll → brush size
            delta = 3 if flags > 0 else -3
            adjust_brush(delta)


def zoom_at(sx, sy, up=True):
    """Zoom in/out centered on the mouse position (sx, sy in screen coords)."""
    global zoom_level, view_offset_x, view_offset_y

    # Original coords under cursor before zoom
    ox_before = sx / effective_scale() + view_offset_x
    oy_before = sy / effective_scale() + view_offset_y

    # Apply zoom
    if up:
        zoom_level = min(ZOOM_MAX, zoom_level * ZOOM_STEP)
    else:
        zoom_level = max(ZOOM_MIN, zoom_level / ZOOM_STEP)

    # Adjust offset so the same original point stays under cursor
    view_offset_x = ox_before - sx / effective_scale()
    view_offset_y = oy_before - sy / effective_scale()

    clamp_offset()
    build_overlay()


def reset_zoom():
    """Reset zoom to fit-to-screen."""
    global zoom_level, view_offset_x, view_offset_y
    zoom_level = 1.0
    view_offset_x = 0.0
    view_offset_y = 0.0
    build_overlay()


PAN_STEP = 50  # pixels in original image coords per arrow key press


def pan_view(dx, dy):
    """Pan the view by (dx, dy) in original image pixels."""
    global view_offset_x, view_offset_y
    view_offset_x += dx
    view_offset_y += dy
    clamp_offset()
    build_overlay()


def adjust_brush(delta):
    global brush_radius
    brush_radius = max(2, min(300, brush_radius + delta))


def toggle_draw_mode():
    """Toggle between brush and rectangle drawing modes."""
    global draw_mode
    draw_mode = 'rect' if draw_mode == 'brush' else 'brush'
    mode_label = 'Brush' if draw_mode == 'brush' else 'Rectangle'
    print(f"Draw mode: {mode_label}")


def toggle_clone_stamp():
    """Toggle clone stamp (texture-preserving paint) mode."""
    global clone_active, clone_phase, clone_source, clone_offset, clone_src_snapshot, drawing
    if clone_active:
        clone_active = False
        clone_source = None
        clone_src_snapshot = None
        drawing = False
        print("Clone stamp OFF \u2014 back to mask mode")
    else:
        clone_active = True
        clone_phase = 'pick'
        clone_source = None
        clone_src_snapshot = None
        drawing = False
        print("Clone stamp ON \u2014 click on a clean area of the background to set source")


# ──────────────────── Inpainting ────────────────────

NUM_PASSES = 3  # number of passes for OpenCV inpainting


def dilate_mask(m, iterations=2):
    """Dilate the mask to ensure object edges are fully covered."""
    k_size = max(5, brush_radius // 2)
    if k_size % 2 == 0:
        k_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    dilated = cv2.dilate(m, kernel, iterations=iterations)
    return dilated


def run_inpainting():
    """Run inpainting with the selected algorithm."""
    global original, result

    if mask is None or np.count_nonzero(mask) == 0:
        return

    algo_name, algo_type = ALGORITHMS[current_algo_idx]

    # Dilate mask to cover object edges
    work_mask = dilate_mask(mask, iterations=3)

    if algo_type == "lama":
        # ── LaMa neural inpainting ──
        result = run_lama_inpainting(original, work_mask)
    else:
        # ── OpenCV inpainting (TELEA or NS) ──
        if algo_name == "TELEA":
            algo_flag = cv2.INPAINT_TELEA
        else:
            algo_flag = cv2.INPAINT_NS

        inpaint_radius = max(10, brush_radius * 2)

        # Multi-pass for better quality
        img = original.copy()
        for i in range(NUM_PASSES):
            img = cv2.inpaint(img, work_mask, inpaint_radius, algo_flag)
            inpaint_radius = max(5, inpaint_radius // 2)

        result = img

    # Set original = result so user can continue editing
    original = result.copy()


def save_result(source_path: str):
    """Save result next to the source file."""
    if result is None:
        return None

    folder = os.path.dirname(source_path)
    name, ext = os.path.splitext(os.path.basename(source_path))
    out_name = f"{name}_cleaned{ext}"
    out_path = os.path.join(folder, out_name)

    # If file already exists — add counter
    counter = 1
    while os.path.exists(out_path):
        out_name = f"{name}_cleaned_{counter}{ext}"
        out_path = os.path.join(folder, out_name)
        counter += 1

    cv2.imwrite(out_path, result)
    return out_path


# ──────────────────── Main loop ────────────────────

def main():
    global original, mask, display_img, result
    global current_algo_idx, brush_radius, base_scale
    global text_active, text_rendered
    global eyedropper_active

    # File selection
    image_path = pick_file()
    if not image_path:
        print("No file selected. Exiting.")
        sys.exit(0)

    # Load image
    original = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if original is None:
        print(f"Failed to open: {image_path}")
        sys.exit(1)

    print(f"Loaded: {image_path}  ({original.shape[1]}x{original.shape[0]})")

    # Scale
    base_scale = compute_base_scale(original)
    mask = np.zeros(original.shape[:2], dtype=np.uint8)
    result = None
    build_overlay()

    win_name = "Object Remover"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win_name, mouse_callback)

    while True:
        vis = draw_ui(display_img)
        cv2.imshow(win_name, vis)

        key = cv2.waitKeyEx(30)

        # ── Arrow keys → pan view ──
        if key == 2424832:    # Left arrow
            pan_view(-PAN_STEP, 0)
        elif key == 2490368:  # Up arrow
            pan_view(0, -PAN_STEP)
        elif key == 2555904:  # Right arrow
            pan_view(PAN_STEP, 0)
        elif key == 2621440:  # Down arrow
            pan_view(0, PAN_STEP)

        # ── Enter → inpainting ──
        elif key == 13:
            if np.count_nonzero(mask) > 0:
                algo_name = ALGORITHMS[current_algo_idx][0]
                print(f"Running inpainting ({algo_name})...")
                run_inpainting()
                # Reset mask and history
                mask = np.zeros(original.shape[:2], dtype=np.uint8)
                stroke_history.clear()
                build_overlay()
                print("Done! You can continue editing or press S to save.")

        # ── Tab → switch algorithm ──
        elif key == 9 or key == (9 | 0xFF):
            current_algo_idx = (current_algo_idx + 1) % len(ALGORITHMS)
            print(f"Algorithm: {ALGORITHMS[current_algo_idx][0]}")

        # ── Z → undo ──
        elif (key & 0xFF) in (ord("z"), ord("Z")):
            if stroke_history:
                prev = stroke_history.pop()
                if prev.ndim == 3:
                    # It's an image backup (text stamp or clone stroke)
                    original = prev
                    result = original.copy()
                    mask = np.zeros(original.shape[:2], dtype=np.uint8)
                else:
                    # It's a mask backup (brush / rect drawing)
                    mask = prev
                build_overlay()
                print("Undo")

        # ── R → reset mask ──
        elif (key & 0xFF) in (ord("r"), ord("R")):
            mask = np.zeros(original.shape[:2], dtype=np.uint8)
            stroke_history.clear()
            build_overlay()
            print("Mask reset")

        # ── S → save ──
        elif (key & 0xFF) in (ord("s"), ord("S")):
            if result is not None:
                saved = save_result(image_path)
                if saved:
                    print(f"Saved: {saved}")
            else:
                print("Run inpainting first (Enter)")

        # ── F → fit to screen (reset zoom) ──
        elif (key & 0xFF) in (ord("f"), ord("F")):
            reset_zoom()
            print("Zoom reset to fit")

        # ── M → toggle draw mode ──
        elif (key & 0xFF) in (ord("m"), ord("M")):
            toggle_draw_mode()

        # ── P → clone stamp (pick source & clone texture) ──
        elif (key & 0xFF) in (ord("p"), ord("P")):
            toggle_clone_stamp()

        # ── T → text overlay ──
        elif (key & 0xFF) in (ord("t"), ord("T")):
            if text_active:
                # Cancel text mode
                text_active = False
                text_rendered = None
                print("Text mode cancelled")
            else:
                eyedropper_active = False
                open_text_dialog()
                build_overlay()

        # ── E → eyedropper (pick color from image) ──
        elif (key & 0xFF) in (ord("e"), ord("E")):
            if eyedropper_active:
                eyedropper_active = False
                print("Eyedropper cancelled")
            else:
                eyedropper_active = True
                text_active = False
                text_rendered = None
                print("Eyedropper ON \u2014 click on the image to pick a color")

        # ── + / = → increase brush ──
        elif (key & 0xFF) in (ord("+"), ord("=")):
            adjust_brush(3)

        # ── - → decrease brush ──
        elif (key & 0xFF) == ord("-"):
            adjust_brush(-3)

        # ── Esc → exit ──
        elif (key & 0xFF) == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
