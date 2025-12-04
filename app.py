# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 13:10:16 2025

@author: g9cv
"""

from flask import Flask, render_template, request
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageChops
from io import BytesIO
import base64

# ======= KONSTANTER (hämtade från din Tk-app) =======

# Objektpositioner
C1 = dict(x_pct=42.0, y_pct=45.0, dia_pct=30.0)
C2 = dict(x_pct=55.0, y_pct=60.0, dia_pct=24.0)
RECT = dict(x_pct=60, y_pct=20.0, w_pct=20.0, h_pct=5.0)

IMG_SIZE = 320

GRID_EFFECTIVE_TRANSMISSION = 0.60
GRID_REVEAL_FACTOR = 1.05

OID_OFFSET_CM = 3.0
THICK_MIN, THICK_MAX = 3.0, 13.0
FIXED_SDD_CM = 180.0  # du körde på fast 180 cm i Tk-GUI:t

# ======= HJÄLPFUNKTIONER =======

def to_pil(arr: np.ndarray):
    arr8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(arr8)

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def make_base_phantom(
    size: int,
    c1_xy_pct=(C1["x_pct"], C1["y_pct"]), d1_pct=C1["dia_pct"],
    c2_xy_pct=(C2["x_pct"], C2["y_pct"]), d2_pct=C2["dia_pct"],
    rect_center_pct=(RECT["x_pct"], RECT["y_pct"]),
    rect_w_pct=RECT["w_pct"], rect_h_pct=RECT["h_pct"]
):
    """
    Basfantom (innan inversion): bakgrund + två cirklar + rektangel.
    """
    w = h = size
    base = np.linspace(0.62, 0.50, h, dtype=np.float32)
    img = np.tile(base[:, None], (1, w))
    yy, xx = np.ogrid[:h, :w]

    # C1
    c1x = int(_clamp(round(w * (c1_xy_pct[0] / 100.0)), 0, w-1))
    c1y = int(_clamp(round(h * (c1_xy_pct[1] / 100.0)), 0, h-1))
    r1  = int(max(1, round(min(w, h) * (d1_pct / 100.0) / 2.0)))
    mask1 = (yy - c1y)**2 + (xx - c1x)**2 <= r1**2
    img[mask1] *= 0.50

    # C2
    c2x = int(_clamp(round(w * (c2_xy_pct[0] / 100.0)), 0, w-1))
    c2y = int(_clamp(round(h * (c2_xy_pct[1] / 100.0)), 0, h-1))
    r2  = int(max(1, round(min(w, h) * (d2_pct / 100.0) / 2.0)))
    mask2 = (yy - c2y)**2 + (xx - c2x)**2 <= r2**2
    img[mask2] *= 0.78

    # Rektangel
    rcx = int(_clamp(round(w * (rect_center_pct[0] / 100.0)), 0, w-1))
    rcy = int(_clamp(round(h * (rect_center_pct[1] / 100.0)), 0, h-1))
    rw  = int(max(1, round(min(w, h) * (rect_w_pct / 100.0))))
    rh  = int(max(1, round(min(w, h) * (rect_h_pct / 100.0))))
    x0 = _clamp(rcx - rw//2, 0, w-1); x1 = _clamp(rcx + rw//2, 0, w)
    y0 = _clamp(rcy - rh//2, 0, h-1); y1 = _clamp(rcy + rh//2, 0, h)
    img[int(y0):int(y1), int(x0):int(x1)] *= 0.90

    return to_pil(np.clip(img, 0, 1))

def apply_contrast_energy(pil_img, kv: int):
    img = np.asarray(pil_img).astype(np.float32) / 255.0
    cf = float(np.interp(kv, [40, 150], [1.65, 0.85]))
    mid = 0.5
    img = (img - mid) * cf + mid
    return to_pil(img)

def apply_thickness_attenuation(pil_img, kv: int, thickness_cm: float):
    img = np.asarray(pil_img).astype(np.float32) / 255.0
    mu = float(np.interp(kv, [40, 150], [0.11, 0.045]))
    T = float(np.exp(-mu * max(thickness_cm, 0.0)))
    img = img * T
    return to_pil(img), T

def apply_scatter_haze(pil_img, kv: int, grid_on: bool, sdd_cm: float, thickness_cm: float):
    base = pil_img.convert("L")
    haze = float(np.interp(kv, [40, 150], [0.12, 0.24]))
    sdd_factor = float(np.interp(sdd_cm, [60.0, 150.0], [1.03, 0.92]))
    thick_factor = float(np.interp(thickness_cm, [THICK_MIN, THICK_MAX], [1.0, 1.6]))
    haze *= sdd_factor * thick_factor
    if grid_on:
        haze *= 0.25
    blur = base.filter(ImageFilter.GaussianBlur(radius=22))
    return Image.blend(base, blur, alpha=haze)

def apply_geometry_blur(pil_img, focal_mm: float, oid_cm_eff: float, sdd_cm: float, grid_on: bool):
    focus_norm = float(np.interp(focal_mm, [0.7, 1.1], [0.0, 1.0]))
    oid_norm   = float(np.interp(oid_cm_eff,  [0.0, 15.0 + OID_OFFSET_CM], [0.0, 1.0]))
    sdd_rel    = float(np.interp(sdd_cm,      [80.0, 180.0], [1.0, 0.0]))
    blur_radius = 0.12 + 2.6 * (0.15*oid_norm + 0.36*focus_norm + 0.16*sdd_rel)
    if not grid_on:
        blur_radius += 15
    else:
        blur_radius = max(0.05, blur_radius - 0.35)
    return pil_img.filter(ImageFilter.GaussianBlur(radius=blur_radius)), blur_radius

def apply_unsharp_if_grid(pil_img, grid_on: bool):
    if not grid_on:
        return pil_img
    return pil_img.filter(ImageFilter.GaussianBlur(radius=1.2))

def apply_motion_blur(pil_img: Image.Image, exp_seconds: float, enabled: bool):
    if (not enabled) or exp_seconds <= 0:
        return pil_img, 0.0

    length_px = int(float(np.interp(exp_seconds, [1.0, 5.0], [0, 25.0])))
    length_px = int(max(0, min(40, length_px)))
    if length_px <= 0:
        return pil_img, 0.0

    steps = max(2, length_px)
    acc = np.zeros((pil_img.size[1], pil_img.size[0]), dtype=np.float32)
    base = pil_img.convert("L")
    for i in range(steps):
        dx = int(round(i * 1.0))
        dy = int(round(i * 0.5))
        shifted = ImageChops.offset(base, dx, dy)
        acc += np.asarray(shifted, dtype=np.float32)

    acc /= float(steps)
    out = Image.fromarray(np.clip(acc, 0, 255).astype(np.uint8))
    extra_blur_est = 0.5 * length_px
    return out, float(extra_blur_est)

def apply_noise(pil_img, eff_mas: float, kv: int, grid_on: bool, sdd_cm: float, transmission_T: float):
    img = np.asarray(pil_img).astype(np.float32) / 255.0
    eff_mas_for_noise = eff_mas * (GRID_EFFECTIVE_TRANSMISSION if grid_on else 1.0)
    base_noise = float(np.interp(eff_mas_for_noise, [0.3, 20.0], [0.19, 0.030]))
    kv_bonus   = float(np.interp(kv, [40, 150], [1.00, 0.85]))
    sdd_bonus  = float(np.interp(sdd_cm, [80.0, 180.0], [0.95, 1.25]))
    trans_factor = 1.0 / float(np.sqrt(max(transmission_T, 1e-3)))
    noise_sigma = base_noise * kv_bonus * sdd_bonus * trans_factor
    if grid_on:
        noise_sigma *= GRID_REVEAL_FACTOR
    noise = np.random.normal(0.0, noise_sigma, img.shape).astype(np.float32)
    out = np.clip(img + noise*0.8, 0.0, 1.0)
    return to_pil(out), noise_sigma

def phantom_image():
    """Skapa (och ev. cache:a) basfantomen."""
    # Enkelt: bygg varje gång – det är billigt.
    return make_base_phantom(IMG_SIZE)

def generate_simulated_image(kv, mas, exp_time, thickness_cm, grid_on, motion_on):
    """
    Bild + stapelvärden (skarph., brus, kontrast, dos).
    Returnerar (PIL-bild, metrics-dict).
    """
    kv = int(kv)
    mas = float(mas)
    exp_time = float(exp_time)
    thickness_cm = float(thickness_cm)
    sdd_cm = FIXED_SDD_CM
    focal_mm = 1.0

    # FYSIK: mAs = mA * s
    eff_mas = mas * exp_time
    oid_eff = OID_OFFSET_CM

    # --- Bildpipeline (samma som tidigare) ---
    img = phantom_image()
    img = apply_contrast_energy(img, kv)
    img, T = apply_thickness_attenuation(img, kv, thickness_cm)
    img = apply_scatter_haze(img, kv, grid_on, sdd_cm, thickness_cm)
    img, blur_radius = apply_geometry_blur(
        img, focal_mm=focal_mm, oid_cm_eff=oid_eff, sdd_cm=sdd_cm, grid_on=grid_on
    )
    img = apply_unsharp_if_grid(img, grid_on)

    # Rörelseoskärpa
    img, motion_extra = apply_motion_blur(
        img, exp_seconds=exp_time, enabled=motion_on
    )

    img_before_noise = img.copy()

    # Brus
    img, noise_sigma = apply_noise(
        img, eff_mas, kv, grid_on, sdd_cm, transmission_T=T
    )

    # Invertera och skala till visning
    img = ImageOps.invert(img.convert("L"))
    img = img.resize((IMG_SIZE, IMG_SIZE), resample=Image.NEAREST)

    # --- Stapelvärden (samma formler som i Tk-appen) ---

    # Total oskärpa: optisk + rörelse
    blur_total = float(np.hypot(blur_radius, motion_extra))
    sharp_idx = float(np.clip(1.0 / (1.0 + (blur_total / 1.8) ** 1.4), 0.0, 1.0))

    # Brusindex
    noise_idx = float(np.clip((noise_sigma / 0.8) ** 0.75, 0.0, 1.0))

    # Kontrastindex
    img_cn = ImageOps.invert(img_before_noise.convert("L"))
    img_np_cn = np.asarray(img_cn).astype(np.float32) / 255.0
    raw_contrast = float(np.std(img_np_cn))
    contrast_idx = 0.15 + 0.35 * float(
        np.clip((raw_contrast / 0.25), 0.0, 1.0)
    )

    # Dos (relativ mot ett cap-värde)
    EFF_MAS_CAP = 40.0
    dose_rel = float(np.clip(eff_mas / EFF_MAS_CAP, 0.0, 1.0))

    metrics = {
        "sharp": sharp_idx,
        "noise": noise_idx,
        "contrast": contrast_idx,
        "dose": dose_rel,
    }

    return img, metrics


def pil_to_data_uri(img: Image.Image) -> str:
    """Gör om PIL-bild till data-URL (base64) som kan visas i <img>."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return "data:image/png;base64," + b64

# ======= FLASK-APP =======
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    # Skicka bara startsidan, själva bilden hämtas via JS /api/image
    return render_template("index.html")

@app.route("/api/image", methods=["POST"])
def api_image():
    data = request.get_json()

    kv = int(data.get("kv", 120))
    mas = float(data.get("mas", 2.0))
    exp_time = float(data.get("exp_time", 1.0))
    thickness = float(data.get("thickness", 5.0))
    grid_on = bool(data.get("grid_on", False))
    motion_on = bool(data.get("motion_on", False))

    img, metrics = generate_simulated_image(
        kv=kv,
        mas=mas,
        exp_time=exp_time,
        thickness_cm=thickness,
        grid_on=grid_on,
        motion_on=motion_on,
    )

    image_data = pil_to_data_uri(img)

    return jsonify({
        "image_data": image_data,
        "metrics": metrics
    })


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

