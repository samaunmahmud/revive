"""Action tools: OpenCV 5 repairs the agent can call.

Each tool takes an image (+ parameters) and returns a new image.
Tools never decide anything — the agent does.
"""
import cv2
import numpy as np


def inpaint_scratches(img, mask, radius: int = 3, method: str = "telea"):
    """Fill scratch/dust pixels from their surroundings."""
    flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    return cv2.inpaint(img, mask, radius, flag)


def denoise(img, strength: float = 6.0):
    """Non-local means: removes grain while keeping edges. strength ~3 (light) .. 15 (heavy)."""
    s = float(strength)
    return cv2.fastNlMeansDenoisingColored(img, None, s, s, 7, 21)


def enhance_contrast(img, clip: float = 2.0):
    """Restore faded tones.

    1) stretch lightness to the full range (undoes uniform fading),
    2) CLAHE for local contrast. Applied to L only so colours don't shift.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    lo, hi = np.percentile(L, [0.5, 99.5])
    L = np.clip((L - lo) * 255.0 / max(hi - lo, 1.0), 0, 255).astype(np.uint8)
    if clip > 0:
        L = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(8, 8)).apply(L)
    lab[:, :, 0] = L
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def correct_colour(img, strength: float = 1.0):
    """Remove a yellow/sepia cast ("white patch" assumption in LAB space).

    Highlights in an un-aged photo are close to neutral, so the tint measured
    in the brightest 10% of pixels is taken as the cast and subtracted from the
    a (green-red) and b (blue-yellow) channels. Lightness is untouched.
    strength 0..1 controls how much of the cast is removed.
    """
    s = float(np.clip(strength, 0.0, 1.0))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    sel = lab[:, :, 0] >= np.percentile(lab[:, :, 0], 90)
    offset = np.array([lab[:, :, 1][sel].mean() - 128.0, lab[:, :, 2][sel].mean() - 128.0])
    cast = float(np.hypot(*offset))
    # Keep a little natural warmth: real highlights are rarely perfectly neutral,
    # and removing ALL of the tint turns warm subjects (skin, fur) grey.
    natural = 8.0
    if cast <= natural:
        return img.copy()
    shift = offset * (1.0 - natural / cast) * s
    for i, ch in enumerate((1, 2)):
        lab[:, :, ch] -= shift[i]
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def sharpen(img, amount: float = 0.6):
    """Unsharp mask, for photos left soft after denoising."""
    blur = cv2.GaussianBlur(img, (0, 0), 1.5)
    return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)


# Registry the agent sees: name -> (function, parameter schema, description)
TOOLS = {
    "inpaint_scratches": {
        "fn": inpaint_scratches,
        "params": {"radius": "int 2-7", "method": "'telea' or 'ns'"},
        "desc": "Fill detected scratches and dust spots. Use when scratch_coverage > 0.0045.",
    },
    "denoise": {
        "fn": denoise,
        "params": {"strength": "float 3-15"},
        "desc": "Remove film grain / noise. Use when noise_sigma > 6. Too strong blurs faces.",
    },
    "enhance_contrast": {
        "fn": enhance_contrast,
        "params": {"clip": "float 0-4"},
        "desc": "Restore faded tones. Use when black_point > 0.15 (blacks lifted to grey). clip=0 is a pure tone stretch.",
    },
    "correct_colour": {
        "fn": correct_colour,
        "params": {"strength": "float 0-1"},
        "desc": "Remove yellow/sepia cast. Use only when highlight_cast > 8 AND black_point > 0.15; a cast with normal blacks may be the real scene colour, so ask_human instead.",
    },
    "sharpen": {
        "fn": sharpen,
        "params": {"amount": "float 0.2-1.5"},
        "desc": "Recover detail lost to softness. Use sparingly, after denoise.",
    },
}
