"""Synthetic ageing, so restoration can be scored against a known clean original.

Seeded: the same (image, profile, seed) always produces the same damage.
"""
import cv2
import numpy as np

PROFILES = {
    "none":      dict(fade=0.0,  cast=0.0,  noise=0,  scratches=0,  dust=0),  # control: do no harm
    "faded":    dict(fade=0.45, cast=0.0,  noise=0,  scratches=0,  dust=0),
    "sepia":     dict(fade=0.25, cast=0.6,  noise=0,  scratches=0,  dust=0),
    "grainy":    dict(fade=0.0,  cast=0.0,  noise=14, scratches=0,  dust=0),
    "scratched": dict(fade=0.0,  cast=0.0,  noise=0,  scratches=6,  dust=40),
    "aged":      dict(fade=0.35, cast=0.45, noise=10, scratches=4,  dust=25),
    "heavy":     dict(fade=0.5,  cast=0.6,  noise=16, scratches=40, dust=200),
}


def age(img: np.ndarray, profile: str, seed: int = 0) -> np.ndarray:
    p = PROFILES[profile]
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32)

    if p["fade"]:  # compress tones toward a light grey, like a sun-faded print
        out = out * (1 - p["fade"]) + 200.0 * p["fade"]
    if p["cast"]:  # warm yellow-brown tint (BGR)
        tint = np.array([0.72, 0.90, 1.08], dtype=np.float32)
        out = out * (1 - p["cast"] + p["cast"] * tint)
    if p["noise"]:
        out = out + rng.normal(0, p["noise"], out.shape).astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)

    h, w = out.shape[:2]
    for _ in range(p["scratches"]):  # thin, mostly vertical scratches (film-style)
        x0 = int(rng.integers(0, w)); y0 = int(rng.integers(0, h // 2))
        x1 = int(np.clip(x0 + rng.integers(-40, 40), 0, w - 1)); y1 = int(rng.integers(h // 2, h))
        shade = 235 if rng.random() < 0.7 else 30
        cv2.line(out, (x0, y0), (x1, y1), (shade, shade, shade), 1, cv2.LINE_AA)
    for _ in range(p["dust"]):
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        shade = 240 if rng.random() < 0.6 else 20
        cv2.circle(out, c, int(rng.integers(1, 3)), (shade, shade, shade), -1, cv2.LINE_AA)
    return out
