"""Perception: OpenCV 5 measurements behave as the agent's decisions assume."""
import cv2
import numpy as np

from revive import perception


def test_report_has_every_measurement(clean):
    r = perception.analyse(clean["astronaut"])
    for key in ("scratch_coverage", "repair_area", "noise_sigma", "black_point", "contrast", "highlight_cast", "sharpness"):
        assert key in r
    assert "_mask" in r and r["_mask"].shape == clean["astronaut"].shape[:2]
    assert "_mask" not in perception.public(r)


def test_clean_photos_measure_as_undamaged(clean):
    for name, img in clean.items():
        r = perception.analyse(img)
        assert r["scratch_coverage"] < 0.0045, name   # below the repair trigger
        assert r["noise_sigma"] < 6.0, name           # below the denoise trigger


def test_fading_lifts_the_black_point(clean, aged):
    assert perception.analyse(aged("astronaut", "none"))["black_point"] < 0.15
    assert perception.analyse(aged("astronaut", "faded"))["black_point"] > 0.15


def test_grain_raises_noise_sigma(aged):
    assert perception.analyse(aged("rocket", "grainy"))["noise_sigma"] > \
        perception.analyse(aged("rocket", "none"))["noise_sigma"] + 5


def test_drawn_scratch_is_detected_and_confident(flat_grey):
    img = flat_grey.copy()
    cv2.line(img, (150, 10), (160, 190), (235, 235, 235), 1, cv2.LINE_AA)
    r = perception.analyse(img)
    assert r["repair_area"] > 0
    assert r["scratch_coverage"] > 0  # stands out from a flat background
    assert r["_mask"][100, 150:162].any()


def test_confident_marks_are_a_subset_of_the_repair_mask(aged):
    img = aged("chelsea", "heavy")
    r = perception.analyse(img)
    conf = perception.confident_marks(img, r["_mask"])
    assert not np.any((conf > 0) & (r["_mask"] == 0))
    assert r["repair_area"] >= r["scratch_coverage"]
