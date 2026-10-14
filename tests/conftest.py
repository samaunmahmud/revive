"""Shared fixtures: small seeded test photos so the suite runs in seconds."""
import os
import sys
import tempfile

# job files from API tests go to a throwaway folder, not data/jobs (read at import time)
os.environ.setdefault("REVIVE_DATA_DIR", tempfile.mkdtemp(prefix="revive-test-"))

import cv2  # noqa: E402
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.evaluate import test_images  # noqa: E402
from revive import damage  # noqa: E402


def _small(img, side=320):
    h, w = img.shape[:2]
    s = side / max(h, w)
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


@pytest.fixture(scope="session")
def clean():
    """Clean public-domain photos (scikit-image), downscaled for speed."""
    return {name: _small(img) for name, img in test_images().items()}


@pytest.fixture(scope="session")
def aged():
    """aged(name, profile) -> seeded damaged version of a clean photo (full size, as in the eval)."""
    full = test_images()
    return lambda name, profile: damage.age(full[name], profile, seed=42)


@pytest.fixture
def quiet():
    return lambda *_: None


@pytest.fixture(scope="session")
def flat_grey():
    return np.full((200, 300, 3), 128, np.uint8)
