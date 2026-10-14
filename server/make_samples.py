"""Generate the bundled demo photos judges can try in one click.

Synthetic samples are public-domain scikit-image photos aged with a seeded
damage profile, each chosen to show one agent behaviour. Real samples are
public-domain scans from Wikimedia Commons (eval/real_photos, see credits.json).
Re-run to regenerate:

    python -m server.make_samples
"""
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.evaluate import test_images  # noqa: E402
from revive import damage  # noqa: E402
from server.jobs import prepare  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")

# (id, source image, damage profile, title, what it shows)
SAMPLES = [
    ("astronaut-aged", "astronaut", "aged", "Faded portrait",
     "Tinted, faded, grainy and scratched. Once the tones are restored, OpenCV finds far more of the real scratches, so the repair covers them."),
    ("chelsea-heavy", "chelsea", "heavy", "Badly damaged cat",
     "Dense scratches and dust over a large area. The agent won't invent that much of the photo without your approval."),
    ("chelsea-clean", "chelsea", "none", "Undamaged control",
     "Nothing wrong with this one. A good restorer leaves it alone, and so does ReVive."),
]


# real scans: (id, file in eval/real_photos, title, what it shows)
REAL = [
    ("real-lemmens-1925", "lemmens-1925.jpg", "Real 1925 portrait",
     "A real faded, tinted print. The blacks have lifted too, so the agent treats the tint as ageing: colour first, then tones."),
    ("real-nara-1972", "nara-chemical-1972.jpg", "Real 1972 colour slide",
     "A real dye-faded slide with a magenta cast, but its blacks are intact. Is that ageing or the real light? The agent asks you."),
]
REAL_DIR = os.path.join(os.path.dirname(OUT), "..", "eval", "real_photos")


def main():
    os.makedirs(OUT, exist_ok=True)
    images = test_images()
    index = []
    credits = {c["file"]: c for c in json.load(open(os.path.join(REAL_DIR, "credits.json")))}
    real_rows = []
    for sid, fname, title, blurb in REAL:
        img = prepare(cv2.imread(os.path.join(REAL_DIR, fname)))
        cv2.imwrite(os.path.join(OUT, f"{sid}.png"), img)
        c = credits[fname]
        real_rows.append({"id": sid, "title": title, "description": blurb, "real": True,
                      "credit": f"{c['license']}, {c['date']}, Wikimedia Commons", "credit_url": c["source"]})
        print(f"wrote {sid}.png  (real scan: {fname})")
    for sid, src, profile, title, blurb in SAMPLES:
        aged = damage.age(images[src], profile, seed=42)
        # PNG, so compression doesn't add or hide damage the demo depends on
        cv2.imwrite(os.path.join(OUT, f"{sid}.png"), aged)
        index.append({"id": sid, "title": title, "description": blurb,
                      "source": src, "profile": profile, "real": False})
        print(f"wrote {sid}.png  ({src}, profile={profile})")
    index = index[:1] + real_rows + index[1:]
    with open(os.path.join(OUT, "samples.json"), "w") as fh:
        json.dump(index, fh, indent=2)


if __name__ == "__main__":
    main()
