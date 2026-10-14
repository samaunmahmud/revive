"""Generate the bundled demo photos judges can try in one click.

Each sample is a public-domain scikit-image photo aged with a seeded damage
profile, chosen to show one agent behaviour. Re-run to regenerate:

    python -m server.make_samples
"""
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.evaluate import test_images  # noqa: E402
from revive import damage  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")

# (id, source image, damage profile, title, what it shows)
SAMPLES = [
    ("astronaut-aged", "astronaut", "aged", "Faded portrait",
     "Tinted, faded, grainy and scratched. Once the tones are restored, OpenCV finds far more of the real scratches, so the repair covers them."),
    ("rocket-faded", "rocket", "faded", "Sun-faded launch",
     "A sun-faded print. The agent restores the tones and touches nothing else, because nothing else measures as damaged."),
    ("coffee-grainy", "coffee", "grainy", "Warm café table",
     "Grainy, with warm highlights but intact blacks. Is that ageing or real coffee colour? The agent asks you."),
    ("chelsea-heavy", "chelsea", "heavy", "Badly damaged cat",
     "Dense scratches and dust over a large area. The agent won't invent that much of the photo without your approval."),
    ("chelsea-clean", "chelsea", "none", "Undamaged control",
     "Nothing wrong with this one. A good restorer leaves it alone, and so does ReVive."),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    images = test_images()
    index = []
    for sid, src, profile, title, blurb in SAMPLES:
        aged = damage.age(images[src], profile, seed=42)
        # PNG, so compression doesn't add or hide damage the demo depends on
        cv2.imwrite(os.path.join(OUT, f"{sid}.png"), aged)
        index.append({"id": sid, "title": title, "description": blurb,
                      "source": src, "profile": profile})
        print(f"wrote {sid}.png  ({src}, profile={profile})")
    with open(os.path.join(OUT, "samples.json"), "w") as fh:
        json.dump(index, fh, indent=2)


if __name__ == "__main__":
    main()
