"""Region map (regions.yaml + label image) and the Vietnamese command parser, without ROS.

regions.yaml is written by segment_map; the label image stores the region id of every map cell
(0 = not in a region), so the region at a map-frame point is a pixel lookup.
"""
import math
import os
import re
import unicodedata

import cv2
import yaml

# Vietnamese names of the letters (after normalize(): no tones, đ -> d).
LETTER_SOUNDS = {
    "A": ["a"], "B": ["be", "bo", "bi"], "C": ["xe", "ce", "co", "xi"], "D": ["de", "do"],
    "E": ["e"], "F": ["ep", "ef", "phe"], "G": ["go", "gie", "ge"], "H": ["hat", "ho", "hach", "ech"],
    "I": ["i"], "J": ["gi", "zi"], "K": ["ca", "ka"], "L": ["lo", "el", "e lo"], "M": ["mo", "em", "em mo"],
    "N": ["no", "en", "en no"], "O": ["o"], "P": ["pe", "po"], "Q": ["quy", "cu", "qui"], "R": ["ro", "e ro", "rot"],
    "S": ["et", "so", "ech si"], "T": ["te", "to"], "U": ["u"], "V": ["ve", "vo"], "W": ["vep", "dap liu", "dup liu"],
    "X": ["ich", "ich xi"], "Y": ["y", "i dai", "oai"], "Z": ["zet", "det", "giet", "zed"],
}
PLACE_WORDS = ("khu vuc", "khu", "vung", "phong", "nha", "san", "bai", "cho", "toa")
STOP_WORDS = ("dung lai", "dung", "huy", "thoi", "stop")
FROM_WORD = r"tu"
TO_WORDS = r"(?:den|toi|sang|qua|ve)"
FILLER = r"(?:robot|hay|vui long|xin|di chuyen|di|chay|toi muon|muon|nhe|nao|dia diem)"


def normalize(text):
    """Lower case, no tone marks or punctuation: 'Đi đến Phòng thí nghiệm.' -> 'di den phong thi nghiem'."""
    t = unicodedata.normalize("NFD", text.lower().replace("đ", "d").replace("Đ", "d"))
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return " ".join(t.split())


class Region:
    def __init__(self, d):
        self.id = int(d["id"])
        self.label = str(d["label"])
        self.name = d["name"]
        self.aliases = list(d.get("aliases", []))
        self.goal = d["goal"]                    # {x, y}
        self.center = d["center"]
        self.area = float(d.get("area_m2", 0.0))
        self.polygon = d.get("polygon", [])      # [[x, y], ...] outline, map frame
        self.color = d.get("color", [0.5, 0.5, 0.5])

    @property
    def title(self):
        return f"{self.label} - {self.name}"

    def spoken_forms(self):
        """Normalized phrases that name this region: {phrase: by_letter}."""
        forms = {normalize(n): False for n in [self.name] + self.aliases}
        for s in [self.label.lower()] + LETTER_SOUNDS.get(self.label.upper(), []):
            forms.update({f"khu vuc {s}": True, f"khu {s}": True, f"vung {s}": True})
        return forms


class RegionMap:
    def __init__(self, regions_file):
        with open(regions_file) as f:
            cfg = yaml.safe_load(f)
        base = os.path.dirname(os.path.abspath(regions_file))
        self.resolution = float(cfg["resolution"])
        self.origin = cfg["origin"]
        self.labels = cv2.imread(os.path.join(base, cfg["label_image"]), cv2.IMREAD_UNCHANGED)
        self.height, self.width = self.labels.shape[:2]
        self.regions = [Region(r) for r in cfg["regions"]]
        self.by_id = {r.id: r for r in self.regions}
        # longest phrase first so "phong thi nghiem" wins over "thi nghiem"
        self.forms = sorted(((f, r, by_letter) for r in self.regions for f, by_letter in r.spoken_forms().items()),
                            key=lambda frl: -len(frl[0]))
        self.letter_only = {s: r for r in self.regions
                            for s in [r.label.lower()] + LETTER_SOUNDS.get(r.label.upper(), [])}

    # ------------------------------------------------------------ geometry
    def to_pixel(self, x, y):
        col = int(math.floor((x - self.origin[0]) / self.resolution))
        row = self.height - 1 - int(math.floor((y - self.origin[1]) / self.resolution))
        return row, col

    def region_at(self, x, y):
        row, col = self.to_pixel(x, y)
        if 0 <= row < self.height and 0 <= col < self.width:
            return self.by_id.get(int(self.labels[row, col]))
        return None

    # ------------------------------------------------------------ language
    def find(self, phrase):
        """Region named in a normalized phrase, or None."""
        p = f" {phrase} "
        for form, region, by_letter in self.forms:
            k = p.find(f" {form} ")
            if k < 0:
                continue
            # a letter must end the place: "khu vuc gi ep" is not J
            if by_letter and re.sub(rf"\b{FILLER}\b", " ", p[k + len(form) + 1:]).strip():
                continue
            return region
        words = re.sub(rf"\b{FILLER}\b", " ", phrase).split()
        if len(words) == 1 and words[0] in self.letter_only:   # "đi đến A"
            return self.letter_only[words[0]]
        return None

    def parse(self, text):
        """Command in a sentence. Returns a dict:
            {"action": "stop"}
            {"action": "go", "route": [Region, ...], "src": Region | None}
            {"action": "error", "reason": "unknown_place", "place": "phong an"}
            {"action": "error", "reason": "not_understood"}
        """
        t = normalize(text)
        if not t:
            return {"action": "error", "reason": "not_understood", "text": t}
        if any(re.search(rf"\b{w}\b", t) for w in STOP_WORDS):
            return {"action": "stop", "text": t}

        src_phrase = None
        m = re.search(rf"\b{FROM_WORD}\b\s+(?P<src>.+?)\s+\b{TO_WORDS}\b\s+(?P<dst>.+)$", t)
        if m:
            src_phrase, dst_phrase = m.group("src"), m.group("dst")
        else:
            parts = re.split(rf"\b{TO_WORDS}\b", t)
            dst_phrase = parts[-1].strip() if len(parts) > 1 and parts[-1].strip() else t

        dst = self.find(dst_phrase)
        if dst is None and dst_phrase != t:    # "... den" at the end, or the place before the verb
            dst = self.find(t)
        if dst is None:
            place = self._place_phrase(dst_phrase, after_verb=dst_phrase != t)
            if place:
                return {"action": "error", "reason": "unknown_place", "place": place, "text": t}
            return {"action": "error", "reason": "not_understood", "text": t}
        route, src = [dst], None
        if src_phrase is not None:
            src = self.find(src_phrase)
            if src is None:
                return {"action": "error", "reason": "unknown_place",
                        "place": self._place_phrase(src_phrase, after_verb=True), "text": t}
            if src is not dst:
                route = [src, dst]
        return {"action": "go", "route": route, "src": src, "text": t}

    @staticmethod
    def _place_phrase(phrase, after_verb):
        """The place a sentence names, when it is not on the map: 'di den khu vuc z' -> 'khu vuc z',
        'di den bai bien' -> 'bai bien'; None if the sentence names no place ('xin chao')."""
        m = re.search(rf"\b(?:{'|'.join(PLACE_WORDS)})\b.*$", phrase)
        if m:
            return m.group(0).strip()
        words = re.sub(rf"\b{FILLER}\b", " ", phrase).split()
        return " ".join(words) if after_verb and 0 < len(words) <= 4 else None
