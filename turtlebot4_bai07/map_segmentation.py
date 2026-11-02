"""Phần 1-2: split the occupancy grid map into regions and give them semantic labels.

    ros2 run turtlebot4_bai07 segment_map [maps/map.yaml] [--labels config/semantic_labels.yaml]

Map Segmentation (OpenCV, distances in m converted with the map resolution):
  1. free space = cells known to be free (pgm value of a free cell; unknown 205 and walls are not),
     opened to remove specks; the distance transform gives each free cell its clearance.
  2. seeds (morphological room detection): the free space is eroded step by step (dist > T for
     growing T); a part that comes loose with an area between seed_min and seed_max is a room
     core and is kept as a seed. Narrow passages (doors, corridor necks) disappear first, so the
     cores are the wide spaces.
  3. every free cell joins the nearest seed through free space (geodesic dilation = watershed
     without crossing walls).
  4. regions smaller than min_area are merged into the neighbour they share most border with.
Semantic Mapping: each entry of semantic_labels.yaml names the region that contains its point
(label A, B, ... and a name such as "Phòng thí nghiệm"); unnamed regions get the next letters.
The goal of a region is a free cell far from obstacles (>= goal_clearance of the region's widest
clearance) and closest to the region's centre; pixel (col, row) -> map frame via map.yaml:
    x = origin_x + (col + 0.5) * resolution,  y = origin_y + (height - row - 0.5) * resolution

Writes next to the map: regions.yaml (regions with label, name, goal, centre, area, outline),
regions_label.png (region id of every cell, read by voice_navigator) and regions.png (picture).
"""
import argparse
import colorsys
import os
import sys

import cv2
import numpy as np
import yaml

from turtlebot4_bai07.regions import LETTER_SOUNDS

K3 = np.ones((3, 3), np.uint8)
FONTS = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/Library/Fonts/Arial Unicode.ttf"]


def share_file(*parts):
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory("turtlebot4_bai07"), *parts)
    except Exception:  # noqa: BLE001 - not installed: source tree
        return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), *parts)


def load_map(map_yaml):
    with open(map_yaml) as f:
        info = yaml.safe_load(f)
    image = os.path.join(os.path.dirname(os.path.abspath(map_yaml)), info["image"])
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit(f"cannot read {image}")
    occ = img.astype(np.float32) / 255.0
    if not info.get("negate", 0):
        occ = 1.0 - occ
    free = occ < float(info.get("free_thresh", 0.25))
    if info.get("mode", "trinary") == "trinary":
        free &= img != 205          # map_saver writes unknown cells as 205
    return info, img, free.astype(np.uint8)


def morphological_seeds(dist, seed_min, seed_max):
    seeds = np.zeros(dist.shape, np.int32)
    k = 0
    for t in np.arange(1.0, dist.max(), 1.0):
        mask = ((dist > t) & (seeds == 0)).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            if seed_min <= stats[i, cv2.CC_STAT_AREA] <= seed_max:
                k += 1
                seeds[lab == i] = k
    return seeds


def grow(labels, free):
    lab = labels.astype(np.float32)
    while True:
        grown = cv2.dilate(lab, K3)
        new = (lab == 0) & (free > 0) & (grown > 0)
        if not new.any():
            return lab.astype(np.int32)
        lab[new] = grown[new]


def merge_small(lab, min_area):
    while True:
        ids, areas = np.unique(lab[lab > 0], return_counts=True)
        if len(ids) == 0 or areas.min() >= min_area:
            return lab
        small = ids[np.argmin(areas)]
        mask = (lab == small).astype(np.uint8)
        ring = (cv2.dilate(mask, K3) > 0) & (mask == 0) & (lab > 0)
        nb, count = np.unique(lab[ring], return_counts=True)
        lab[lab == small] = nb[np.argmax(count)] if len(nb) else 0


def relabel(lab):
    out = np.zeros_like(lab)
    for k, i in enumerate(np.unique(lab[lab > 0]), 1):
        out[lab == i] = k
    return out


def segment(free, res, seed_min_m2, seed_max_m2, min_area_m2):
    free = cv2.morphologyEx(free, cv2.MORPH_OPEN, K3)
    # drop free specks that are not part of the explored area
    n, cc, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] * res * res < min_area_m2:
            free[cc == i] = 0
    dist = cv2.distanceTransform(free, cv2.DIST_L2, 5)
    seeds = morphological_seeds(dist, seed_min_m2 / res ** 2, seed_max_m2 / res ** 2)
    lab = merge_small(grow(seeds, free), min_area_m2 / res ** 2)
    return relabel(lab), dist, free


class Grid:
    def __init__(self, info, shape):
        self.res = float(info["resolution"])
        self.ox, self.oy = float(info["origin"][0]), float(info["origin"][1])
        self.h, self.w = shape

    def world(self, col, row):
        return self.ox + (col + 0.5) * self.res, self.oy + (self.h - row - 0.5) * self.res

    def pixel(self, x, y):
        return int((x - self.ox) / self.res), int(self.h - 1 - int((y - self.oy) / self.res))


def region_goal(mask, dist, clearance):
    ys, xs = np.nonzero(mask)
    cy, cx = ys.mean(), xs.mean()
    d = dist[ys, xs]
    ok = d >= clearance * d.max()
    j = np.argmin(np.where(ok, (xs - cx) ** 2 + (ys - cy) ** 2, np.inf))
    return xs[j], ys[j], cx, cy


def colors(n):
    return [colorsys.hsv_to_rgb((0.07 + k * 0.618034) % 1.0, 0.55, 0.95) for k in range(n)]


def draw(img, lab, regions, grid, path, scale=2):
    """regions.png: map + coloured regions, outline, goal and "A - Phòng học" labels."""
    h, w = lab.shape
    out = np.full((h, w, 3), 255, np.uint8)
    out[img == 205] = (205, 205, 205)
    for r in regions:
        rgb = np.array(r["color"]) * 255
        out[lab == r["id"]] = (0.55 * rgb + 0.45 * 255)[::-1].astype(np.uint8)
    out[img < 100] = (0, 0, 0)
    out = cv2.resize(out, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    for r in regions:
        mask = cv2.resize((lab == r["id"]).astype(np.uint8), (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, tuple(int(c * 160) for c in r["color"][::-1]), 2)
        gx, gy = grid.pixel(r["goal"]["x"], r["goal"]["y"])
        cv2.circle(out, (gx * scale, gy * scale), 6, (0, 0, 200), -1)
        cv2.circle(out, (gx * scale, gy * scale), 6, (255, 255, 255), 1)
    try:
        from PIL import Image, ImageDraw, ImageFont
        font_path = next(f for f in FONTS if os.path.exists(f))
        big, small = ImageFont.truetype(font_path, 30), ImageFont.truetype(font_path, 15)
        pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        for r in regions:
            gx, gy = grid.pixel(r["goal"]["x"], r["goal"]["y"])
            x, y = gx * scale, gy * scale
            for text, font, dy in ((r["label"], big, -40), (r["name"], small, 9)):
                tw = d.textlength(text, font=font)
                d.text((x - tw / 2, y + dy), text, font=font, fill=(20, 20, 20), stroke_width=3, stroke_fill=(255, 255, 255))
        out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except (ImportError, StopIteration, OSError):
        for r in regions:
            gx, gy = grid.pixel(r["goal"]["x"], r["goal"]["y"])
            cv2.putText(out, r["label"], (gx * scale - 9, gy * scale - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
    cv2.imwrite(path, out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("map", nargs="?", default=share_file("maps", "map.yaml"), help="map.yaml")
    ap.add_argument("--labels", default=share_file("config", "semantic_labels.yaml"), help="semantic labels")
    ap.add_argument("--out-dir", help="output directory (default: next to the map)")
    ap.add_argument("--seed-min", type=float, default=0.4, help="m², smallest room core")
    ap.add_argument("--seed-max", type=float, default=5.0, help="m², largest room core")
    ap.add_argument("--min-area", type=float, default=14.0, help="m², smaller regions are merged")
    ap.add_argument("--goal-clearance", type=float, default=0.8, help="goal clearance / widest clearance")
    args = ap.parse_args(argv)

    info, img, free = load_map(args.map)
    grid = Grid(info, img.shape)
    res = grid.res
    lab, dist, free = segment(free, res, args.seed_min, args.seed_max, args.min_area)
    n = int(lab.max())
    print(f"{args.map}: {img.shape[1]}x{img.shape[0]} cells, {res} m/cell, "
          f"free {free.sum() * res * res:.0f} m² -> {n} regions")

    with open(args.labels) as f:
        semantic = yaml.safe_load(f).get("regions", [])
    named = {}
    for s in semantic:
        col, row = grid.pixel(*s["point"])
        rid = int(lab[row, col]) if 0 <= row < grid.h and 0 <= col < grid.w else 0
        if rid == 0:   # point on a wall or outside: nearest region cell
            ys, xs = np.nonzero(lab)
            j = np.argmin((xs - col) ** 2 + (ys - row) ** 2)
            rid = int(lab[ys[j], xs[j]])
        if rid in named:
            print(f"warning: '{s['name']}' and '{named[rid]['name']}' are in the same region {rid}")
            continue
        named[rid] = s
    used = {str(s["label"]) for s in named.values()}
    spare = iter(L for L in LETTER_SOUNDS if L not in used)

    regions = []
    palette = colors(n)
    for rid in range(1, n + 1):
        mask = lab == rid
        gx, gy, cx, cy = region_goal(mask, dist, args.goal_clearance)
        s = named.get(rid, {})
        label = str(s.get("label") or next(spare))
        goal = grid.world(gx, gy) if "goal" not in s else tuple(s["goal"])
        cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outline = cv2.approxPolyDP(max(cs, key=cv2.contourArea), 1.0, True)[:, 0, :]
        regions.append({
            "id": rid, "label": label, "name": s.get("name", f"Khu vực {label}"),
            "aliases": list(s.get("aliases", [])),
            "goal": {"x": round(float(goal[0]), 2), "y": round(float(goal[1]), 2)},
            "center": dict(zip("xy", (round(float(v), 2) for v in grid.world(cx, cy)))),
            "area_m2": round(float(mask.sum() * res * res), 1),
            "clearance_m": round(float(dist[mask].max() * res), 2),
            "color": [round(c, 3) for c in palette[rid - 1]],
            "polygon": [[round(float(v), 2) for v in grid.world(c + 0.0, r + 0.0)] for c, r in outline],
        })
    regions.sort(key=lambda r: r["label"])
    # ids in label order: A = 1, B = 2, ...
    remap = np.zeros(n + 1, np.uint8)
    for k, r in enumerate(regions, 1):
        remap[r["id"]] = k
        r["id"] = k
    lab = remap[lab]

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.map))
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "regions_label.png"), lab)
    draw(img, lab, regions, grid, os.path.join(out_dir, "regions.png"))
    doc = {
        "map": os.path.basename(args.map), "label_image": "regions_label.png",
        "resolution": res, "origin": [float(v) for v in info["origin"]],
        "regions": regions,
    }
    with open(os.path.join(out_dir, "regions.yaml"), "w") as f:
        f.write("# Written by segment_map (turtlebot4_bai07): regions of the map, map frame (m).\n")
        f.write("# label_image: region id of every map cell (0 = none). goal: where Nav2 sends the robot.\n")
        yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False, default_flow_style=None, width=200)
    for r in regions:
        print(f"  {r['label']}  {r['name']:<18} area {r['area_m2']:5.1f} m²  goal ({r['goal']['x']:+.2f}, {r['goal']['y']:+.2f})")
    print(f"-> {out_dir}/regions.yaml, regions_label.png, regions.png")


if __name__ == "__main__":
    main()
