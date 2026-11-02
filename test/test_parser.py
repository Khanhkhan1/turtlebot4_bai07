"""Command parser and region lookup on the bài 07 map (pytest, no ROS needed)."""
import os

import pytest

from turtlebot4_bai07.regions import RegionMap, normalize

REGIONS = os.path.join(os.path.dirname(__file__), "..", "maps", "regions.yaml")


@pytest.fixture(scope="module")
def rmap():
    return RegionMap(REGIONS)


def labels(cmd):
    return [r.label for r in cmd["route"]]


def test_normalize():
    assert normalize("Đi đến Phòng thí nghiệm.") == "di den phong thi nghiem"


@pytest.mark.parametrize("text, route", [
    ("Đi đến khu vực A.", ["A"]),
    ("Di chuyển từ khu vực A đến khu vực B.", ["A", "B"]),
    ("Đi đến phòng thí nghiệm.", ["G"]),
    ("đi tới thư viện", ["C"]),
    ("tôi muốn đi đến hội trường", ["H"]),
    ("từ căng tin đến nhà kho", ["I", "J"]),
    ("đi đến khu vực đê", ["D"]),
    ("khu vực xê", ["C"]),
    ("đi đến A", ["A"]),
    ("nghiệm khu vực a", ["A"]),               # Vosk sometimes hears "đi đến" as another word
    ("từ khu vực bờ tới khu vực a", ["B", "A"]),
])
def test_go(rmap, text, route):
    cmd = rmap.parse(text)
    assert cmd["action"] == "go" and labels(cmd) == route


@pytest.mark.parametrize("text, place", [
    ("Đi đến khu vực Z.", "khu vuc z"),
    ("đi đến phòng ngủ", "phong ngu"),
    ("đi đến khu vực gi ép", "khu vuc gi ep"),   # not J
    ("từ nhà bếp đến khu vực a", "nha bep"),
])
def test_unknown_place(rmap, text, place):
    cmd = rmap.parse(text)
    assert cmd["action"] == "error" and cmd["reason"] == "unknown_place" and cmd["place"] == place


def test_stop_and_noise(rmap):
    assert rmap.parse("dừng lại")["action"] == "stop"
    assert rmap.parse("xin chào")["reason"] == "not_understood"


def test_region_at(rmap):
    assert rmap.region_at(0.0, 0.0).label == "F"          # spawn point: Hành lang
    assert rmap.region_at(-7.0, 7.7).label == "A"
    for r in rmap.regions:                                  # every goal lies in its own region
        assert rmap.region_at(r.goal["x"], r.goal["y"]) is r
    assert rmap.region_at(30.0, 0.0) is None
