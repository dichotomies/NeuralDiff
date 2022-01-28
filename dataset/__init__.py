from .annotations import MaskLoader
from .rays import EPICDiff

VIDEO_IDS = [
    "P01_01",
    "P03_04",
    "P04_01",
    "P05_01",
    "P06_03",
    "P08_01",
    "P09_02",
    "P13_03",
    "P16_01",
    "P21_01",
]

# e.g. for summary video in `evaluate.py` or for debugging
SAMPLE_IDS = {
    "P01_01": 716,
    "P03_04": 702,
    "P04_01": 745,
    "P05_01": 237,
    "P06_03": 957,
    "P08_01": 217,
    "P09_02": 89,
    "P13_03": 884,
    "P16_01": 76,
    "P21_01": 238,
}