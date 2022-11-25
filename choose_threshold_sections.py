"""
choose_threshold_sections.py by Chris Collazo
Allows the expert to pick a threshold and extract the threshold sections for accept or reject.
"""

import sys
import os

import numpy as np
import imageio as io
import tifffile as tif

import quickjson as qj

CONFIG = qj.load_file(sys.argv[-2])
CONF_THRESHOLD = float(sys.argv[-1])
DRY_RUN = "-d" in sys.argv


version_name = "last3_epochs"

EXPERIMENT = CONFIG["experiment"]
DATA_DIR = CONFIG["data_dir"]
RUN_NAME = CONFIG["run_name"]
PRE_PROC_TYPE = CONFIG["pre_proc_type"]

os.chdir(f"/home/user/Data Sets/al_{EXPERIMENT}")

VERSION_DIR = f"{RUN_NAME}/{version_name}"
PREDICTIONS_DIR = f"{VERSION_DIR}/predictions"
ADJUSTED_DIR = f"{PREDICTIONS_DIR}/adjusted"
HIGH_CONF_DIR = f"{VERSION_DIR}/high_conf"

confidences = qj.load_file(f"{PREDICTIONS_DIR}/confidences_{RUN_NAME}_{version_name}.json")
given_conf_sections = [s for s in confidences if confidences[s] >= (CONF_THRESHOLD / 100)]

print(f"Processing {len(given_conf_sections)} sections at {CONF_THRESHOLD}% confidence for config {sys.argv[-2]}...")

if DRY_RUN:
    exit()

if not os.path.exists(HIGH_CONF_DIR):
    os.mkdir(HIGH_CONF_DIR)

qj.save_file(
    f"{HIGH_CONF_DIR}/confidences_{RUN_NAME}_{version_name}.json",
    confidences
)
qj.save_file(
    f"{HIGH_CONF_DIR}/expert_review.json",
    {s : "waiting_for_review" for s in given_conf_sections}
)


for section_name in given_conf_sections:

    print(f"Saving for section {section_name} with confidence {confidences[section_name]:0.4f}...")

    if PRE_PROC_TYPE != "raw":
        blue_adjusted = io.imread(f"{ADJUSTED_DIR}/{section_name}_1_{PRE_PROC_TYPE}.png")
        green_adjusted = io.imread(f"{ADJUSTED_DIR}/{section_name}_2_{PRE_PROC_TYPE}.png")
    else:
        blue_adjusted = io.imread(f"{DATA_DIR}/{section_name}_1_{PRE_PROC_TYPE}.png")
        green_adjusted = io.imread(f"{DATA_DIR}/{section_name}_2_{PRE_PROC_TYPE}.png")

    ground_truth = io.imread(f"{PREDICTIONS_DIR}/pred_{section_name}.png")

    # height, width = ground_truth.shape[:2]

    blue = np.dstack((blue_adjusted, blue_adjusted, blue_adjusted))
    green = np.dstack((green_adjusted, green_adjusted, green_adjusted))

    tif.imwrite(f"{HIGH_CONF_DIR}/{section_name}.tif", blue, compress=6, append=True)
    tif.imwrite(f"{HIGH_CONF_DIR}/{section_name}.tif", green, compress=6, append=True)
    tif.imwrite(f"{HIGH_CONF_DIR}/{section_name}.tif", ground_truth, compress=6, append=True)


print("Done.")
