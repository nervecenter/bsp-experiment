"""
integrate_accepted_sections.py by Chris Collazo
Inserts accepted sections into the training set and generates a configuration for the next iteration.
"""

import sys
import os

import numpy as np
import imageio as io
import tifffile as tif

import quickjson as qj


JUST_CONFIG = "-c" in sys.argv
CONFIG = qj.load_file(sys.argv[-3])
NEW_RUN_NAME = sys.argv[-2]
SOURCE_DIR = sys.argv[-1]

DEST_DIR = "/home/user/Data Sets/ami/actual_ami/" + CONFIG["accepted_data_subdir"]
PRE_PROC_TYPE = CONFIG["pre_proc_type"]

accepted_sections = [
    section
    for section, decision in qj.load_file(f"{SOURCE_DIR}/accept_reject.json").items()
    if decision == "accept"
]

print(f"{len(accepted_sections)} accepted sections: {accepted_sections}")

CONFIG["run_name"] = NEW_RUN_NAME
CONFIG["accepted_sections"] += accepted_sections
CONFIG["active_sections"] = [s for s in CONFIG["active_sections"] if s not in accepted_sections]

assert len(CONFIG["active_sections"] + CONFIG["accepted_sections"]) == 89
assert len(set(CONFIG["active_sections"]).intersection(set(CONFIG["accepted_sections"]))) == 0

if not os.path.isdir(DEST_DIR):
    os.mkdir(DEST_DIR)

qj.save_file(f"{DEST_DIR}/config_{NEW_RUN_NAME}.json", CONFIG)

if JUST_CONFIG:
    exit()

for s, section_name in enumerate(accepted_sections, 1):

    blue_file = f"{DEST_DIR}/{section_name}_1_{PRE_PROC_TYPE}.png"
    green_file = f"{DEST_DIR}/{section_name}_2_{PRE_PROC_TYPE}.png"
    gt_file = f"{DEST_DIR}/{section_name}_ground_truth_2ch.png"

    if os.path.isfile(blue_file) and os.path.isfile(green_file) and os.path.isfile(gt_file):
        print(f"{s}/{len(accepted_sections)} Skipping already finished section {section_name}...")
        continue

    print(f"{s}/{len(accepted_sections)} Saving section {section_name}...")

    with tif.TiffFile(f"{SOURCE_DIR}/{section_name}.tif") as tfile:

        if len(tfile.pages) == 3:
            print(f"Found 3 pages, probably correct.")
            ch1 = np.squeeze(tfile.pages[0].asarray())
            ch2 = np.squeeze(tfile.pages[1].asarray())
            gt = np.squeeze(tfile.pages[2].asarray())
        else:
            print(f"Found more than 3 pages, probably incorrect. Correcting...")
            ch1 = tfile.pages[0].asarray()
            ch2 = tfile.pages[2].asarray()
            gt = tfile.pages[3].asarray()

        gt[np.all(gt == (255, 0, 255, 255), axis=-1)] = (0, 0, 0, 0)

        none = np.all(gt == (0, 0, 0, 0), axis=-1)
        normal = np.all(gt == (0, 255, 0, 255), axis=-1)
        risk = np.all(gt == (255, 255, 0, 255), axis=-1)

        erroneous = ~(none | normal | risk)

        gt[erroneous] = (0, 0, 0, 0)

        none = np.all(gt == (0, 0, 0, 0), axis=-1)
        normal = np.all(gt == (0, 255, 0, 255), axis=-1)
        risk = np.all(gt == (255, 255, 0, 255), axis=-1)

        assert np.all(none | normal | risk)

        io.imsave(blue_file, ch1[:,:,0])
        io.imsave(green_file, ch2[:,:,0])
        io.imsave(gt_file, gt[:,:,:])

print("Done.")
