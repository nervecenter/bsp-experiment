"""
generate_training_gdbsp_bsp.py by Chris Collazo
Train a 100-epoch model on the raw training set, and generate BSP and BDBSP samples
for training in those experiments.
"""

import sys
import os
import gc

import numpy as np
import imageio as io
import pandas as pd
import tifffile as tif

import unet
import quickjson as qj
import ami_utility as amiu


test_sections = ["p177"]

validation_sections = ["p179", "p300"]

training_sections = [
    "p270", "p271", "p272", "p273",
    "p274", "p277", "p301", "p302"
]

# BSP_BLUE_TARGET = [6.96745360309041, 20.28935167969]
# BSP_GREEN_TARGET = [0.968171754124953, 14.5505361922275]

BSP_BLUE_TARGET = {
    "mean": [6.967453603090411, 20.289351679689968],
    "stdev": [6.214510647496673, 24.3832076440424]
}
BSP_GREEN_TARGET = {
    "mean": [11.278378955768614, 21.285373772823007],
    "stdev": [16.55424572971975, 23.98704044895674]
}

DEVICE = "cuda"
NUM_GPUS = 1
MAX_EPOCHS = 100
DATA_DIR = "/home/user/Data Sets/ami/actual_ami"
SNAPSHOT_EPOCHS = list(range(100, MAX_EPOCHS + 1, 100))
BATCH_SIZE = 24
GENERATE_DIR = "/home/user/Data Sets/generated"
SNAPSHOTS_DIR = f"{GENERATE_DIR}/snapshots"

if not os.path.isdir(GENERATE_DIR):
    os.mkdir(GENERATE_DIR)
if not os.path.isdir(SNAPSHOTS_DIR):
    os.mkdir(SNAPSHOTS_DIR)


#
# TRAINING
#

if os.path.isfile(f"{SNAPSHOTS_DIR}/snapshot_epoch_{MAX_EPOCHS}.pt"):
    print("Final snapshot found, skipping training...")
else:
    unet.train_ami(
        unet.make_gpu_parallel(unet.UnetSegmenter(
            num_input_channels=2,
            num_output_channels=3,
            snapshot_file=unet.find_latest_snapshot(SNAPSHOTS_DIR, SNAPSHOT_EPOCHS)[0]
        ), NUM_GPUS),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        training_sections=training_sections,
        validation_sections=validation_sections,
        data_dir=DATA_DIR,
        channels_list=[1, 2],
        preproc_list=["raw"] * 2,
        metrics_file=f"{GENERATE_DIR}/metrics_training.csv",
        snapshots_dir=SNAPSHOTS_DIR,
        snapshot_epochs=SNAPSHOT_EPOCHS,
        use_gpu=NUM_GPUS > 0
    )


final_snapshot = unet.UnetSegmenter(
    num_input_channels=2,
    num_output_channels=3,
    snapshot_file=f"{SNAPSHOTS_DIR}/snapshot_epoch_{MAX_EPOCHS}.pt"
).eval().to(DEVICE)

#
# TESTING
#

TEST_DIR = f"{GENERATE_DIR}/test"

if not os.path.isdir(TEST_DIR):
    os.mkdir(TEST_DIR)

if np.all(
    [os.path.isfile(f"{TEST_DIR}/pred_{s}.png") for s in test_sections] \
    + [os.path.isfile(f"{TEST_DIR}/metrics_testing.csv")]
):
    print("Already tested, skipping...")
else:
    test_metrics = []

    for section in test_sections:
        # Pick the highest confidence class, make class map
        _, pred_map, pred_image = final_snapshot.predict_section(np.array([
            io.imread(f"{DATA_DIR}/{section}_1_raw.png"),
            io.imread(f"{DATA_DIR}/{section}_2_raw.png")
        ]))

        # Save ensemble softmax and image, and append metrics
        print(f"Saving predictions for test section {section}...")

        io.imsave(f"{TEST_DIR}/pred_{section}.png", pred_image)

        gt = io.imread(f"{DATA_DIR}/{section}_ground_truth_2ch.png")
        dice_score = unet.dice_coefficient(pred_image, gt)

        test_metrics.append([section, dice_score])
        pd.DataFrame(
            test_metrics, columns=["section", "dice_score"]
        ).to_csv(f"{TEST_DIR}/metrics_testing.csv")

        del pred_map, pred_image
        gc.collect()


#
# ADJUSTING
#


PRE_PROC_DIR = f"{GENERATE_DIR}/gdbsp"

if not os.path.isdir(PRE_PROC_DIR):
    os.mkdir(PRE_PROC_DIR)

semdists_file = f"{PRE_PROC_DIR}/semantic_distances.csv"
semdists_header = ["section", "nuclear stain", "nuclear stain w ah", "lectin-488", "Lectin-488 w ah"]
semdists_types = {
    "section": str,
    "nuclear stain": float, "nuclear stain w ah": float,
    "lectin-488": float, "Lectin-488 w ah": float,
}
semdists = pd.read_csv(semdists_file, dtype=semdists_types).values.tolist() \
    if os.path.isfile(semdists_file) else []

for section in (training_sections + validation_sections + test_sections):

    semdists_entry = semdists[-1] if semdists[-1][0] == section else [section, None, None, None, None]

    for adaptive_histeq in (False, True):

        blue_image_file = f"{PRE_PROC_DIR}/{section}_1_gdbsp{'_ah' if adaptive_histeq else ''}.png"
        green_image_file = f"{PRE_PROC_DIR}/{section}_2_gdbsp{'_ah' if adaptive_histeq else ''}.png"
        new_gt_file = f"{PRE_PROC_DIR}/{section}_ground_truth_2ch.png"

        if os.path.isfile(blue_image_file) and os.path.isfile(green_image_file):
            print(f"Already processed section {section} {'with' if adaptive_histeq else 'without'} adaptive histeq.")
            continue

        print(f"Adjusting section {section} using gdbsp {'with' if adaptive_histeq else 'without'} adaptive histeq...")

        adjusted_blue, adjusted_green, _, _, prediction_image, blue_log, green_log = amiu.adjust_section_gdbsp(
            io.imread(f"{DATA_DIR}/{section}_1_raw.png"),
            io.imread(f"{DATA_DIR}/{section}_2_raw.png"),
            BSP_BLUE_TARGET,
            BSP_GREEN_TARGET,
            final_snapshot,
            adaptive_histeq=adaptive_histeq
        )

        if not adaptive_histeq:
            semdists_entry[1], semdists_entry[3] = blue_log[-1], green_log[-1]
        else:
            semdists_entry[2], semdists_entry[4] = blue_log[-1], green_log[-1]


        amiu.save_gdbsp_semdist_log_plots(
            PRE_PROC_DIR,
            f"{section} Nuclear Stain GDBSP Log{' with Adaptive Histeq' if adaptive_histeq else ''}",
            blue_log
        )
        amiu.save_gdbsp_semdist_log_plots(
            PRE_PROC_DIR,
            f"{section} Lectin-488 GDBSP Log{' with Adaptive Histeq' if adaptive_histeq else ''}",
            green_log
        )

        # Save adjusted images
        io.imsave(blue_image_file, adjusted_blue)
        io.imsave(green_image_file, adjusted_green)
        io.imsave(new_gt_file, prediction_image)

        del adjusted_blue, adjusted_green
        gc.collect()

        if not adaptive_histeq:
            semdists.append(semdists_entry)
        else:
            semdists[-1] = semdists_entry

        pd.DataFrame(semdists).to_csv(semdists_file, header=semdists_header, index=False)


print("Done.")
