"""
active_learning.py by Chris Collazo
Trains an active learning ensemble on AMI sections, including given active
sections if provided.
Tests an active learning ensemble on specified AMI sections.
Uses trained active learning ensemble to predict maps for a specified
active set.
Uses active learning ensemble predictions to vote on plurality ensemble maps,
calculate confidence levels.
Prints metrics for the active learning run, then gathers high confidence
samples within a sub-directory in the run's directory.
"""

import shutil
import sys
import os
import gc

import torch

import numpy as np
import imageio as io
import pandas as pd

import unet
import quickjson as qj
import ami_utility as amiu


# Load the training and validation set names
NUM_GPUS = 1
CONFIG = qj.load_file(sys.argv[-1])

DEVICE = "cuda"
EXPERIMENT = CONFIG["experiment"]
FOLDS = qj.load_file(f"{EXPERIMENT}/{EXPERIMENT}_folds.json")
MAX_EPOCHS = 110
SNAPSHOT_EPOCHS = list(range(10, MAX_EPOCHS + 1, 10))
BATCH_SIZE = 24
DATA_DIR = CONFIG["data_dir"]
VERSIONS = [
    # ("all_epochs", SNAPSHOT_EPOCHS),
    ("last3_epochs", SNAPSHOT_EPOCHS[-3:])
]
# else:
#     DEVICE = "cpu"
#     FOLDS = qj.load_file("test_vc_folds.json")
#     MAX_EPOCHS = 1
#     SNAPSHOT_EPOCHS = [1]
#     BATCH_SIZE = 1
#     DATA_DIR = "/home/user/Data/ami"
#     VERSIONS = [
#         ("all_epochs", SNAPSHOT_EPOCHS)
#     ]

RUN_NAME = CONFIG["run_name"]
PRE_PROC_TYPE = CONFIG["pre_proc_type"]
ACTIVE_SECTIONS = CONFIG["active_sections"]
CONF_THRESHOLD = CONFIG["conf_threshold"]


# p177 section brightness targets
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
HISTMATCH_BLUE_TARGET = io.imread(f"{DATA_DIR}/p177_1_raw.png")
HISTMATCH_GREEN_TARGET = io.imread(f"{DATA_DIR}/p177_2_raw.png")


#
#
# TRAINING
#
#

print(f"Training for active learning {EXPERIMENT} experiment run {RUN_NAME}...")

OPTIMIZER_CONFIG = CONFIG["optimizer_config"]
SCHEDULER_CONFIG = CONFIG["scheduler_config"] if "scheduler_config" in CONFIG else None

os.chdir(f"/home/user/Data Sets/al_{EXPERIMENT}")

if not os.path.isdir(RUN_NAME):
    os.mkdir(RUN_NAME)


for fold in FOLDS:

    unet.seed_everything(2020)

    FOLD_DIR = f"{RUN_NAME}/fold_{fold}"
    SNAPSHOTS_DIR = f"{FOLD_DIR}/snapshots"

    if not os.path.isdir(FOLD_DIR):
        os.mkdir(FOLD_DIR)
    if not os.path.isdir(SNAPSHOTS_DIR):
        os.mkdir(SNAPSHOTS_DIR)

    if np.all([os.path.isfile(f"{SNAPSHOTS_DIR}/snapshot_epoch_{e}.pt") for e in SNAPSHOT_EPOCHS]):
        print(f"Already trained for {EXPERIMENT} experiment run {RUN_NAME} fold {fold}, skipping...")
        continue

    print(f"Active training for {EXPERIMENT} experiment run {RUN_NAME} fold {fold}")

    unet.train_ami(
        unet.make_gpu_parallel(unet.UnetSegmenter(
            num_input_channels=2,
            num_output_channels=3
        ), NUM_GPUS),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        optimizer_config=OPTIMIZER_CONFIG,
        scheduler_config=SCHEDULER_CONFIG,
        training_sections=FOLDS[fold]["training_sections"],
        validation_sections=FOLDS[fold]["validation_sections"],
        data_dir=DATA_DIR,
        accepted_sections=CONFIG["accepted_sections"],
        accepted_data_subdir=CONFIG["accepted_data_subdir"],
        channels_list=[1, 2],
        preproc_list=[PRE_PROC_TYPE] * 2,
        metrics_file=f"{FOLD_DIR}/metrics_training.csv",
        snapshots_dir=SNAPSHOTS_DIR,
        snapshot_epochs=SNAPSHOT_EPOCHS,
        use_gpu=NUM_GPUS > 0
    )
    torch.cuda.empty_cache()
    gc.collect()

    print(f"Done training for active learning {EXPERIMENT} experiment run {RUN_NAME} fold {fold}.")

print(f"Done training for active learning {EXPERIMENT} experiment run {RUN_NAME}.")



#
#
# TESTING
#
#

print(f"\nTesting for active learning {EXPERIMENT} experiment run {RUN_NAME}...")

for version_name, epoch_selection in VERSIONS:

    for fold in FOLDS:

        print(f"Testing for active learning {EXPERIMENT} experiment run {RUN_NAME} fold {fold} version {version_name}...\n")

        FOLD_DIR = f"{RUN_NAME}/fold_{fold}"
        SNAPSHOTS_DIR = f"{FOLD_DIR}/snapshots"
        VERSION_DIR = f"{FOLD_DIR}/{version_name}"
        TEST_DIR = f"{VERSION_DIR}/test"
        TEST_METRICS_FILE = f"{TEST_DIR}/metrics_testing.csv"

        if not os.path.isdir(FOLD_DIR):
            os.mkdir(FOLD_DIR)
        if not os.path.isdir(VERSION_DIR):
            os.mkdir(VERSION_DIR)
        if not os.path.exists(TEST_DIR):
            os.mkdir(TEST_DIR)

        test_sections = FOLDS[fold]["test_sections"]
       
        # Have snapshots vote on segmentation, calculate accuracy and confidence
        test_metrics = pd.read_csv(TEST_METRICS_FILE).values.tolist() \
            if os.path.isfile(TEST_METRICS_FILE) else []

        finished_sections = set([tm[0] for tm in test_metrics])

        if finished_sections and finished_sections == set(test_sections):
            print(f"Fold {fold} already tested, continuing...")
            continue

        for section_name in test_sections:

            if finished_sections and section_name in finished_sections:
                print(f"Already tested section {section_name} fold {fold} version {version_name}, skipping...")
                continue

            print(f"\nProcessing test section {section_name}...")

            height, width = io.imread(f"{DATA_DIR}/{section_name}_1_raw.png").shape[:2]

            # We will accumulate softmaxes for each epoch then average them for
            # the mean ensemble softmax
            ensemble_softmax = np.zeros((3, height, width), dtype=np.float32)

            # If it's a bsp run, perform BSP on the test section(s)
            if PRE_PROC_TYPE in ("gdbsp", "bsp"):

                if PRE_PROC_TYPE == "bsp":
                    adjusted_blue, adjusted_green, adjusted_softmax, _, _, _, _ = amiu.adjust_section_bsp(
                        io.imread(f"{DATA_DIR}/{section_name}_1_raw.png"),
                        io.imread(f"{DATA_DIR}/{section_name}_2_raw.png"),
                        BSP_BLUE_TARGET["mean"],
                        BSP_GREEN_TARGET["mean"],
                        unet.UnetSegmenter(
                            num_input_channels=2,
                            num_output_channels=3,
                            snapshot_file=f"{SNAPSHOTS_DIR}/snapshot_epoch_{epoch_selection[-1]}.pt"
                        ).eval().to(DEVICE)
                    )
                elif PRE_PROC_TYPE == "gdbsp":
                    adjusted_blue, adjusted_green, adjusted_softmax, _, _, _, _ = amiu.adjust_section_gdbsp(
                        io.imread(f"{DATA_DIR}/{section_name}_1_raw.png"),
                        io.imread(f"{DATA_DIR}/{section_name}_2_raw.png"),
                        BSP_BLUE_TARGET,
                        BSP_GREEN_TARGET,
                        unet.UnetSegmenter(
                            num_input_channels=2,
                            num_output_channels=3,
                            snapshot_file=f"{SNAPSHOTS_DIR}/snapshot_epoch_{epoch_selection[-1]}.pt"
                        ).eval().to(DEVICE),
                        adaptive_histeq=True
                    )

                # Save adjusted images
                io.imsave(f"{TEST_DIR}/{section_name}_1_{PRE_PROC_TYPE}.png", adjusted_blue)
                io.imsave(f"{TEST_DIR}/{section_name}_2_{PRE_PROC_TYPE}.png", adjusted_green)

                ensemble_softmax += adjusted_softmax

                del adjusted_blue, adjusted_green, adjusted_softmax
                gc.collect()
                torch.cuda.empty_cache()
    
            for epoch in epoch_selection:

                print(f"Ensemble epoch {epoch}...")

                if PRE_PROC_TYPE in ("gdbsp", "bsp") and epoch == epoch_selection[-1]:
                    print(f"Already made {PRE_PROC_TYPE.upper()} last epoch test prediction.")
                    continue

                snapshot = unet.UnetSegmenter(
                    num_input_channels=2,
                    num_output_channels=3,
                    snapshot_file=f"{SNAPSHOTS_DIR}/snapshot_epoch_{epoch}.pt"
                ).eval().to(DEVICE)

                print(f"Snapshot is on cuda: {snapshot.is_on_cuda()}")

                # If it's a raw, histeq, or histmatch run, load the existing Cohort 1 image
                if PRE_PROC_TYPE in ("gdbsp", "bsp"):
                    prediction_softmax, _, _ = snapshot.predict_section(np.array([
                        io.imread(f"{TEST_DIR}/{section_name}_1_{PRE_PROC_TYPE}.png"),
                        io.imread(f"{TEST_DIR}/{section_name}_2_{PRE_PROC_TYPE}.png")
                    ]))
                else:
                    prediction_softmax, _, _ = snapshot.predict_section(np.array([
                        io.imread(f"{DATA_DIR}/{section_name}_1_{PRE_PROC_TYPE}.png"),
                        io.imread(f"{DATA_DIR}/{section_name}_2_{PRE_PROC_TYPE}.png")
                    ]))

                ensemble_softmax += prediction_softmax

                del snapshot, prediction_softmax
                gc.collect()
                torch.cuda.empty_cache()


            # Average softmaxes
            ensemble_softmax /= len(epoch_selection)

            # Note highest confidence class as pixel confidence
            # Extract average confidence
            avg_confidence = float(np.mean(np.max(ensemble_softmax, axis=0)))
            assert 0 <= avg_confidence <= 1

            # Pick the highest confidence class, make class map
            ensemble_map = np.argmax(ensemble_softmax, axis=0)
            ensemble_image = amiu.ground_truth_map_to_image(ensemble_map)

            # Save ensemble softmax and image, and append metrics
            print(f"Saving predictions for test section {section_name}...")

            # np.savez_compressed(f"{TEST_DIR}/pred_softmax_{section_name}.npz", ensemble_softmax)
            io.imsave(f"{TEST_DIR}/pred_{section_name}.png", ensemble_image)

            gt = io.imread(f"{DATA_DIR}/{section_name}_ground_truth_2ch.png")
            dice_score = unet.dice_coefficient(ensemble_image, gt)

            test_metrics.append([section_name, dice_score, avg_confidence])
            pd.DataFrame(
                test_metrics, columns=["section_name", "dice_score", "avg_confidence"]
            ).to_csv(f"{TEST_DIR}/metrics_testing.csv", index=False)

            del ensemble_softmax, ensemble_map, ensemble_image
            gc.collect()


        print(
            "Done testing for active learning for run ",
            f"{RUN_NAME} fold {fold} version {version_name}."
        )

print(f"Done testing for active learning {EXPERIMENT} experiment run {RUN_NAME}.")



#
#
# PREDICTING & CONFIDENCE
#
#


print(f"\nPredicting for active learning {EXPERIMENT} experiment run {RUN_NAME}...")


for version_name, epoch_selection in VERSIONS:

    # Pick the worst of the four folds and predict using that
    folds_compare = []

    for fold in FOLDS:

        FOLD_DIR = f"{RUN_NAME}/fold_{fold}"
        VERSION_DIR = f"{FOLD_DIR}/{version_name}"
        TEST_DIR = f"{VERSION_DIR}/test"

        metrics = pd.read_csv(f"{TEST_DIR}/metrics_testing.csv")

        dice_conf = metrics["dice_score"].mean() + metrics["avg_confidence"].mean()
        folds_compare.append((fold, dice_conf))

    folds_compare.sort(key=lambda f: f[1])
    worst_fold, _ = folds_compare[0]

    print(f"Worst fold for {EXPERIMENT} experiment run {RUN_NAME} version {version_name} was {worst_fold}...\n")

    VERSION_DIR = f"{RUN_NAME}/{version_name}"
    PREDICTIONS_DIR = f"{VERSION_DIR}/predictions"
    MAPS_DIR = f"{PREDICTIONS_DIR}/maps"
    ADJUSTED_DIR = f"{PREDICTIONS_DIR}/adjusted"

    if not os.path.exists(VERSION_DIR):
        os.mkdir(VERSION_DIR)
    if not os.path.exists(PREDICTIONS_DIR):
        os.mkdir(PREDICTIONS_DIR)
    if not os.path.exists(MAPS_DIR):
        os.mkdir(MAPS_DIR)
    if not os.path.exists(ADJUSTED_DIR):
        os.mkdir(ADJUSTED_DIR)

    with open(f"{VERSION_DIR}/worst_fold.txt", "w") as fp:
        fp.write(worst_fold)

    print(
        f"Predicting for active learning run {EXPERIMENT} experiment {RUN_NAME}, ",
        f"version {version_name}, worst fold {worst_fold}...\n"
    )

    SNAPSHOTS_DIR = f"{RUN_NAME}/fold_{worst_fold}/snapshots"

    confidences_file = f"{PREDICTIONS_DIR}/confidences_{RUN_NAME}_{version_name}.json"

    confidences = {} if not os.path.isfile(confidences_file) \
        else qj.load_file(confidences_file)


    # Using epoch 100 snapshot to adjust section images
    for section_name in ACTIVE_SECTIONS:

        if os.path.isfile(f"{PREDICTIONS_DIR}/pred_{section_name}.png") \
        and section_name in confidences:
            print(f"Predicting and confidence already calculated for section {section_name}, skipping...")
            continue

        print(f"Predicting and confidence for section {section_name} version {version_name}...")

        section_blue = io.imread(f"{DATA_DIR}/{section_name}_1_raw.png")
        section_green = io.imread(f"{DATA_DIR}/{section_name}_2_raw.png")

        height, width = section_blue.shape[:2]
        ensemble_softmax = np.zeros((3, height, width), dtype=np.float32)


        if PRE_PROC_TYPE != "raw":

            if PRE_PROC_TYPE in ("gdbsp", "bsp"):

                if PRE_PROC_TYPE == "bsp":
                    adjusted_blue, adjusted_green, adjusted_softmax, _, _, _, _ = amiu.adjust_section_bsp(
                        section_blue,
                        section_green,
                        BSP_BLUE_TARGET["mean"],
                        BSP_GREEN_TARGET["mean"],
                        unet.UnetSegmenter(
                            num_input_channels=2,
                            num_output_channels=3,
                            snapshot_file=f"{SNAPSHOTS_DIR}/snapshot_epoch_{epoch_selection[-1]}.pt"
                        ).eval().to(DEVICE)
                    )
                elif PRE_PROC_TYPE == "gdbsp":
                    adjusted_blue, adjusted_green, adjusted_softmax, _, _, _, _ = amiu.adjust_section_gdbsp(
                        section_blue,
                        section_green,
                        BSP_BLUE_TARGET,
                        BSP_GREEN_TARGET,
                        unet.UnetSegmenter(
                            num_input_channels=2,
                            num_output_channels=3,
                            snapshot_file=f"{SNAPSHOTS_DIR}/snapshot_epoch_{epoch_selection[-1]}.pt"
                        ).eval().to(DEVICE),
                        adaptive_histeq=True
                    )

                ensemble_softmax += adjusted_softmax

                del adjusted_softmax
                gc.collect()
                torch.cuda.empty_cache()

            elif PRE_PROC_TYPE == "histmatch":
                adjusted_blue, adjusted_green = amiu.adjust_section_histmatch(
                    section_blue,
                    section_green,
                    HISTMATCH_BLUE_TARGET,
                    HISTMATCH_GREEN_TARGET
                )

            elif PRE_PROC_TYPE == "histeq":
                adjusted_blue, adjusted_green = amiu.adjust_section_histeq(
                    section_blue,
                    section_green
                )

            elif PRE_PROC_TYPE == "dog":
                adjusted_blue, adjusted_green = amiu.adjust_section_dog(
                    section_blue,
                    section_green
                )

            elif PRE_PROC_TYPE == "adaptive_histeq":
                adjusted_blue, adjusted_green = amiu.adjust_section_adaptive_histeq(
                    section_blue,
                    section_green
                )

            io.imsave(f"{ADJUSTED_DIR}/{section_name}_1_{PRE_PROC_TYPE}.png", adjusted_blue)
            io.imsave(f"{ADJUSTED_DIR}/{section_name}_2_{PRE_PROC_TYPE}.png", adjusted_green)

            del section_blue, section_green, adjusted_blue, adjusted_green
            gc.collect()


        # Run predictions for the entire ensemble
        for epoch in epoch_selection:

            if PRE_PROC_TYPE in ("gdbsp", "bsp") and epoch == epoch_selection[-1]:
                print(f"Skipping final epoch for {PRE_PROC_TYPE.upper()}, already did that beforehand...")
                continue

            snapshot = unet.UnetSegmenter(
                num_input_channels=2,
                num_output_channels=3,
                snapshot_file=f"{SNAPSHOTS_DIR}/snapshot_epoch_{epoch}.pt"
            ).eval().to(DEVICE)

            if PRE_PROC_TYPE == "raw":
                section_blue = io.imread(f"{DATA_DIR}/{section_name}_1_raw.png")
                section_green = io.imread(f"{DATA_DIR}/{section_name}_2_raw.png")
            else:
                section_blue = io.imread(f"{ADJUSTED_DIR}/{section_name}_1_{PRE_PROC_TYPE}.png")
                section_green = io.imread(f"{ADJUSTED_DIR}/{section_name}_2_{PRE_PROC_TYPE}.png")

            prediction_softmax, _, _ = snapshot.predict_section(np.array([
                section_blue, section_green
            ]))

            ensemble_softmax += prediction_softmax

            del section_blue, section_green, prediction_softmax, snapshot
            gc.collect()
            torch.cuda.empty_cache()


        # CALCULATE CONFIDENCE

        # Average all epoch softmaxes for ensemble softmax
        ensemble_softmax /= len(epoch_selection)

        # Max of softmax at each pixel is pixel confidence
        # Mean of pixel confidences is average section confidence
        avg_confidence = float(np.mean(np.max(ensemble_softmax, axis=0)))
        assert 0 <= avg_confidence <= 1

        # Pick the highest confidence class, make class map image, save it
        print(f"Saving predictions for active section {section_name}...")

        ensemble_image = amiu.ground_truth_map_to_image(np.argmax(ensemble_softmax, axis=0))

        # Store the section's confidence, save with each loop
        if EXPERIMENT == "c1":
            gt = io.imread(f"{DATA_DIR}/{section_name}_ground_truth_2ch.png")
            dice_score = unet.dice_coefficient(ensemble_image, gt)

            confidences[section_name] = {
                "dice": dice_score,
                "conf": avg_confidence
            }
        else:
            confidences[section_name] = avg_confidence

        qj.save_file(confidences_file, confidences)

        io.imsave(
            f"{PREDICTIONS_DIR}/pred_{section_name}.png",
            ensemble_image
        )

        del ensemble_softmax, ensemble_image
        gc.collect()


print(f"Done computing confidences for {EXPERIMENT} experiment run {RUN_NAME}.")



#
#
# WRAP UP
#
#

print(f"Wrapping up {EXPERIMENT} experiment run {RUN_NAME}...")

for version_name, _ in VERSIONS:

    VERSION_DIR = f"{RUN_NAME}/{version_name}"
    PREDICTIONS_DIR = f"{VERSION_DIR}/predictions"
    MAPS_DIR = f"{PREDICTIONS_DIR}/maps"
    ADJUSTED_DIR = f"{PREDICTIONS_DIR}/adjusted"

    with open(f"{VERSION_DIR}/worst_fold.txt", "r") as fp:
        worst_fold = fp.read().strip()

    test_dice = [
        np.mean(pd.read_csv(
            f"{RUN_NAME}/fold_{fold}/{version_name}/test/metrics_testing.csv"
        )["dice_score"].tolist())
        for fold in FOLDS
    ]
    # test_conf = [
    #     np.mean(pd.read_csv(
    #         f"{RUN_NAME}/fold_{fold}/{version_name}/test/metrics_testing.csv"
    #     )["avg_confidence"].tolist())
    #     for fold in ['a', 'b', 'c', 'd']
    # ]

    # Read the confidences json
    confidences = qj.load_file(f"{PREDICTIONS_DIR}/confidences_{RUN_NAME}_{version_name}.json")

    if EXPERIMENT == "c1":
        pred_dice = [confidences[s]["dice"] for s in confidences]
        confidences = {
            s: confidences[s]["conf"]
            for s in confidences
        }
        
    conf_average = np.mean(list(confidences.values()))

    # Print out run results
    print(f"Printing {EXPERIMENT} experiment {RUN_NAME} results to file...")

    with open(f"{RUN_NAME}/results.txt", "a") as results_file:
        print(
            f"\nResults for active learning {EXPERIMENT} experiment run {RUN_NAME},\n",
            f"worst fold {worst_fold}, preprocessing type {PRE_PROC_TYPE},\n",
            f"version {version_name}:\n",
            file=results_file
        )
        for fold, dice in zip(FOLDS, test_dice):
            print(f"Fold {('(worst) ' if fold == worst_fold else '') + fold} test Dice: {dice:0.4f}", file=results_file)
        print(f"Average test Dice: {np.mean(test_dice):0.4f}", file=results_file)
        if EXPERIMENT == "c1":
            print(f"Average prediction Dice: {np.mean(pred_dice):0.4f}", file=results_file)
        # for fold, conf in zip(FOLDS, test_conf):
        #     print(f"Fold {('(worst) ' if fold == worst_fold else '') + fold} test confidence: {conf:0.4f}")
        # print(f"Average test confidence: {np.mean(test_conf):0.4f}")
        for threshold in range(90, 100):
            given_conf_sections = [s for s in confidences if confidences[s] >= (threshold / 100)]
            print(f"# sections over {threshold}% confidence threshold: {len(given_conf_sections)}/{len(confidences)}", file=results_file)
        print(f"Average active set confidence: {conf_average:0.4f}\n", file=results_file)

    shutil.copy(
        f"{RUN_NAME}/results.txt",
        f"/data/bsp_experiment/{EXPERIMENT}/{RUN_NAME}_results.txt"
    )


print(f"Done wrapping up {EXPERIMENT} experiment run {RUN_NAME}.")
