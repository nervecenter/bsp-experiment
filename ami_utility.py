"""
ami_utility.py by Chris Collazo
Helper functions for handling fluorescence images of sections for
acute myocardial infarction deep learning and active learning.
"""

import gc
from math import isclose

import numpy as np
import imageio as io
import matplotlib.pyplot as plt

from numba import jit, njit

from skimage.exposure import adjust_gamma, equalize_hist, equalize_adapthist, match_histograms
from skimage.filters import threshold_otsu, difference_of_gaussians

# Disable decompression bomb warnings for large images
from warnings import simplefilter
from PIL.Image import DecompressionBombWarning
simplefilter('ignore', DecompressionBombWarning)

import unet


# original_sections = [
#     "p177", "p179", "p270", "p271", "p272", "p273",
#     "p274", "p277", "p300", "p301", "p302"
# ]

def ground_truth_map_to_image(pred_map):

    assert pred_map.ndim == 2, f"Expected 2-dimensional prediction map, got {pred_map.ndim} dimensions."

    height, width = pred_map.shape[:2]

    prediction_image = np.full((height, width, 4), (255, 0, 255, 255), dtype=np.uint8)
    prediction_image[pred_map == 0] = (0, 0, 0, 0)
    prediction_image[pred_map == 1] = (255, 255, 0, 255)
    prediction_image[pred_map == 2] = (0, 255, 0, 255)

    none = np.all(prediction_image == (0, 0, 0, 0), axis=-1)
    normal = np.all(prediction_image == (0, 255, 0, 255), axis=-1)
    risk = np.all(prediction_image == (255, 255, 0, 255), axis=-1)
    assert np.all(none | normal | risk), "Prediction map has an invalid class color, can't turn to image."

    return prediction_image


def ground_truth_image_to_1hot(gt_image):

    border_map = np.all(gt_image == (255,255,0,255), axis=-1)
    normal_map = np.all(gt_image == (0,255,0,255), axis=-1)
    none_map = np.all(gt_image == (0,0,0,0), axis=-1)
    assert np.all(none_map == ~(border_map | normal_map))

    gt_1hot = (np.array((none_map, border_map, normal_map)) * 1).astype(np.uint8)
    assert gt_1hot.dtype == np.uint8
    assert np.max(gt_1hot) == 1

    return gt_1hot


def ground_truth_image_to_map(gt_image):

    border_map = np.all(gt_image == (255,255,0,255), axis=-1)
    normal_map = np.all(gt_image == (0,255,0,255), axis=-1)
    none_map = np.all(gt_image == (0,0,0,0), axis=-1)
    assert np.all(none_map == ~(border_map | normal_map))

    gt_map = np.zeros(gt_image.shape[:2], dtype=np.uint8)
    gt_map[none_map] = 0
    gt_map[border_map] = 1
    gt_map[normal_map] = 2
    assert gt_map.dtype == np.uint8
    assert np.all((gt_map == 0) | (gt_map == 1) | (gt_map == 2))

    return gt_map


def tile_locations(height, width):
    """
    Return a list of tuples of the (j, i) coordinates of every tile location
    for the given image size.
    """
    tile_height = height // 512
    tile_width = width // 512

    y_coords = [j * 512 for j in range(tile_height)]
    x_coords = [i * 512 for i in range(tile_width)]
    tile_locs = [(j, i) for j in y_coords for i in x_coords]

    return tile_locs


def tile_locations_evaluation(height, width):
    """
    Return a list of tuples of the (j, i) coordinates of every tile location
    for the given image size, with smaller shifts for evaluation
    of a whole-slide image.
    """
    tile_width = (width - 256) // 256
    tile_height = (height - 256) // 256

    y_coords = [j * 256 for j in range(tile_height)]
    x_coords = [i * 256 for i in range(tile_width)]
    tile_locs = [(j, i) for j in y_coords for i in x_coords]

    # Add on the right and bottom sides for processing
    tile_locs += [(j, width - 512) for j in y_coords]
    tile_locs += [(height - 512, i) for i in x_coords]

    return tile_locs


def load_samples(data_dir, section, channels_list, preproc_list):

    print(f"Loading samples for section {section}...")

    num_channels = len(channels_list)
    assert len(preproc_list) == num_channels

    section_raw = np.array([
        io.imread(f"{data_dir}/{section}_{channel}_{preproc}.png")
        for channel, preproc in zip(channels_list, preproc_list)
    ]).astype(np.float32) / 255.0
    assert section_raw.shape[0] == num_channels

    height, width = section_raw.shape[1:3]
    
    tile_locs = tile_locations(height, width)
    num_tiles = len(tile_locs)

    tiles = np.array([
        section_raw[:, j : j + 512, i : i + 512]
        for (j, i) in tile_locs
    ])
    assert tiles.shape == (num_tiles, num_channels, 512, 512), f"tiles shape: {tiles.shape}"

    return tiles


def load_ground_truth_samples_2ch(data_dir, section):

    print(f"Loading ground truth samples for section {section}...")

    section_gt = io.imread(f"{data_dir}/{section}_ground_truth_2ch.png")
    assert section_gt.shape[2] == 4

    height, width = section_gt.shape[:2]
    
    tile_locs = tile_locations(height, width)
    num_tiles = len(tile_locs)

    segmentation = ground_truth_image_to_map(section_gt)

    tiles_maps = np.array([
        segmentation[j : j + 512, i : i + 512]
        for (j, i) in tile_locs
    ])
    assert tiles_maps.shape == (num_tiles, 512, 512), f"tiles_maps shape: {tiles_maps.shape}"

    return tiles_maps


def evaluation_samples_from_raw(section_raw):

    # print("Loading overlapped evaluation samples for raw input...")

    section_raw = section_raw.astype(np.float32) / 255.0
    num_channels = section_raw.shape[0]

    height, width = section_raw.shape[1:3]
    
    tile_locs = tile_locations_evaluation(height, width)
    num_tiles = len(tile_locs)

    tiles = np.array([
        section_raw[:, j : j + 512, i : i + 512]
        for (j, i) in tile_locs
    ])
    assert tiles.shape == (num_tiles, num_channels, 512, 512), f"tiles shape: {tiles.shape}"

    return tiles, tile_locs, height, width


def load_evaluation_samples(data_dir, section, channels_list, preproc_list):
    
    print(f"Loading overlapped evaluation samples for section {section}...")

    num_channels = len(channels_list)
    assert len(preproc_list) == num_channels

    section_raw = np.array([
        io.imread(f"{data_dir}/{section}_{channel}_{preproc}.png")
        for channel, preproc in zip(channels_list, preproc_list)
    ]).astype(np.float32) / 255.0
    assert section_raw.shape[0] == num_channels

    height, width = section_raw.shape[1:3]
    
    tile_locs = tile_locations_evaluation(height, width)
    num_tiles = len(tile_locs)

    tiles = np.array([
        section_raw[:, j : j + 512, i : i + 512]
        for (j, i) in tile_locs
    ])
    assert tiles.shape == (num_tiles, num_channels, 512, 512), f"tiles shape: {tiles.shape}"

    return tiles, tile_locs, height, width


def load_evaluation_ground_truth_samples_2ch(data_dir, section):
    
    print(f"Loading overlapped evaluation ground truth samples for section {section}...")

    section_gt = io.imread(f"{data_dir}/{section}_ground_truth_2ch.png")
    assert section_gt.shape[2] == 4

    height, width = section_gt.shape[:2]
    
    tile_locs = tile_locations_evaluation(height, width)
    num_tiles = len(tile_locs)

    segmentation = ground_truth_image_to_map(section_gt)

    tiles_maps = np.array([
        segmentation[j : j + 512, i : i + 512]
        for (j, i) in tile_locs
    ])
    assert tiles_maps.shape == (num_tiles, 512, 512), f"tiles_maps shape: {tiles_maps.shape}"

    return tiles_maps


def match_intensity_euclidean(image,
                              class_map,
                              target_intensities,
                              num_classes):

    def class_mean(image, class_map, class_num, baseline):
        vals = image[class_map == class_num]
        return np.mean(vals) if np.any(vals) else baseline

    assert num_classes >= np.max(class_map) + 1
    assert num_classes == len(target_intensities)

    target_intensities = np.array(target_intensities)
    intensity_log = []

    # Scale of gamma adjustment, grows until we hit 1.0
    delta_gamma = 0.05

    # Compute metrics on the initial input image
    # The average class intensities
    # The per-class distance
    # The relative Euclidean distance, positive or negative (brighter or darker)
    # And the distance magnitude
    prev_image = image
    # print(f"Initial image for Euclidean adjustment: {prev_image}")
    prev_image_avg_intensities = np.array([
        class_mean(prev_image, class_map, c, target_intensities[c])
        for c in range(num_classes)
    ])
    prev_intensity_dists = target_intensities - prev_image_avg_intensities
    prev_dist_relative = np.sum(prev_intensity_dists)
    prev_dist_magnitude = np.linalg.norm(prev_intensity_dists)

    while True:

        # print(f"Distance magnitude: {prev_dist_magnitude}")

        # Depending on relative distance, lighten or darken
        # the PREVIOUS IMAGE to get closer
        if prev_dist_relative > 0:
            adjusted_image = adjust_gamma(prev_image, 1.0 - delta_gamma)
        else:
            adjusted_image = adjust_gamma(prev_image, 1.0 + delta_gamma)

        # Get adjusted image intensities, relative distance,
        # distance magnitude to target
        adjusted_image_avg_intensities = np.array([
            class_mean(adjusted_image, class_map, c, target_intensities[c])
            for c in range(num_classes)
        ])
        adjusted_intensity_dists = target_intensities - adjusted_image_avg_intensities
        adjusted_dist_relative = np.sum(adjusted_intensity_dists)
        adjusted_dist_magnitude = np.linalg.norm(adjusted_intensity_dists)

        # If the adjusted image is FURTHER than the previous or stops,
        # the prev image is the one we want, break to return it
        if adjusted_dist_magnitude > prev_dist_magnitude \
        or isclose(adjusted_dist_magnitude, prev_dist_magnitude):
            adjusted_image = prev_image
            break
        # Otherwise, the last adjustment is fine, keep going
        prev_image = adjusted_image
        prev_image_avg_intensities = adjusted_image_avg_intensities
        prev_intensity_dists = adjusted_intensity_dists
        prev_dist_relative = adjusted_dist_relative
        prev_dist_magnitude = adjusted_dist_magnitude

        intensity_log.append(prev_image_avg_intensities.tolist())

    return adjusted_image, intensity_log


def reconstruct_evaluation(predicted_tiles_softmax,
                           tile_locs,
                           height,
                           width):

    num_classes = predicted_tiles_softmax.shape[1]
    prediction_softmax = np.zeros((num_classes, height, width), dtype=np.float32)

    for n, (j, i) in enumerate(tile_locs):

        # Slice out the center 256x256 of the tile for the final confidence map
        prediction_softmax[
            :,
            j + 128 : j + 384,
            i + 128 : i + 384
        ] = predicted_tiles_softmax[n, :, 128:384, 128:384]

        # Keep the left if this is the first column
        if i == 0:
            prediction_softmax[
                :,
                j : j + 512,
                i : i + 128
            ] = predicted_tiles_softmax[n, :, :, 0:128]

        # Keep the top if this is the first row
        if j == 0:
            prediction_softmax[
                :,
                j : j + 128,
                i : i + 512
            ] = predicted_tiles_softmax[n, :, 0:128, :]

        # Keep the right if this is the right side
        if i == width - 512:
            prediction_softmax[
                :,
                j : j + 512,
                i + 384 : i + 512
            ] = predicted_tiles_softmax[n, :, :, 384:512]

        # Keep the bottom if this is the bottom side
        if j == height - 512:
            prediction_softmax[
                :,
                j + 384 : j + 512,
                i : i + 512
            ] = predicted_tiles_softmax[n, :, 384:512, :]

    prediction_map = np.argmax(prediction_softmax, axis=0)
    prediction_image = ground_truth_map_to_image(prediction_map)

    return prediction_softmax, prediction_map, prediction_image


def iterative_adjust_section(data_dir,
                             section_name,
                             blue_target,
                             green_target,
                             model,
                             num_iterations=5):

    print(f"Iterative adjustment for section {section_name}...")

    blue_raw = io.imread(f"{data_dir}/{section_name}_1_raw.png")
    green_raw = io.imread(f"{data_dir}/{section_name}_2_raw.png")

    height, width = blue_raw.shape[:2]
    assert (height, width) == green_raw.shape[:2]

    # Otsu threshold, save as class map, then use to adjust for iter 1
    blue_otsu = blue_raw >= threshold_otsu(blue_raw)
    green_otsu = green_raw >= threshold_otsu(green_raw)

    prev_blue_map = blue_otsu
    prev_green_map = green_otsu
    

    #
    # ITERATE, adjusting brightness and generating new maps each time
    #

    for itr in range(1, num_iterations + 1):

        print(f"Iterative adjustment step {itr}")

        # print("Matching intensities...")
        blue_distmin, _ = match_intensity_euclidean(
            blue_raw, prev_blue_map * 1, blue_target, 2
        )
        green_distmin, _ = match_intensity_euclidean(
            green_raw, prev_green_map * 1, green_target, 2
        )

        # Evaluate first class map
        # print("Evaluating...")
        prediction_softmax, prediction_map, prediction_image = model.predict_section(np.array([
            blue_distmin, green_distmin
        ]))
        assert prediction_map.shape == (height, width)

        # print("Isolating classes...")
        prev_blue_map = ((prediction_map == 1) | (prediction_map == 2))
        prev_green_map = (prediction_map == 2)

    return blue_distmin, green_distmin, prediction_softmax, prediction_map, prediction_image


def adjust_section_bsp(blue_raw,
                       green_raw,
                       blue_target,
                       green_target,
                       model,
                       num_iterations=5):

    # print(f"Model is on cuda: {model.is_on_cuda()}")

    height, width = blue_raw.shape[:2]
    assert (height, width) == green_raw.shape[:2]

    # Otsu threshold, save as class map, then use to adjust for iter 1
    blue_otsu = blue_raw >= threshold_otsu(blue_raw)
    green_otsu = green_raw >= threshold_otsu(green_raw)

    prev_blue_map = blue_otsu
    prev_green_map = green_otsu
    

    #
    # ITERATE, adjusting brightness and generating new maps each time
    #

    for itr in range(1, num_iterations + 1):

        print(f"Iterative adjustment step {itr}")

        # print("Matching intensities...")
        adjusted_blue, blue_log = match_intensity_euclidean(
            blue_raw, prev_blue_map * 1, blue_target, 2
        )
        adjusted_green, green_log = match_intensity_euclidean(
            green_raw, prev_green_map * 1, green_target, 2
        )

        # Evaluate first class map
        # print("Evaluating...")
        prediction_softmax, prediction_map, prediction_image = model.predict_section(np.array([
            adjusted_blue, adjusted_green
        ]))
        assert prediction_map.shape == (height, width)

        # print("Isolating classes...")
        prev_blue_map = ((prediction_map == 1) | (prediction_map == 2))
        prev_green_map = (prediction_map == 2)

    return adjusted_blue, adjusted_green, prediction_softmax, prediction_map, prediction_image, blue_log, green_log


def adjust_section_histeq(blue_raw, green_raw):

    adjusted_blue = (equalize_hist(blue_raw) * 255).astype(np.uint8)
    adjusted_green = (equalize_hist(green_raw) * 255).astype(np.uint8)

    return adjusted_blue, adjusted_green


def adjust_section_dog(blue_raw, green_raw, low_sigma=15, high_sigma=75):

    adjusted_blue = difference_of_gaussians(blue_raw, low_sigma, high_sigma=high_sigma)
    adjusted_green = difference_of_gaussians(green_raw, low_sigma, high_sigma=high_sigma)

    return adjusted_blue, adjusted_green


def adjust_section_adaptive_histeq(blue_raw, green_raw):

    adjusted_blue = equalize_adapthist(blue_raw)
    adjusted_green = equalize_adapthist(green_raw)

    return adjusted_blue, adjusted_green


def adjust_section_histmatch(blue_raw,
                             green_raw,
                             histmatch_blue_target_image,
                             histmatch_green_target_image):

    adjusted_blue = match_histograms(
        blue_raw,
        histmatch_blue_target_image
    ).astype(np.uint8)
    adjusted_green = match_histograms(
        green_raw,
        histmatch_green_target_image
    ).astype(np.uint8)

    return adjusted_blue, adjusted_green




# TODO: Handle the edge case of missing class areas
@njit
def class_mean_vector(im, gt, num_classes=2):
    # num_classes = np.max(gt) + 1
    im = im.flatten()
    gt = gt.flatten()
    return np.array([np.mean(im[gt == c]) for c in range(num_classes)])
# @njit
# def class_mean_vector(im, gt, num_classes=2):
#     # num_classes = np.max(gt) + 1
#     class_means = np.zeros((num_classes))
#     class_counts = np.zeros((num_classes))
#     height, width = gt.shape
#     for j in range(height):
#         for i in range(width):
#             class_counts[gt[j, i]] += 1
#             class_means[gt[j, i]] += im[j, i]
#     for c in range(num_classes):
#         class_means[c] /= class_counts[c]
#     return class_means

@njit
def class_stdev_vector(im, gt, num_classes=2):
    # num_classes = np.max(gt) + 1
    im = im.flatten()
    gt = gt.flatten()
    return np.array([np.std(im[gt == c]) for c in range(num_classes)])

@njit
def all_metrics_vector(im, gt, num_classes=2):
    return np.concatenate((class_mean_vector(im, gt), class_stdev_vector(im, gt)))

@njit
def contrast(im, fac):
    return np.clip(((im.astype(np.int16) - 128) * fac) + 128, 0, 255).astype(np.uint8)

@njit
def gamma(im, g):
    return np.clip(im**g, 0, 255).astype(np.uint8)

@njit
def brightness(im, fac):
    return np.clip(im * fac, 0, 255).astype(np.uint8)

@njit
def semantic_distance(sv, tv):
    return np.linalg.norm(tv - sv)

# def adaptive_equalization(im):
#     return (equalize_adapthist(im) * 255).astype(np.uint8)

@jit
def gdsp(source, gt, target_metrics, num_classes=2, adaptive_histeq=False):

    methods = [contrast, brightness, gamma]
    target_vector = np.concatenate((target_metrics["mean"], target_metrics["stdev"]))

    semdist_log = [semantic_distance(all_metrics_vector(source, gt), target_vector)]

    # Start with adaptive histogram equalization
    # prev_adjusted = adaptive_equalization(source) if adaptive_histeq else source
    prev_adjusted = source

    metric_vector = np.array(all_metrics_vector(prev_adjusted, gt))
    nan_elements = np.isnan(metric_vector)
    metric_vector[nan_elements] = target_vector[nan_elements]
    prev_dist = semantic_distance(metric_vector, target_vector)

    # mod_value = 0.1
    # mod_diff = 0.01
    mod_values = [0.2, 0.1, 0.5, 0.01]
    m = 0

    if adaptive_histeq:
        semdist_log.append(semantic_distance(metric_vector, target_vector))

    while m < len(mod_values):

        while True:

            method_results = []

            for method in methods:

                for fac in [1.0 - mod_values[m], 1.0 + mod_values[m]]:

                    adjusted = method(prev_adjusted, fac)
                    metric_vector = all_metrics_vector(adjusted, gt)
                    metric_vector[np.isnan(metric_vector)] = target_vector[np.isnan(metric_vector)]
                    semdist = semantic_distance(metric_vector, target_vector)

                    if semdist < prev_dist:
                        method_results.append((method.__name__, semdist, method, fac, adjusted, metric_vector))

            # del adjusted, metric_vector, semdist
            # gc.collect()

            if len(method_results) == 0:
                # print(f"No good method, stopping.")
                break

            best_method = np.argmin([mr[1] for mr in method_results])
            # method_name, semdist, method, fac, adjusted, metvec = method_results[best_method]
            _, semdist, _, _, adjusted, _ = method_results[best_method]

            semdist_log.append(semdist)

            # print(f"Best is {method_name} at {fac}, dist {semdist}")

            prev_adjusted = adjusted
            prev_dist = semdist

            # del method_results
            # gc.collect()

        # mod_value -= mod_diff
        m += 1

    semdist_log = np.array(semdist_log)

    return prev_adjusted, semdist_log


def adjust_section_gdbsp(blue_raw,
                         green_raw,
                         blue_target,
                         green_target,
                         model,
                         num_iterations=5,
                         adaptive_histeq=False):

    # print(f"Model is on cuda: {model.is_on_cuda()}")

    height, width = blue_raw.shape[:2]
    assert (height, width) == green_raw.shape[:2]

    # Otsu threshold, save as class map, then use to adjust for iter 1
    blue_otsu = blue_raw >= threshold_otsu(blue_raw)
    green_otsu = green_raw >= threshold_otsu(green_raw)

    prev_blue_map = blue_otsu
    prev_green_map = green_otsu
    

    #
    # ITERATE, adjusting brightness and generating new maps each time
    #

    for itr in range(1, num_iterations + 1):

        # print(f"Iterative adjustment step {itr}")

        # print("Matching intensities...")
        adjusted_blue, blue_log = gdsp(
            blue_raw, prev_blue_map * 1, blue_target, 2, adaptive_histeq=adaptive_histeq
        )
        adjusted_green, green_log = gdsp(
            green_raw, prev_green_map * 1, green_target, 2, adaptive_histeq=adaptive_histeq
        )

        # Evaluate first class map
        # print("Evaluating...")
        prediction_softmax, prediction_map, prediction_image = model.predict_section(np.array([
            adjusted_blue, adjusted_green
        ]))
        assert prediction_map.shape == (height, width)

        # print("Isolating classes...")
        prev_blue_map = ((prediction_map == 1) | (prediction_map == 2))
        prev_green_map = (prediction_map == 2)

    return adjusted_blue, adjusted_green, prediction_softmax, prediction_map, prediction_image, blue_log, green_log


def save_gdbsp_semdist_log_plots(directory, title, semdist_log):

    num_iters = semdist_log.shape[0]

    plt.title(f"{title} - Semantic Distance")
    plt.plot(range(0, len(semdist_log)), semdist_log, "o", linestyle="--")
    plt.xlabel("Iteration")
    plt.ylabel("Semantic Distance")
    plt.savefig(f"{directory}/{title}_semdist.png", bbox_inches="tight", dpi=600)
    plt.close()


def main():
    from unet import UnetSegmenter

    data_dir = "/data/user/ami"
    section = "p274"
    blue_target = [6.96745360309041, 20.28935167969]
    green_target = [0.968171754124953, 14.5505361922275]

    from unet import UnetSegmenter
    model = UnetSegmenter(
        num_input_channels=2,
        num_output_channels=3,
        snapshot_file="bsp_cyclic_lowlr_iter_1/fold_a/snapshots/snapshot_epoch_110.pt"
    ).eval().cuda()

    # Test iterative adjustment
    blue, green, _, _, gt = iterative_adjust_section(
        data_dir,
        section,
        blue_target,
        green_target,
        model
    )

    io.imsave("blue_image.png", blue)
    io.imsave("green_image.png", green)
    io.imsave("ground_truth_2ch.png", gt)

if __name__ == "__main__":
    main()
