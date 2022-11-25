# bsp-experiment

Source code for my research concerning Bootstrapped Semantic Preprocessing, submitted to IEEE Transactions on Image Processing.

## Deep Learning-Based Preprocessing Enables Automatic Labeling of Very Large Whole-Slide Images

> Since the FDA approved the first whole slide image system for diagnosis in 2017, whole slide images have provided enriched critical information to advance the field of medical histopathology. However, progress in this field has been greatly hindered due to the tremendous cost and time associated with generating region of interest (ROI) ground truth for super- vised machine learning, alongside concerns with inconsistent microscopy imaging acquisition. Active learning has presented a potential solution to these problems, expanding dataset ground truth by algorithmically choosing the most informative samples for ground truth labeling; still, this incurs the costs of human labeling efforts, which need minimization. Alternatively, auto- matic labeling approaches using active learning tend to overfit and select samples most similar to the training set distribu- tion, while excluding out-of-distribution samples which might be informative and improve model effectiveness. We propose that inconsistent cross microscopic images induce the bulk of this disparity. We quantify and demonstrate the inconsistencies present in our own dataset. A deep learning-based preprocessing algorithm which aims to normalize unknown samples to the training set distribution and short-circuit the overfitting problem is presented. We demonstrate that our approach greatly increases the amount of automatic ROI ground truth labeling possible on very large whole-slide images with active deep learning. We accept 92% of the automatic labels generated for our unlabeled data cohort, thereby expanding the labeled dataset by 845%. We also demonstrate a 96% expert time savings relative to manual expert ground-truth.

- Experiment is run using `sh run_experiment.sh`.
- `active_learning.py` contains the primary experimental code.
- `ami_utility.py` contains the novel image processing algorithms (BSP, GDBSP) invented for this experiment.
- `unet.py` contains the Unet deep learning architecture used.
- `choose_threshold_samples.py` and `integrate_accepted_sections.py` contain post-iteration processing for thresholding and integrating accepted samples, to be run after each iteration for each preprocessing type.
- `generate_training_gdbsp_bsp.py` helps to produce ground truth versions of the training set for the novel image processing algorithms (BSP, GDBSP).
- `c1` and `c23` contain configurations for the initial iterations of the Cohort 1 experiments and Cohorts 2 & 3 experiments respectively.
