# Generate training data for BSP and GDBSP

time python3 generate_training_gdbsp_bsp.py

# Cohort 1 Experiments

time python3 active_learning.py c1/config_raw_cohort1_iter_1.json
time python3 active_learning.py c1/config_dog_cohort1_iter_1.json
time python3 active_learning.py c1/config_histeq_cohort1_iter_1.json
time python3 active_learning.py c1/config_adaptive_histeq_cohort1_iter_1.json
time python3 active_learning.py c1/config_histmatch_cohort1_iter_1.json
time python3 active_learning.py c1/config_bsp_cohort1_iter_1.json
time python3 active_learning.py c1/config_gdbsp_cohort1_iter_1.json

# Cohorts 2 & 3 Experiments

time python3 active_learning.py c23/config_raw_adadelta_iter_1.json
time python3 active_learning.py c23/config_dog_adadelta_iter_1.json
time python3 active_learning.py c23/config_histeq_adadelta_iter_1.json
time python3 active_learning.py c23/config_adaptive_histeq_adadelta_iter_1.json
time python3 active_learning.py c23/config_histmatch_adadelta_iter_1.json
time python3 active_learning.py c23/config_bsp_adadelta_iter_1.json
time python3 active_learning.py c23/config_gdbsp_adadelta_iter_1.json

echo "Sleeping in 60 seconds."
sleep 60
systemctl suspend
