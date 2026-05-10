# Data

This repository does not include the full raw or processed ICU datasets.

The project uses:

- PhysioNet/Computing in Cardiology Challenge 2019 sepsis dataset
- PHEMS ICU dataset

The full datasets are excluded from GitHub due to file size, licensing, and privacy considerations.

## Expected Local Folder Structure

To run the scripts locally, place the datasets in the following locations:

```text
data/
├── raw/
│   ├── physionet_2019.csv
│   └── phems_data/
│       └── training_data/
│           ├── SepsisLabel_train.csv
│           ├── measurement_meds_train.csv
│           ├── measurement_lab_train.csv
│           ├── measurement_observation_train.csv
│           └── person_demographics_episode_train.csv
│
└── processed/
    ├── physionet_pruned.csv
    ├── physionet_balanced.csv
    ├── phems_flat.csv
    ├── phems_clean.csv
    └── phems_balanced.csv