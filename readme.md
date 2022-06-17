# Hippocampus Segmentation

## Setup

It is recommended to create a new virtual environment with python 3.8.

To install the required packages run 
```
pip install -r requirements.txt
```

## Run

To generate predictions run the command

```
python predict.py DATASET_PATH OUT_FOLDER OUTPUT_FILENAME
```

For more options and information run `python predict.py --help`

Input data within `DATASET_PATH` is required to be in a specific format.

```
DATASET_PATH
└── subjects
    ├── sub001
    │   ├── mean_dwi.nii.gz
    │   ├── fa.mif.nii
    │   └── md.mif.nii
    ├── sub002
    │   ├── mean_dwi.nii.gz
    │   ├── fa.mif.nii
    │   └── md.mif.nii
    .
    .
    .

```

Within the `DATASET_PATH` directory, there must be a directory named `subjects`. Within the subject directory, each subject will have their own directory containing the 3 input files with file stem names matching exactly (mean_dwi, fa, md). gz compression is optional. The name of each subject folder does not need to match the above example.

The output segmentations will be saved to `OUT_FOLDER` with the same per subject structure. Each segmentation file will be saved using `OUTPUT_FILENAME`.  Provide the `OUTPUT_FILENAME` without any extensions.

Example command 
```
python predict.py input-data/ output/ whole_roi
```

If you get an out of memory error, reduce the batch size with the argument `--batch_size`. Larger batch_size produces faster execution but has larger memory requirements. By default, the batch_size is set to 4.

Use `--cpu=True` if errors occur with gpu usage. Be aware the setting this flag will take significantly longer to generate predictions.

## Additional Information
This automatic method was trained on healthy subjects with an age between 5 - 74 years using the cbbrain/ab300 protocol. Segmentations should be manually verified, especially if they do not match the above training data characteristics.


Segmentation encoding is:   
0 - Background   
1 - Left Hippocampus   
2 - Right Hippocampus   


## Authors

Dylan Miller (email: millerd238@mymacewan.ca)   
Cory Efird 

Contact Dylan if you have any questions/problems.