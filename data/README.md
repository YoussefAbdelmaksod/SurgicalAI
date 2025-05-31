# SurgicalAI Datasets

This directory contains the datasets used for training and evaluating the SurgicalAI models.

## Required Datasets

The SurgicalAI system uses three main datasets:

1. **Cholec80** - For phase recognition
2. **m2cai16-tool-locations** - For tool detection
3. **EndoScapes** - For mistake detection

## Dataset Setup Instructions

### 1. Cholec80 Dataset

The Cholec80 dataset consists of 80 videos of cholecystectomy procedures with phase annotations.

1. Download the dataset from the [Cholec80 website](http://camma.u-strasbg.fr/datasets)
2. Extract the dataset into the `data/Cholec80.v5-cholec80-10-2.coco` directory
3. The directory structure should look like:
   ```
   data/Cholec80.v5-cholec80-10-2.coco/
   ├── train/
   │   ├── _annotations.coco.json
   │   ├── video01_frame000.jpg
   │   ├── video01_frame001.jpg
   │   └── ...
   ├── val/
   │   ├── _annotations.coco.json
   │   └── ...
   └── test/
       ├── _annotations.coco.json
       └── ...
   ```

### 2. m2cai16-tool-locations Dataset

The m2cai16-tool-locations dataset contains tool detection annotations for laparoscopic procedures.

1. Download the dataset from the [m2cai16 website](http://camma.u-strasbg.fr/m2cai2016)
2. Extract the dataset into the `data/m2cai16-tool-locations` directory
3. The directory structure should look like:
   ```
   data/m2cai16-tool-locations/
   ├── train/
   │   ├── _annotations.coco.json
   │   ├── image001.jpg
   │   ├── image002.jpg
   │   └── ...
   ├── val/
   │   ├── _annotations.coco.json
   │   └── ...
   └── test/
       ├── _annotations.coco.json
       └── ...
   ```

### 3. EndoScapes Dataset

The EndoScapes dataset contains annotated endoscopic images with segmentation masks and mistake annotations.

1. Register and download the dataset from the [EndoScapes website](https://endoscapes.grand-challenge.org/)
2. Extract the dataset into the `data/endoscapes` directory
3. The directory structure should look like:
   ```
   data/endoscapes/
   ├── train/
   │   ├── video001/
   │   │   ├── annotations.json
   │   │   ├── frame0001.jpg
   │   │   ├── frame0002.jpg
   │   │   └── ...
   │   └── ...
   ├── val/
   │   └── ...
   └── test/
       └── ...
   ```

## Videos for Inference

Place your test videos in the `data/videos` directory for inference. Supported formats include:
- `.mp4`
- `.avi`
- `.mov`

## Procedure Knowledge

The `data/procedure_knowledge.json` file contains structured knowledge about laparoscopic cholecystectomy procedures, including:
- Surgical phases and their descriptions
- Tool descriptions and usage
- Critical anatomical structures
- Common mistakes and their risk levels

## Data Preparation

After downloading the datasets, run the preparation script to convert and organize the data:

```bash
python scripts/prepare_datasets.py
```

This script will:
1. Validate the dataset structures
2. Convert annotations to the required formats
3. Generate train/val/test splits if not already present
4. Create necessary index files

## Note on Dataset Licenses

Please respect the licenses and terms of use for each dataset. The datasets are provided by their respective owners for research purposes only. 