# Vehicle Damage Detection

[![Tests](https://github.com/oyetzmedic/vehicle-damage-detection/actions/workflows/tests.yml/badge.svg)](https://github.com/oyetzmedic/vehicle-damage-detection/actions/workflows/tests.yml)

PyTorch object-detection project for identifying visible vehicle damage in photographs.

**Status:** build in progress. Training and evaluation results will be added only after they are produced and verified.

## Goal

Fine-tune a COCO-pretrained Faster R-CNN ResNet50-FPN detector on four vehicle-damage classes:

- Crack
- Scratch
- Dent
- Broken Glass

The project also includes a simple human-in-the-loop routing policy so low-confidence detections can be flagged for manual review rather than treated as reliable automated decisions.

## Dataset

Source: **Damage-on-cars** by **brigada4** on Roboflow Universe.

- Task: object detection
- Source images listed by Roboflow Universe: 1,025
- Original labels: Crack, Scratch, Dent, Broken Glass, Broken Light
- Licence: CC BY 4.0
- Project URL: https://universe.roboflow.com/brigada4/damage-on-cars

This repository does **not** commit the dataset images. See `DATASET_LICENSE.md` and `scripts/download_data.py`.

The preparation step keeps the four target classes, caps the working set at up to 900 eligible images, uses seed 42 for deterministic sampling, and preserves the source train/validation/test membership.

## Model

The detector uses `torchvision.models.detection.fasterrcnn_resnet50_fpn` with COCO-pretrained weights and a replaced ROI classifier head for the four project classes plus background.

For the reference development machine, the model uses a low-memory image transform of **384 px minimum / 640 px maximum**, reduced RPN proposal counts, mixed-precision CUDA training, and a default **batch size of 1** so Faster R-CNN can train on a 4 GB GTX 1050 Ti.

## OpenCV

OpenCV is used for:

- image loading and resizing
- horizontal-flip and brightness augmentation
- bounding-box adjustment and clipping
- drawing predicted boxes, labels and confidence scores
- folder inference that writes annotated output images

## Evaluation plan

The held-out test split will report:

- mAP@0.5
- overall mAP
- per-class AP
- confidence-threshold sweep
- number and percentage of candidate detections routed to manual review

mAP measures detector quality across ranked confidence scores. The confidence-threshold sweep is reported separately as an operational human-review policy.

## Repository layout

```text
vehicle-damage-detection/
├── .github/workflows/tests.yml
├── src/vehicle_damage/
│   ├── dataset.py
│   ├── model.py
│   ├── routing.py
│   ├── transforms.py
│   └── visualize.py
├── scripts/
│   ├── download_data.py
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── infer_folder.py
├── tests/test_utils.py
├── DATASET_LICENSE.md
├── LICENSE
├── requirements-torch-cu121.txt
├── requirements.txt
├── requirements-test.txt
├── pytest.ini
├── pyproject.toml
└── README.md
```

## Setup

The reference Windows/CUDA environment uses PyTorch 2.5.1 and torchvision 0.20.1 with CUDA 12.1:

```bash
python -m pip install -r requirements-torch-cu121.txt
python -m pip install -r requirements.txt
python -m pip install -e .
```

The CUDA-specific file is separate so the non-PyTorch dependencies can also be installed independently on other machines.

Set your Roboflow API key in the environment, then download and prepare the data:

```bash
python scripts/download_data.py
python scripts/prepare_data.py --max-images 900 --seed 42
```

Train the detector:

```bash
python scripts/train.py --epochs 5 --batch-size 1
```

Evaluate the held-out test split:

```bash
python scripts/evaluate.py
```

Run folder inference and write annotated images:

```bash
python scripts/infer_folder.py path/to/images --threshold 0.5
```

Run the hermetic utility tests:

```bash
python -m pip install -r requirements-test.txt
python -m pytest
```

## Results

Evaluation numbers are intentionally not reported yet. This section will be updated only after a reproducible fine-tuning run and held-out evaluation.

## Limitations

This is a portfolio/learning project, not a production insurance assessment system. Expected limitations include small data volume, a single source domain, label noise, class imbalance, sensitivity to image conditions, and no deployment or claims-decision validation.

## Licence

Code is released under the MIT License. The dataset remains under its original CC BY 4.0 licence and attribution requirements.
