# Vehicle Damage Detection

[![Tests](https://github.com/oyetzmedic/vehicle-damage-detection/actions/workflows/tests.yml/badge.svg)](https://github.com/oyetzmedic/vehicle-damage-detection/actions/workflows/tests.yml)

PyTorch object-detection project for identifying visible vehicle damage in photographs.

**Status:** reproducible baseline complete. The final checkpoint was selected using the validation split, then evaluated once on the held-out test split.

## Goal

Fine-tune a COCO-pretrained Faster R-CNN ResNet50-FPN detector on four vehicle-damage classes:

- Crack
- Scratch
- Dent
- Broken Glass

The project also includes a simple human-in-the-loop routing analysis so low-confidence candidate detections can be flagged for manual review rather than treated as reliable automated decisions.

## Dataset

Source: **Damage-on-cars** by **brigada4** on Roboflow Universe.

- Task: object detection
- Roboflow dataset version used: **3**
- COCO export downloaded for this run: **726 images**
- Source split sizes: 508 train / 145 validation / 73 test
- Source categories in the export: cars, Broken Glass, Broken Light, Crack, Damaged Bumper, Deflated tire, Dent, Scratch
- Project classes: Crack, Scratch, Dent, Broken Glass
- Licence: CC BY 4.0
- Project URL: https://universe.roboflow.com/brigada4/damage-on-cars

This repository does **not** commit the dataset images. See `DATASET_LICENSE.md` and `scripts/download_data.py`.

After filtering to images containing at least one of the four project classes, the working dataset contains **683 images** and **1,815 target annotations**:

| Split | Images | Target annotations |
| --- | ---: | ---: |
| Train | 477 | 1,254 |
| Validation | 140 | 388 |
| Test | 66 | 173 |

The training data are imbalanced. Scratch is the most frequent class, while Crack is the rarest, with 66 train, 14 validation and 8 test annotations.

## Model

The detector uses Faster R-CNN ResNet50-FPN with official COCO-pretrained torchvision weights and a replaced ROI classifier head for four damage classes plus background.

For the reference development machine, the model uses a low-memory profile:

- 384 px minimum / 640 px maximum image transform
- reduced RPN proposal counts
- mixed-precision CUDA training
- batch size 1
- Windows-friendly weight loading order to reduce peak CPU/virtual-memory pressure

Reference hardware: NVIDIA GeForce GTX 1050 Ti, 4 GB VRAM.

## OpenCV

OpenCV is used for:

- image loading and resizing
- horizontal-flip and brightness augmentation
- bounding-box adjustment and clipping
- drawing predicted boxes, labels and confidence scores
- folder inference that writes annotated output images

## Model selection

The test split was not used to select the checkpoint. Epoch count was selected on the 140-image validation split.

| Checkpoint | mAP | mAP@0.5 | mAP@0.75 | Crack AP | Scratch AP | Dent AP | Broken Glass AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Epoch 1 | 0.0470 | 0.1508 | 0.0137 | 0.0000 | 0.0388 | 0.0340 | 0.1153 |
| Epoch 3 | 0.0746 | 0.2010 | 0.0490 | 0.0000 | 0.0425 | 0.0852 | 0.1706 |
| **Epoch 5** | **0.0903** | **0.2401** | **0.0576** | **0.0179** | **0.0493** | **0.1080** | **0.1860** |

Epoch 5 was selected because it achieved the strongest validation performance across the tracked aggregate metrics and all four class AP values.

## Held-out test results

The selected epoch-5 checkpoint was evaluated once on the 66-image held-out test split.

| Metric | Test result |
| --- | ---: |
| mAP | **0.1039** |
| mAP@0.5 | **0.2727** |
| mAP@0.75 | **0.0664** |
| Crack AP | 0.0276 |
| Scratch AP | 0.0565 |
| Dent AP | 0.0670 |
| Broken Glass AP | **0.2643** |

The strongest class is Broken Glass. Crack remains the weakest class, consistent with its small number of training and evaluation examples.

Exact metric outputs are retained under `results/`.

## Human-review threshold analysis

The evaluator also reports a confidence-threshold sweep over candidate detections with score >= 0.05.

For the final test run, at a threshold of 0.50:

- candidate detections: 787
- detections below 0.50 and routed to review: 722
- detections at or above 0.50: 65
- review rate: 91.7%

This is a **candidate-detection routing analysis**, not a calibrated claims-level automation rule. It does not measure precision/recall at each threshold and does not account for missed objects with no candidate detection.

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
├── results/
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

Set your Roboflow API key in the environment, then download and prepare the data:

```bash
python scripts/download_data.py
python scripts/prepare_data.py --max-images 900 --seed 42
```

Because version 3 contains only 683 eligible images after filtering, the 900-image cap does not remove any eligible image in this run.

Train the detector:

```bash
python scripts/train.py --epochs 5 --batch-size 1
```

Evaluate validation or test:

```bash
python scripts/evaluate.py --split valid --checkpoint checkpoints/fasterrcnn.pt
python scripts/evaluate.py --split test --checkpoint checkpoints/fasterrcnn.pt
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

## Limitations

This is a portfolio/learning baseline, not a production insurance assessment system. Key limitations include:

- small single-source dataset
- strong class imbalance, especially limited Crack examples
- low absolute AP values for several classes
- no cross-dataset or external-domain validation
- no calibrated case-level decision threshold
- no deployment, latency, robustness or claims-decision validation
- CI covers hermetic utilities rather than the full GPU training path

## Licence

Code is released under the MIT License. The dataset remains under its original CC BY 4.0 licence and attribution requirements.
