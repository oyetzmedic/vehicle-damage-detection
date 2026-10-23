# Vehicle Damage Detection

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

The preparation step keeps the four target classes and uses deterministic sampling/splitting with seed 42.

## Model

The detector uses `torchvision.models.detection.fasterrcnn_resnet50_fpn` with COCO-pretrained weights and a replaced ROI classifier head for the project classes plus background.

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
- number and percentage of detections routed to manual review

mAP measures detector quality across ranked confidence scores. The threshold sweep is reported separately as an operational human-review policy.

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
│   └── infer_folder.py
├── tests/
├── DATASET_LICENSE.md
├── LICENSE
├── requirements.txt
└── README.md
```

## Development

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

Training commands and verified evaluation numbers will be documented after the first reproducible run.

## Limitations

This is a portfolio/learning project, not a production insurance assessment system. Expected limitations include small data volume, a single source domain, label noise, class imbalance, sensitivity to image conditions, and no deployment or claims-decision validation.

## Licence

Code is released under the MIT License. The dataset remains under its original CC BY 4.0 licence and attribution requirements.
