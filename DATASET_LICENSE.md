# Dataset licence and attribution

This project uses the **Damage-on-cars** object-detection dataset published by **brigada4** on Roboflow Universe.

- Source: https://universe.roboflow.com/brigada4/damage-on-cars
- Publisher: Roboflow Universe
- Dataset version used: 3
- Original task: object detection
- Categories present in the downloaded COCO export: cars, Broken Glass, Broken Light, Crack, Damaged Bumper, Deflated tire, Dent, Scratch
- Project classes used here: Crack, Scratch, Dent, Broken Glass
- Licence: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Licence text: https://creativecommons.org/licenses/by/4.0/
- Export inspected: 2026-10-28

The version-3 COCO export used for this run contained 726 images across the source train/validation/test splits. After filtering to images containing at least one project class, the working dataset contained 683 images and 1,815 target annotations.

The dataset images and annotations are not redistributed in this repository. Users must obtain the dataset from the source and comply with the CC BY 4.0 attribution requirements.

The preparation script filters the source labels to the four project classes and can deterministically cap the working dataset using seed 42. The 900-image cap used in this run did not exclude any eligible image because only 683 images remained after filtering.

Dataset citation supplied by the source:

> brigada4. "Damage-on-cars Dataset." Roboflow Universe, 2025.
