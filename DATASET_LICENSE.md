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

The raw dataset images and annotations are not redistributed in this repository. Four modified derivative images are included under `docs/portfolio/` solely as qualitative portfolio examples. These derivatives originate from the Damage-on-cars dataset and have been modified by adding model-prediction boxes, labels, confidence values and/or diagnostic overlays.

The preparation script filters the source labels to the four project classes and can deterministically cap the working dataset using seed 42. The 900-image cap used in this run did not exclude any eligible image because only 683 images remained after filtering.

Dataset citation supplied by the source:

> brigada4. "Damage-on-cars Dataset." Roboflow Universe, 2025.


## Portfolio image attribution

The following repository files are modified derivatives of images from the **Damage-on-cars** dataset by **brigada4**:

- `docs/portfolio/broken_glass_success.jpg`
- `docs/portfolio/dent_success.jpg`
- `docs/portfolio/scratch_success.jpg`
- `docs/portfolio/crack_diagnostic_clean.jpg`

Original dataset source: https://universe.roboflow.com/brigada4/damage-on-cars

Original material licence: Creative Commons Attribution 4.0 International (CC BY 4.0): https://creativecommons.org/licenses/by/4.0/

Changes made: model-prediction bounding boxes, class labels, confidence values and/or failure-analysis diagnostic overlays were added. No endorsement by the original dataset creator or Roboflow is implied.

The repository's MIT licence applies to the project code, not to these CC BY 4.0-derived images.
