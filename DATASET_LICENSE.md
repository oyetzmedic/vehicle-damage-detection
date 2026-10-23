# Dataset licence and attribution

This project uses the **Damage-on-cars** object-detection dataset published by **brigada4** on Roboflow Universe.

- Source: https://universe.roboflow.com/brigada4/damage-on-cars
- Publisher: Roboflow Universe
- Original task: object detection
- Original classes: Crack, Scratch, Dent, Broken Glass, Broken Light
- Project classes used here: Crack, Scratch, Dent, Broken Glass
- Licence: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Licence text: https://creativecommons.org/licenses/by/4.0/
- Accessed: 2026-10-23

The dataset images and annotations are not redistributed in this repository. Users must obtain the dataset from the source and comply with the CC BY 4.0 attribution requirements.

The preparation script filters the original labels to the four project classes and can deterministically cap the working dataset at 900 images using seed 42.

Dataset citation supplied by the source:

> brigada4. "Damage-on-cars Dataset." Roboflow Universe, 2025.
