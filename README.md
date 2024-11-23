# Skin Detection and Face Analysis

This homework processes images to detect and analyze skin regions and face-like structures using Python and OpenCV.

## Prerequisites


- Required Python libraries:
  - OpenCV
  - NumPy
  - Scikit-Image
  - Glob
  - OS


## Directory Structure

Place your input images in the `images` folder. The output will be saved in the following directories:
- `results/face_detection/`: Images with detected faces highlighted.
- `results/skin_masks/`: Binary masks of detected skin regions.
- `results/cleaned_masks/`: Cleaned binary skin masks after removing noise.

## Usage

1. Copy all the input images into the `images` folder.
2. Double-click the `run.bat` file to start the script.
3. Processed results will be saved in the `results` directory under respective subfolders.

## Notes

- Adjust the skin color range in the `segment_skin` function if detection results are not satisfactory.
- Output images will have ellipses drawn around detected face-like regions.
