import cv2
import numpy as np
import glob
import os
from skimage import morphology, measure

# Paths
input_path = 'images/*.*'
output_base_path = 'results'

# Create output subdirectories
faces_output_path = os.path.join(output_base_path, 'face_detection')
skin_masks_output_path = os.path.join(output_base_path, 'skin_masks')
cleaned_masks_output_path = os.path.join(output_base_path, 'cleaned_masks')

os.makedirs(faces_output_path, exist_ok=True)
os.makedirs(skin_masks_output_path, exist_ok=True)
os.makedirs(cleaned_masks_output_path, exist_ok=True)


def read_and_preprocess_image(image_file):
    """Read an image file and convert it to RGB color space.

    Args:
        image_file (str): Path to the image file.

    Returns:
        tuple: Original image in BGR and converted image in RGB.
    """
    img = cv2.imread(image_file)
    if img is None:
        print(f"Warning: Unable to read {image_file}. Skipping.")
        return None, None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, img_rgb


def equalize_histogram(img):
    """Apply histogram equalization to the V channel in HSV color space.

    Args:
        img (numpy.ndarray): Original BGR image.

    Returns:
        numpy.ndarray: HSV image with equalized histogram in the V channel.
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
    return img_hsv


def segment_skin(img_hsv):
    """Segment skin regions based on predefined HSV color ranges.

    Args:
        img_hsv (numpy.ndarray): HSV image.

    Returns:
        numpy.ndarray: Binary mask where skin regions are white.
    """
    # Adjusted skin color range in HSV
    lower_skin = np.array([0, 5, 49], dtype=np.uint8)
    upper_skin = np.array([44, 150, 250], dtype=np.uint8)
    # Threshold the HSV image to get only skin colors
    skin_mask = cv2.inRange(img_hsv, lower_skin, upper_skin)
    # Apply Gaussian Blur to reduce noise
    skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)
    return skin_mask


def remove_noise(skin_mask, min_size=200):
    """Remove noise and small objects from the skin mask.

    Args:
        skin_mask (numpy.ndarray): Binary mask of skin regions.
        min_size (int, optional): Minimum size of connected components to keep. Defaults to 200.

    Returns:
        numpy.ndarray: Cleaned binary mask.
    """
    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
    # Remove small connected components
    skin_mask_clean = morphology.remove_small_objects(skin_mask.astype(bool), min_size=min_size)
    skin_mask_clean = skin_mask_clean.astype(np.uint8) * 255
    return skin_mask_clean


def detect_faces(skin_mask_clean):
    """Detect faces by analyzing connected regions in the cleaned skin mask.

    Args:
        skin_mask_clean (numpy.ndarray): Cleaned binary mask.

    Returns:
        list: List of region properties for each connected component.
    """
    # Label connected regions
    labels = measure.label(skin_mask_clean, connectivity=2)
    properties = measure.regionprops(labels)
    return properties


def draw_results(img_rgb, skin_mask, skin_mask_clean, properties, image_file):
    """Draw ellipses around detected faces and save intermediate and final images.

    Args:
        img_rgb (numpy.ndarray): Original RGB image.
        skin_mask (numpy.ndarray): Binary skin mask.
        skin_mask_clean (numpy.ndarray): Cleaned binary skin mask.
        properties (list): List of region properties for connected components.
        image_file (str): Path to the original image file.
    """
    output_image = img_rgb.copy()
    scaling_factor = 0.75

    for prop in properties:
        # Criteria for selecting face-like regions
        if prop.eccentricity < 0.85 and prop.solidity > 0.6:
            y0, x0 = prop.centroid  # (row, col)
            orientation = prop.orientation
            major_axis_length = prop.major_axis_length * scaling_factor
            minor_axis_length = prop.minor_axis_length * scaling_factor

            center = (int(x0), int(y0))  # (x, y) format for OpenCV
            axes = (int(major_axis_length / 2), int(minor_axis_length / 2))

            # Correct angle conversion
            angle = 90 - np.degrees(orientation)

            # Draw the ellipse
            cv2.ellipse(output_image, center, axes, angle, 0, 360, (0, 255, 0), 5)

    base_filename = os.path.splitext(os.path.basename(image_file))[0]
    faces_filename = os.path.join(faces_output_path, f"{base_filename}_face_detection.jpg")
    skin_mask_filename = os.path.join(skin_masks_output_path, f"{base_filename}_skin_mask.jpg")
    cleaned_mask_filename = os.path.join(cleaned_masks_output_path, f"{base_filename}_cleaned_mask.jpg")

    cv2.imwrite(faces_filename, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

    cv2.imwrite(skin_mask_filename, skin_mask)

    cv2.imwrite(cleaned_mask_filename, skin_mask_clean)


def process_images():
    image_files = glob.glob(input_path)
    if not image_files:
        print("No images found. Please check the input path.")
        return
    for image_file in image_files:
        print(f"Processing {image_file}")
        img, img_rgb = read_and_preprocess_image(image_file)
        if img is None:
            continue
        img_hsv = equalize_histogram(img)
        skin_mask = segment_skin(img_hsv)
        skin_mask_clean = remove_noise(skin_mask)
        properties = detect_faces(skin_mask_clean)
        draw_results(img_rgb, skin_mask, skin_mask_clean, properties, image_file)


if __name__ == '__main__':
    process_images()
