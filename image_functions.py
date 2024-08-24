import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.filters import sobel
from skimage.morphology import reconstruction
from skimage.measure import regionprops
from scipy.signal import find_peaks
from PIL import Image, ImageFilter
import cv2
import os


def read_image(path):
    """Reads an image from the given path."""
    image = plt.imread(path)
    return image


def convert_grayscale(image):
    """Converts the given RGB image to grayscale using the luminosity method."""
    # Apply luminosity method to convert RGB to grayscale
    grayscale_image = np.dot(image[..., :3], [0.299, 0.587, 0.114])

    # Ensure values are within the valid range [0, 255]
    #grayscale_image = np.clip(grayscale_image, 0, 255)

    # Convert to uint8 type for consistency (optional)
    grayscale_image = grayscale_image.astype(np.uint8)

    return grayscale_image


def display_image(image, title=None):
    """Displays the given image with an optional title."""
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


def display_resized_image(image, title=None, size=(4, 4)):
    """Displays the given image with an optional title and resizes it."""
    plt.rcParams["figure.figsize"] = size
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.show()


def display_image_grid(images, titles, heading=None):
    """Displays a grid of images with the given titles and an optional heading."""
    plt.figure(figsize=(16, 4))
    plt.title("Image Grid")
    plt.suptitle(heading)
    for i, (img, title) in enumerate(zip(images, titles), 1):
        plt.subplot(1, len(images), i)
        plt.imshow(img)
        plt.title(title)
    plt.show()


def estimate_markers(image):
    """Estimates markers for watershed segmentation based on local maxima."""
    coordinates = peak_local_max(image, min_distance=20, threshold_abs=0.1)
    markers = np.zeros_like(image, dtype=int)
    markers[tuple(coordinates.T)] = np.arange(1, len(coordinates) + 1)
    return markers


def imgradient(image):
    """Computes the gradient magnitude of the given image."""
    # Compute gradients along x and y axes
    dx = sobel(image, axis=0)
    dy = sobel(image, axis=1)

    # Compute gradient magnitude
    gmag = np.sqrt(dx**2 + dy**2)

    return gmag


def imposemin(img, minima):
    """
    Applies the 'imposemin' operation to the given image.
    """
    # Create marker image
    marker = np.full(img.shape, np.inf)
    marker[minima == 1] = 0

    # Create mask
    mask = np.minimum((img + 1), marker)

    # Apply reconstruction with erosion
    result = reconstruction(marker, mask, method="erosion")

    return result


min_size = 600  # Minimum number of pixels in the region to be considered as a candidate
min_aspect_ratio = 3 # Minimum aspect ratio to be considered as a license plate
max_aspect_ratio = 5  # Maximum aspect ratio to be considered as a license plate


def select_license_plate_regions(labels):
    """
    Selects potential license plate regions from the labeled image.
    """
    # Initialize a mask to store potential license plate regions
    candidate_labels = np.zeros_like(labels)

    # Iterate over each region in the labeled image
    for region in regionprops(labels):
        # Ignore small regions
        if region.area >= min_size:
            # Calculate the aspect ratio of the bounding box of the region
            minr, minc, maxr, maxc = region.bbox
            aspect_ratio = (maxc - minc) / (maxr - minr)

            # Check if the aspect ratio falls within the specified range
            if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
                # This region is a candidate for being a license plate
                candidate_labels[labels == region.label] = region.label

    return candidate_labels


def compute_mer(segmented_array, label):
    # Find the indices of the pixels that belong to the segment
    y_indices, x_indices = np.where(np.all(segmented_array == label, axis=-1))

    # Find the minimum and maximum x and y coordinates
    min_x = np.min(x_indices)
    max_x = np.max(x_indices)
    min_y = np.min(y_indices)
    max_y = np.max(y_indices)

    tl = np.array((min_x, min_y))
    tr = np.array((max_x, min_y))
    bl = np.array((min_x, max_y))
    br = np.array((max_x, max_y))

    coords = tl, tr, bl, br

    # The corners of the MER are (min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)
    return coords


def score_license_plate(vpm, threshold=0.7, min_peak_distance=20, plateau_size=1):

    # Find peaks and consider plateaus
    peaks, properties = find_peaks(
        vpm, height=threshold, distance=min_peak_distance, plateau_size=plateau_size
    )
    peak_heights = vpm[peaks]

    # Calculate the width of the plateaus to include them in the scoring
    widths = properties.get("plateau_sizes", np.zeros_like(peaks))

    num_characters = 7
    max_characters = 14

    # Initialize the score
    score = 0

    # Score for the number of peaks (characters)
    if num_characters <= len(peaks) <= max_characters:
        score += 0.8  # add to the score if the number of characters is within the expected range

    # Score based on the standard deviation of the peak heights
    height_std = np.std(peak_heights)
    if height_std < 0.1:
        score += 0.1 - height_std  # add to the score based on how small the std is

    # Score based on the evenness of the peak spacing
    peak_distances = np.diff(peaks)
    distance_std = np.std(peak_distances)
    if distance_std < 10:
        score += (
            10 - distance_std
        ) * 0.05  # add to the score based on how small the std is

    # Consider the uniformity of the plateau sizes in the scoring
    if widths.size > 0:  # ensure that we have plateau data
        width_std = np.std(widths)
        if width_std < 5:
            score += (
                5 - width_std
            ) * 0.05  # add to the score based on how small the std is

    # Normalize the score to be between 0 and 1
    score = np.clip(score / (1.1 + 0.5), 0, 1)  # updated normalization factor

    return score


def numpy_array_to_pil(image_array):
    # Convert the NumPy array to a PIL Image
    return Image.fromarray(np.uint8(image_array))


def preprocess_image(img):
    # Convert to grayscale
    img = img.convert("L")
    # Resize image to increase font size
    img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
    # Apply threshold to get a binary image
    img = img.point(lambda p: p > 128 and 255)
    # Enhance edges
    img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    return img


def overlay_bounding_box_on_image(image_path, coords):
    image = cv2.imread(image_path)
    # Unpack the coordinates
    tl, tr, bl, br = coords
    xMin, yMin = tl
    xMax, _ = tr
    _, yMax = bl

    # Define the bounding box color and thickness
    color = (0, 255, 0)  # Green color
    thickness = 4

    # Draw the bounding box
    cv2.rectangle(image, (xMin, yMin), (xMax, yMax), color, thickness)

    # Save the image with the bounding box
    cv2.imwrite(os.path.join("uploads", image_path), image)
