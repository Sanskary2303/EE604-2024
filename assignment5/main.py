import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.color import rgb2lab
from skimage import io, img_as_float
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.exposure import rescale_intensity

# Load the image
image_path = 'image.jpg'
image = io.imread(image_path)

# Convert image to LAB color space
lab_image = rgb2lab(image)

# Generate SLIC superpixels with more segments (increase detail)
num_segments = 500  # Increase to capture finer details
segments = slic(lab_image, n_segments=num_segments, sigma=5)

# Get unique superpixels
unique_segments = np.unique(segments)

# Calculate the center and average color of each superpixel
centers = []
avg_colors = []

for segment in unique_segments:
    mask = segments == segment
    coordinates = np.argwhere(mask)
    center = np.mean(coordinates, axis=0)
    avg_color = np.mean(lab_image[mask], axis=0)
    centers.append(center)
    avg_colors.append(avg_color)

centers = np.array(centers)
avg_colors = np.array(avg_colors)

# Calculate color_distance and spatial_distance
color_distances = cdist(avg_colors, avg_colors, metric='euclidean')
spatial_distances = cdist(centers, centers, metric='euclidean')

# Adjust weighting between color and spatial distances
alpha = 0.8  # Give more weight to color, can tweak this value
beta = 0.2  # Give less weight to spatial distance
effective_distances = alpha * color_distances + beta * np.exp(-spatial_distances)

# Calculate saliency values by summing the effective distances
saliency_values = np.sum(effective_distances, axis=1)

# Normalize saliency values
saliency_values = (saliency_values - np.min(saliency_values)) / (np.max(saliency_values) - np.min(saliency_values))

# Assign saliency values to the pixels in the image
saliency_map = np.zeros(image.shape[:2])
for i, segment in enumerate(unique_segments):
    saliency_map[segments == segment] = saliency_values[i]

# Background suppression (Assume background is near edges)
border_region_size = 0.1  # Suppress saliency near the borders
rows, cols = saliency_map.shape
border_mask = np.zeros_like(saliency_map)

# Suppress saliency near borders (top, bottom, left, right)
border_mask[:int(border_region_size*rows), :] = 1
border_mask[-int(border_region_size*rows):, :] = 1
border_mask[:, :int(border_region_size*cols)] = 1
border_mask[:, -int(border_region_size*cols):] = 1
saliency_map = saliency_map * (1 - border_mask)

# Apply Gaussian blur to smooth the saliency map
saliency_map_blurred = gaussian(saliency_map, sigma=2)

# Rescale intensity to enhance contrast
saliency_map_rescaled = rescale_intensity(saliency_map_blurred, out_range=(0, 1))

# Apply thresholding to highlight the bird
threshold = 0.45
saliency_map_thresholded = (saliency_map_rescaled > threshold).astype(float)

# Display the image with superpixel boundaries
plt.figure(figsize=(10, 10))
plt.imshow(mark_boundaries(image, segments))
plt.title(f"Superpixels with {num_segments} Segments")
plt.axis("off")

# Display the result
plt.figure(figsize=(10, 5))

# Enhanced Saliency Map
plt.subplot(1, 2, 1)
plt.imshow(saliency_map_rescaled, cmap='gray')
plt.title("Enhanced Saliency Map (More Superpixels)")
plt.axis('off')

# Thresholded Saliency Map
plt.subplot(1, 2, 2)
plt.imshow(saliency_map_thresholded, cmap='gray')
plt.title("Thresholded Map (With Background Suppression)")
plt.axis('off')

plt.show()
