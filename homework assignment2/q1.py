import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

def preprocess_image(image):
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur
    blurred = ndimage.gaussian_filter(gray, sigma=1)
    
    # Detect edges using Sobel
    dx = ndimage.sobel(blurred, axis=0)
    dy = ndimage.sobel(blurred, axis=1)
    edges = np.hypot(dx, dy)
    edges = edges / edges.max()
    edges = (edges > 0.2).astype(np.uint8)
    
    return edges

def hough_circle_transform(edges, r_min, r_max, step=1):
    height, width = edges.shape
    radii = np.arange(r_min, r_max, step)
    accumulator = np.zeros((height, width, len(radii)))
    
    # Get edge points
    y_idx, x_idx = np.nonzero(edges)
    
    # Create lookup tables for sine and cosine
    theta = np.linspace(0, 2*np.pi, 360)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Vote in accumulator
    for i, r in enumerate(radii):
        for x, y in zip(x_idx, y_idx):
            # Calculate circle points
            a = (x - r * cos_theta).astype(np.int32)
            b = (y - r * sin_theta).astype(np.int32)
            
            # Keep only valid points
            valid = (a >= 0) & (a < width) & (b >= 0) & (b < height)
            accumulator[b[valid], a[valid], i] += 1
            
    return accumulator, radii

def detect_circles(accumulator, radii, threshold):
    circles = []
    threshold = np.max(accumulator) * 0.5
    for i, r in enumerate(radii):
        layer = accumulator[:, :, i]
        peaks = (layer > threshold) & (layer == ndimage.maximum_filter(layer, size=10))
        y_peaks, x_peaks = np.nonzero(peaks)
        for x, y in zip(x_peaks, y_peaks):
            circles.append((x, y, r))
    return circles

def create_binary_mask(shape, circles):
    mask = np.zeros(shape, dtype=np.uint8)
    y_coords, x_coords = np.ogrid[:shape[0], :shape[1]]
    
    for x, y, r in circles:
        # Create circle mask
        circle_mask = (x_coords - x)**2 + (y_coords - y)**2 <= r**2
        mask |= circle_mask
    
    return mask

def detect_coins(image):
    # Preprocess image
    edges = preprocess_image(image)
    
    # Apply Hough Circle Transform
    r_min, r_max = 15, 90
    accumulator, radii = hough_circle_transform(edges, r_min, r_max, step=2)
    
    # Detect circles
    circles = detect_circles(accumulator, radii, threshold=15)

    if len(circles) == 0:  # Debug check
        print("No circles detected!")
        return np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Create binary mask
    mask = create_binary_mask(image.shape[:2], circles)
    
    return mask

# Example usage
if __name__ == "__main__":
    # Load image
    image = plt.imread('coins.jpg')
    
    # Detect coins and create mask
    mask = detect_coins(image)
    
    # Visualize results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(mask, cmap='gray')
    plt.title('Binary Mask')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(image)
    plt.imshow(mask, alpha=0.5, cmap='Reds')
    plt.title('Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()