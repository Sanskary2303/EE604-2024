import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
image_path = 'lhc.png'
original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# function to compute LBP
def compute_lbp(img):
    lbp_img = np.zeros_like(img)
    
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            center = img[i, j]
            binary_string = ''
            
            # Check each of the 8 neighboring pixels
            for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]:
                neighbor = img[i + dy, j + dx]
                # Append '1' if neighbor is >= center, else '0'
                binary_string += '1' if neighbor >= center else '0'
            
            # Convert the binary string to a decimal integer and assign it to LBP image
            lbp_img[i, j] = int(binary_string, 2)
    return lbp_img

lbp_image = compute_lbp(original_img)

# Divide the image into a 2x2 grid and calculate the histogram for each cell
grid_size = (2, 2)
cell_height, cell_width = original_img.shape[0] // grid_size[0], original_img.shape[1] // grid_size[1]
lbp_histograms = []


for row in range(grid_size[0]):
    for col in range(grid_size[1]):
       
        cell = lbp_image[row * cell_height:(row + 1) * cell_height, col * cell_width:(col + 1) * cell_width]
        
        # Calculate the histogram for the LBP values in the cell
        hist, _ = np.histogram(cell.ravel(), bins=np.arange(257), range=(0, 256))
        lbp_histograms.append(hist)

# Concatenate the histograms to form a global feature vector
global_feature_vector = np.concatenate(lbp_histograms)

# Plot the LBP image and the global feature vector
plt.figure(figsize=(15, 6))

# Display the LBP image
plt.subplot(1, 2, 1)
plt.imshow(lbp_image, cmap='gray')
plt.title("LBP Image")
plt.axis('off')

# Plot the global feature vector
plt.subplot(1, 2, 2)
plt.plot(global_feature_vector, color='orange')
plt.title("Global Feature Vector")
plt.xlabel("Feature Index")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
