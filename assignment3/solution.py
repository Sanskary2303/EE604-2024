# Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('image.jpeg', 0)

# Fourier transform
f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# Create Gaussian high-pass filter
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2 # Center of the image
sigma = 10  # Standard deviation of Gaussian

# Normalized coordinates
x = np.linspace(-0.5, 0.5, cols)
y = np.linspace(-0.5, 0.5, rows) 

# Create a grid
x, y = np.meshgrid(x, y)
d = np.sqrt(x*x + y*y) # Euclidean distance
gaussian_hp = 1 - np.exp(-(d**2) / (2 * (sigma**2))) # Gaussian high-pass filter

# Apply the filter
fshift_filtered = fshift * gaussian_hp
magnitude_spectrum_filtered = 20 * np.log(np.abs(fshift_filtered)) 

# Inverse Fourier transform
f_ishift = np.fft.ifftshift(fshift_filtered)
image_back = np.fft.ifft2(f_ishift) 
image_back = np.abs(image_back)

# Plot all the images/spectrums
plt.figure(figsize=(15, 10))

# Original Image
plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Fourier Transform (magnitude spectrum)
plt.subplot(2, 3, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum (Original)')
plt.axis('off')

# Gaussian High-Pass Filter
plt.subplot(2, 3, 3)
plt.imshow(gaussian_hp, cmap='gray')
plt.title('Gaussian High-Pass Filter')
plt.axis('off')

# Spectrum after applying Gaussian HP filter
plt.subplot(2, 3, 4)
plt.imshow(magnitude_spectrum_filtered, cmap='gray')
plt.title('Magnitude Spectrum (Filtered)')
plt.axis('off')

# Inverse Fourier Transform
plt.subplot(2, 3, 5)
plt.imshow(image_back, cmap='gray')
plt.title('Filtered Image (Spatial Domain)')
plt.axis('off')

# Fourier Transform of Filtered Image
f_transformed = np.fft.fftshift(np.fft.fft2(image_back))
magnitude_spectrum_transformed = 20 * np.log(np.abs(f_transformed))
plt.subplot(2, 3, 6)
plt.imshow(magnitude_spectrum_transformed, cmap='gray')
plt.title('Magnitude Spectrum (Transformed)')
plt.axis('off')

# Save the plot to a PDF file
plt.savefig('/home/sanskar/EE604/assignment3/plot.pdf')
plt.show()

