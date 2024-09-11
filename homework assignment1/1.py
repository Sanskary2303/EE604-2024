import numpy as np
import cv2
from PIL import Image

# Load image and convert it to numpy array (RGB)
img = Image.open('image.png')
img_cv2 = np.array(img.convert('RGB'))[..., ::-1]

# Thresholds to detect raindrops (high brightness areas)
low_limit = np.array([140, 140, 140])
high_limit = np.array([250, 250, 250])

# Create a binary mask using the defined thresholds
mask = cv2.inRange(img_cv2, low_limit, high_limit)

# Generate the mask as an RGB image to be used for subtraction
rgb_removed = cv2.bitwise_and(img_cv2, img_cv2, mask=mask)

# Subtract the identified raindrops from the original image
img_no_raindrops = img_cv2 - rgb_removed

# Apply dilation to enlarge the masked regions
structure_element = np.ones((3, 3), np.uint8)
dilated_mask = cv2.dilate(mask, structure_element, iterations=2)

# Smooth the image with a median filter to reduce noise
median_smoothed = cv2.medianBlur(img_cv2, 7)

# Apply inpainting to the masked areas to interpolate missing information
final_img_inpaint = cv2.inpaint(median_smoothed, dilated_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

# Convert BGR to RGB for final display
final_inpaint_rgb = final_img_inpaint[..., ::-1]

# Display final image in OpenCV window
cv2.imshow('Inpainted Image', final_inpaint_rgb[:, :, ::-1])
cv2.imwrite('output.png', final_inpaint_rgb[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()
