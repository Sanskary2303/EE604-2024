import numpy as np
import cv2
import math

# Function to draw a curve based on polar coordinates
def draw_curve(image, center, radius, angle_start, angle_end, color, thickness, verbose=False):
    # Adjust for angle wrapping
    if angle_end < angle_start:
        angle_end += 360

    # Number of points to approximate the curve
    num_points = 1000
    step = (angle_end - angle_start) / num_points

    # List to store curve points
    curve_points = []

    # Generate the curve points based on angle and radius
    for i in range(num_points + 1):
        angle = angle_start + i * step
        angle = angle % 360  # Keep angle within 0-360 degrees
        x = int(center[0] + radius * math.cos(math.radians(angle)))
        y = int(center[1] + radius * math.sin(math.radians(angle)))
        curve_points.append((x, y))

    # If verbose mode is enabled, print the first and last points
    if verbose:
        print(f"Initial : {curve_points[0]}")
        print(f"Final : {curve_points[-1]}")

    # Draw the curve by connecting the points
    for i in range(len(curve_points) - 1):
        cv2.line(image, curve_points[i], curve_points[i + 1], color, thickness)

# Define the dimensions of the canvas
width = 512
height = 512

# Create an empty white image (logo)
logo = np.ones((height, width), dtype=np.uint8) * 255

# Define the center coordinates of the logo
x_centre = width // 2
y_centre = height // 2

# Draw three concentric circles to form the outer, middle, and inner rings
outer_r = 160
draw_curve(logo, (x_centre, y_centre), outer_r, 0, 360, 0, 4)

middle_r = 120
draw_curve(logo, (x_centre, y_centre), middle_r, 0, 360, 0, 4)

inner_r = 80
draw_curve(logo, (x_centre, y_centre), inner_r, 0, 360, 0, 4)

# Define two radii to create spoke shapes
r1 = 90
r2 = 105

# Define step angle to draw 24 spokes
step = 360 // 24

# Draw the spokes using lines between two radii
for i in range(24):
    angle = i * step

    # Calculate points for the pentagonal spoke sections
    point_1_x = int(x_centre + r1 * math.cos(math.radians(angle)))
    point_1_y = int(y_centre + r1 * math.sin(math.radians(angle)))

    point_2_x = int(x_centre + r1 * math.cos(math.radians(angle + step / 2)))
    point_2_y = int(y_centre + r1 * math.sin(math.radians(angle + step / 2)))

    point_3_x = int(x_centre + r2 * math.cos(math.radians(angle + step / 2)))
    point_3_y = int(y_centre + r2 * math.sin(math.radians(angle + step / 2)))

    point_4_x = int(x_centre + r2 * math.cos(math.radians(angle + step)))
    point_4_y = int(y_centre + r2 * math.sin(math.radians(angle + step)))

    point_5_x = int(x_centre + r1 * math.cos(math.radians(angle + step)))
    point_5_y = int(y_centre + r1 * math.sin(math.radians(angle + step)))

    # Draw lines between the calculated points to form the spokes
    cv2.line(logo, (point_1_x, point_1_y), (point_2_x, point_2_y), 0, 2)
    cv2.line(logo, (point_2_x, point_2_y), (point_3_x, point_3_y), 0, 2)
    cv2.line(logo, (point_3_x, point_3_y), (point_4_x, point_4_y), 0, 2)
    cv2.line(logo, (point_4_x, point_4_y), (point_5_x, point_5_y), 0, 2)

# Draw smaller curved sections on either side of the inner circle
draw_curve(logo, (x_centre+inner_r, y_centre), 100, 160, 217, 0, 2)
draw_curve(logo, (x_centre-inner_r, y_centre), 100, 323, 20, 0, 2)

# Additional curves to enhance the design near the middle
draw_curve(logo, (x_centre+51, y_centre), 20, 140, 310, 0, 2)
draw_curve(logo, (x_centre-51, y_centre), 20, 230, 40, 0, 2)

draw_curve(logo, (x_centre+15, y_centre+15), 20, 330, 100, 0, 2)
draw_curve(logo, (x_centre-15, y_centre+15), 20, 80, 210, 0, 2)

# Another set of curves at the center
draw_curve(logo, (x_centre, y_centre), 57, 5, 75, 0, 2)
draw_curve(logo, (x_centre, y_centre), 57, 105, 175, 0, 2)

# Calculate the position for the small circles
right_x = int(x_centre + 140 * math.cos(math.radians(step)))
right_y = int(y_centre - 140 * math.sin(math.radians(step)))

left_x = int(x_centre - 140 * math.cos(math.radians(step)))
left_y = int(y_centre - 140 * math.sin(math.radians(step)))

# Draw additional lines and rectangles near the center
cv2.line(logo, (x_centre+15, y_centre+55), (x_centre-15, y_centre+55), 0, 2)
cv2.line(logo, (x_centre+15, y_centre+55), (x_centre+15, y_centre+68), 0, 2)
cv2.line(logo, (x_centre-15, y_centre+55), (x_centre-15, y_centre+68), 0, 2)
cv2.line(logo, (x_centre+15, y_centre+68), (x_centre-15, y_centre+68), 0, 2)

# Additional curves on both sides of the center
draw_curve(logo, (x_centre+61, y_centre-5), 10, 110, 310, 0, 2)
draw_curve(logo, (x_centre-61, y_centre-5), 10, 210, 55, 0, 2)

# Final curves to complete the design
draw_curve(logo, (x_centre-40, y_centre), 50, 320, 40, 0, 2)
draw_curve(logo, (x_centre+40, y_centre), 50, 140, 220, 0, 2)

# Draw small black circles at specific positions
cv2.circle(logo, (right_x, right_y), 5, 0, -1)
cv2.circle(logo, (left_x, left_y), 5, 0, -1)
cv2.circle(logo, (x_centre, y_centre), 4, 0, -1)
cv2.circle(logo, (x_centre+45, y_centre+10), 3, 0, -1)
cv2.circle(logo, (x_centre-45, y_centre+10), 3, 0, -1)

# Display the generated logo in a window
cv2.imshow("IITK Logo", logo)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the logo as an image file
cv2.imwrite('iitk_logo.png', logo)
