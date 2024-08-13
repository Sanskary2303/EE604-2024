import numpy as np
import matplotlib.pyplot as plt

r = 1000

width = 4 * r
height = 4 * r

image = np.ones((height, width), dtype=np.uint8)

center_left = (r, 2 * r)
center_right = (3 * r, 2 * r)
center_top = (2 * r, r)
center_bottom = (2 * r, 3 * r)

def draw_circle(img, center, radius, value):
    y, x = np.ogrid[-center[1]:height-center[1], -center[0]:width-center[0]]
    mask = x*x + y*y <= radius*radius
    img[mask] = value

def draw_square(img, top_left, size, value):
    img[top_left[1]:top_left[1]+size, top_left[0]:top_left[0]+size] = value

def draw_diamond(img, center, size, value):
    for i in range(-size, size):
        img[center[0] - abs(i):center[0] + abs(i),center[1] + i] = value

draw_square(image, (r, r), 2*r, 0)

draw_circle(image, center_top, r, 0)
draw_circle(image, center_bottom, r, 0)

draw_diamond(image, (2*r,0), r, 0)
draw_square(image, (int(1.5*r), int(1.5*r)), r, 1)

plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()
    