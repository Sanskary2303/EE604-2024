import numpy as np
import matplotlib.pyplot as plt

# Function to create a circle on the image at a specified position and with a specific color
def create_circle(img, center, radius, color):
    # Create a grid of coordinates (y, x) for a 400x400 image
    y, x = np.ogrid[:400, :400]
    
    # Compute the distance from the center of the circle
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    
    # Create a mask where the distance is less than or equal to the radius
    mask = dist_from_center <= radius
    
    # Apply the color to the masked area (the circle) on the image
    img[mask] = color

# Function to create an RGB image with four colored circles
def create_rgb_image():
    # Initialize a 400x400 image with a white background (RGB values are 1.0 for white)
    img = np.ones((400, 400, 3), dtype=np.float32)
    
    # Define the colors for the circles in RGB (Cyan, Magenta, Yellow, Black)
    colors = [(0, 1, 1), (1, 0, 1), (1, 1, 0), (0, 0, 0)]  # CMYK represented in RGB
    
    # Define the positions (centers) of the four circles
    centers = [(100, 100), (300, 100), (100, 300), (300, 300)]
    
    # Loop through each circle's position and color and draw the circle on the image
    for center, color in zip(centers, colors):
        create_circle(img, center, 50, color)  # Radius of 50
    
    return img  # Return the image with the circles

# Function to convert an RGB image to CMYK
def rgb_to_cmyk(rgb):
    # Extract the red, green, and blue channels
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    
    # Compute the black (K) channel as 1 minus the maximum value across R, G, B
    k = 1 - np.max(rgb, axis=2)
    
    # Compute the cyan (C), magenta (M), and yellow (Y) channels
    # Using 1 - color component, adjusted by the black (K) value
    c = (1 - r - k) / (1 - k + 1e-8)  # Added small value to avoid division by zero
    m = (1 - g - k) / (1 - k + 1e-8)
    y = (1 - b - k) / (1 - k + 1e-8)
    
    # Stack the C, M, Y, and K channels together to form a CMYK image
    return np.stack([c, m, y, k], axis=-1)

# Function to save an image using matplotlib's imsave function
def save_image(img, filename):
    # Clip the image values to [0, 1] to ensure it's within the valid range, then save
    plt.imsave(filename, np.clip(img, 0, 1))

# Main function to create and save images
def main():
    # Create the RGB image with the circles
    rgb_img = create_rgb_image()
    
    # Save the RGB image to a file
    save_image(rgb_img, 'rgb_image.png')
    
    # Convert the RGB image to CMYK
    cmyk_img = rgb_to_cmyk(rgb_img)
    
    # Define names for the CMYK channels
    channel_names = ['cyan', 'magenta', 'yellow', 'black']
    
    # Loop through each channel (C, M, Y, K), save and display the channels
    for i, name in enumerate(channel_names):
        channel = cmyk_img[:, :, i]  # Extract the specific channel
        
        # Save each CMYK channel as a separate grayscale image
        plt.figure(figsize=(5, 5))  # Create a figure
        plt.imshow(channel, cmap='gray')  # Display the channel as grayscale
        plt.axis('off')  # Hide the axes
        plt.title(f'{name.capitalize()} Channel')  # Add a title
        plt.savefig(f'{name}_channel_only.png', bbox_inches='tight', pad_inches=0.1)  # Save the image
        plt.close()  # Close the figure after saving

    # Display all images (RGB and CMYK channels) in a grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Create a 2x3 grid of subplots
    
    # Show the RGB image in the first subplot
    axes[0, 0].imshow(rgb_img)
    axes[0, 0].set_title('RGB Image')  # Title for RGB image
    
    # Show each CMYK channel in the subsequent subplots
    for i, name in enumerate(channel_names):
        ax = axes[(i+1)//3, (i+1)%3]  # Select the appropriate subplot
        ax.imshow(1 - cmyk_img[:, :, i], cmap='gray')  # Show the inverted channel
        ax.set_title(f'{name.capitalize()} Channel')  # Set the title for the channel
    
    # Turn off axes for all subplots
    for ax in axes.flat:
        ax.axis('off')
    
    # Adjust layout for better appearance
    plt.tight_layout()
    
    # Save the composite image showing all channels and the RGB image
    plt.savefig('all_images.png')
    plt.close()  # Close the figure

# Run the main function if this script is executed
if __name__ == "__main__":
    main()
