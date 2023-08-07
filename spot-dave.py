import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def find_template_in_image(sample_image_path, field_image_path):
    # Read the sample and field images
    sample_img = cv2.imread(sample_image_path)
    field_img = cv2.imread(field_image_path)

    # Convert images to RGB color space for advanced color processing
    sample_img_rgb = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
    field_img_rgb = cv2.cvtColor(field_img, cv2.COLOR_BGR2RGB)

    # Define the range of scales to use for smooth variation
    min_scale = 0.2
    max_scale = 2.0
    num_scales = 100  # Adjust the number of scales as needed

    # Generate a smooth and continuous scale variation
    scales = np.linspace(min_scale, max_scale, num_scales)

    best_match_val = 0
    best_match_coords = None
    best_match_scale = 1.0

    for scale in scales:
        # Resize the sample image to the current scale
        resized_sample_img = cv2.resize(sample_img_rgb, None, fx=scale, fy=scale)

        # Apply color-based template matching using normalized cross-correlation
        result = cv2.matchTemplate(field_img_rgb, resized_sample_img, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Keep track of the best match across all scales
        if max_val > best_match_val:
            best_match_val = max_val
            best_match_coords = max_loc
            best_match_scale = scale

    # Threshold for matching
    threshold = 0.5

    if best_match_val >= threshold:
        print("Sample image found in the field image!")

        # Calculate the coordinates of the rectangle overlay at the detected location
        x, y = best_match_coords
        w = int(sample_img.shape[1] * best_match_scale)
        h = int(sample_img.shape[0] * best_match_scale)

        # Create a copy of the field image to draw the rectangle overlay
        field_img_with_overlay = field_img.copy()

        # Draw a bounding rectangle overlay on the copy of the field image
        cv2.rectangle(field_img_with_overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the result
        plt.subplot(121), plt.imshow(result, cmap='gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(field_img_with_overlay)
        plt.title('Field Image with Rectangle Overlay'), plt.xticks([]), plt.yticks([])
        plt.show()

        # Return the field image with the rectangle overlay
        return field_img_with_overlay
    else:
        print("Sample image not found in the field image.")
        return None

if __name__ == "__main__":
    sample_image_path = "spot-dave.jpeg"
    field_image_path = "sample.jpeg"

    result_image = find_template_in_image(sample_image_path, field_image_path)
    if result_image is not None:
        # Save the resulting image in the same folder as the Python code
        output_file_path = os.path.join(os.path.dirname(__file__), "output_image.jpeg")
        cv2.imwrite(output_file_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
