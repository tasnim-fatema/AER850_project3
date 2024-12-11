# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 20:25:21 2024

@author: 16477
"""

def mask_image(input_path, output_path, intermediate_path_prefix, min_contour_area=50, canny_thresholds=(30, 150)):
    """
    Enhanced object masking and extraction with improved edge refinement, contrast enhancement, and text readability.

    Args:
        input_path (str): Path to the input image.
        output_path (str): Path to save the output masked image.
        intermediate_path_prefix (str): Prefix to save intermediate results.
        min_contour_area (int): Minimum contour area to filter small objects.
        canny_thresholds (tuple): Thresholds for Canny edge detection.
    """
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    # Load the image
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Image at {input_path} could not be loaded.")

    # Resize the image for consistent processing
    resized_image = cv2.resize(image, (1024, 1024))

    # Convert to grayscale and enhance edges
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gaussian_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    sharpened = cv2.addWeighted(gray, 1.5, gaussian_blur, -0.5, 0)

    # Adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Edge detection using Canny
    t_lower, t_upper = canny_thresholds
    edges = cv2.Canny(gaussian_blur, t_lower, t_upper)

    # Find contours and filter small objects
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # Create a blank mask and draw the filtered contours
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

    # Refine the mask using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)

    # Save the refined mask for debugging
    cv2.imwrite(f"{intermediate_path_prefix}_refined_mask.jpg", mask)

    # Apply the mask to isolate the object
    masked_image = cv2.bitwise_and(resized_image, resized_image, mask=mask)

    # Enhance contrast and color in LAB color space
    lab = cv2.cvtColor(masked_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced_image = cv2.merge((l, a, b))
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_LAB2BGR)

    # Text enhancement: Sharpen text regions
    text_kernel = np.ones((3, 3), np.uint8)
    text_mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, text_kernel)
    enhanced_text = cv2.addWeighted(enhanced_image, 1.5, cv2.GaussianBlur(enhanced_image, (5, 5), 0), -0.5, 0)
    enhanced_image = np.where(text_mask[..., None] > 0, enhanced_text, enhanced_image)

    # Add a uniform black background
    black_background = np.zeros_like(resized_image)
    final_image = np.where(mask[..., None] > 0, enhanced_image, black_background)

    # Crop and center the object
    x, y, w, h = cv2.boundingRect(np.vstack(filtered_contours))
    padding = 20
    cropped = final_image[
        max(0, y - padding) : min(y + h + padding, final_image.shape[0]),
        max(0, x - padding) : min(x + w + padding, final_image.shape[1]),
    ]
    centered_image = cv2.resize(cropped, (1024, 1024))

    # Save final results
    cv2.imwrite(f"{intermediate_path_prefix}_cropped.jpg", cropped)
    cv2.imwrite(output_path, centered_image)
    print(f"Final image saved to {output_path}")

    # Display results
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(adaptive_thresh, cmap="gray")
    plt.title("Thresholded Image")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(edges, cmap="gray")
    plt.title("Edges Detected")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(mask, cmap="gray")
    plt.title("Refined Mask")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    plt.title("Cropped Object")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(centered_image, cv2.COLOR_BGR2RGB))
    plt.title("Final Centered Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    
# Main function
if __name__ == "__main__":
    # Define paths
    input_image_path = r"C:\Users\16477\Documents\GitHub\AER850_project3\Project 3 Data\motherboard_image.JPEG"
    output_masked_path = r"C:\Users\16477\Documents\GitHub\AER850_project3\Project 3 Data\final_motherboard.JPEG"
    intermediate_path_prefix = r"C:\Users\16477\Documents\GitHub\AER850_project3\Project 3 Data\intermediate"

    # Call the masking function
    mask_image(input_image_path, output_masked_path, intermediate_path_prefix)
