# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:06:55 2024

@author: 16477
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def mask_image(input_path, output_path, intermediate_path_prefix, min_contour_area=50, canny_thresholds=(30, 150)):
    """
    Performs object masking, background removal, edge refinement, and enhancement on the input image.

    Args:
        input_path (str): Path to the input image.
        output_path (str): Path to save the output masked image.
        intermediate_path_prefix (str): Prefix to save intermediate results.
        min_contour_area (int): Minimum contour area to filter small objects.
        canny_thresholds (tuple): Thresholds for Canny edge detection.
    """
    # Step 1: Load the image
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Image at {input_path} could not be loaded.")

    # Step 2: Resize the image (standardize dimensions for consistent processing)
    resized_image = cv2.resize(image, (1024, 1024))

    # Step 3: Convert the image to grayscale
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Step 4: Blur the grayscale image to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Step 5: Apply adaptive thresholding to segment the object from the background
    threshold = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Step 6: Detect edges using Canny edge detection
    t_lower, t_upper = canny_thresholds
    edges = cv2.Canny(blurred, t_lower, t_upper, L2gradient=True)

    # Step 7: Find and filter contours to isolate the object of interest
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # Step 8: Create a blank mask and draw the filtered contours
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

    # Step 9: Refine the mask for smoother edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Close gaps
    mask = cv2.GaussianBlur(mask, (7, 7), 0)  # Feather edges

    # Save the refined mask for debugging
    cv2.imwrite(f"{intermediate_path_prefix}_refined_mask.jpg", mask)

    # Step 10: Apply the mask to isolate the object
    masked_image = cv2.bitwise_and(resized_image, resized_image, mask=mask)

    # Step 11: Enhance color and contrast
    lab = cv2.cvtColor(masked_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced_image = cv2.merge((l, a, b))
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_LAB2BGR)

    # Step 12: Add a uniform black background
    black_background = np.zeros_like(resized_image)
    final_image = np.where(mask[..., None] > 0, enhanced_image, black_background)

    # Step 13: Center the object by cropping and resizing
    x, y, w, h = cv2.boundingRect(np.vstack(filtered_contours))
    cropped = final_image[y : y + h, x : x + w]
    centered_image = cv2.resize(cropped, (1024, 1024))

    # Save intermediate and final results
    cv2.imwrite(f"{intermediate_path_prefix}_cropped.jpg", cropped)
    cv2.imwrite(output_path, centered_image)
    print(f"Final image saved to {output_path}")

    # Optional: Display intermediate and final results
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(threshold, cmap="gray")
    plt.title("Thresholded Image")
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(edges, cmap="gray")
    plt.title("Edges Detected")
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(mask, cmap="gray")
    plt.title("Refined Mask")
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    plt.title("Cropped Object")
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(centered_image, cv2.COLOR_BGR2RGB))
    plt.title("Final Centered Image")
    plt.axis('off')

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

