
# Import required libraries
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
def mask_image(input_path, output_path):
    # """
    # Performs object masking on the input image to extract the PCB from the background.
    
    # Args:
    #     input_path (str): Path to the input image.
    #     output_path (str): Path to save the output masked image.
    # """
    # Step 1: Load the image
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Image at {input_path} could not be loaded.")
    
    # Step 2: Resize the image (optional, for uniform processing)
    resized_image = cv2.resize(image, (800, 800))

    # Step 3: Convert the image to grayscale
    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Step 4: Apply thresholding to segment the image
    _, threshold = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

    # Step 5: Detect edges using Canny edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Step 6: Find contours from the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 7: Filter contours based on area to remove noise
    min_area = 500  # Minimum area threshold
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    # Step 8: Create a blank mask and draw the filtered contours
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

    # Step 9: Apply the mask to the original image
    masked_image = cv2.bitwise_and(resized_image, resized_image, mask=mask)

    # Step 10: Save the masked image
    cv2.imwrite(output_path, masked_image)
    print(f"Masked image saved to {output_path}")

      # Step 10: Save the masked image
    cv2.imwrite(output_path, masked_image)
    print(f"Masked image saved to {output_path}")

    # Optional: Display the intermediate and final results using Matplotlib
    plt.figure(figsize=(10, 10))

    # Display original image
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    # Display thresholded image
    plt.subplot(2, 3, 2)
    plt.imshow(threshold, cmap='gray')
    plt.title("Thresholded Image")
    plt.axis('off')

    # Display edges detected
    plt.subplot(2, 3, 3)
    plt.imshow(edges, cmap='gray')
    plt.title("Edges Detected")
    plt.axis('off')

    # Display filtered contours
    plt.subplot(2, 3, 4)
    plt.imshow(mask, cmap='gray')
    plt.title("Filtered Contours")
    plt.axis('off')

    # Display masked image
    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    plt.title("Masked Image")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# Main function
if __name__ == "__main__":
    # Define paths
    input_image_path = r"C:\Users\16477\Documents\GitHub\Project 3 Data\motherboard_image.JPEG"
    output_masked_path = r"C:\Users\16477\Documents\GitHub\Project 3 Data\masked_motherboard.JPEG"

    # Call the masking function
    mask_image(input_image_path, output_masked_path)
