import cv2
import matplotlib.pyplot as plt
import numpy as np

def mask_image(input_image_path, output_image_path):
    # Step 1: Load the image from the specified path
    # Read the image using OpenCV
    original_image = cv2.imread(input_image_path)
    if original_image is None:
        raise ValueError(f"Image at {input_image_path} could not be loaded.")
    
    # Convert the image from BGR (OpenCV default) to RGB for processing
    rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Step 2: Preprocess the image for thresholding
    # Convert the RGB image to grayscale for thresholding
    grayscale_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to smooth the image and reduce noise
    blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
    
    # Enhance edges by sharpening using weighted addition
    sharpened_image = cv2.addWeighted(grayscale_image, 1.5, blurred_image, -0.5, 0)
    
    # Apply a binary inverse threshold to segment the image
    _, threshold_image = cv2.threshold(sharpened_image, 120, 255, cv2.THRESH_BINARY_INV)

    # Step 3: Perform edge detection to identify contours
    # Find external contours in the thresholded image
    contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Select the largest contour based on area (assumes PCB is the largest object)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create a copy of the original image for visualization
    contour_visualization = rgb_image.copy()
    
    # Draw the largest contour on the image for visualization
    cv2.drawContours(contour_visualization, [largest_contour], -1, (0, 255, 0), 3)

    # Step 4: Create a mask for the detected object (PCB)
    # Create a blank mask with the same dimensions as the grayscale image
    binary_mask = np.zeros_like(grayscale_image)
    
    # Fill the detected contour on the mask
    cv2.drawContours(binary_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Step 5: Extract the PCB using the mask
    # Use the mask to extract the PCB region from the original image
    pcb_extracted_image = cv2.bitwise_and(rgb_image, rgb_image, mask=binary_mask)
    
    # Convert the extracted PCB back to BGR for further processing
    pcb_bgr_image = cv2.cvtColor(pcb_extracted_image, cv2.COLOR_RGB2BGR)
    
    # Create a black background and overlay the PCB region on it
    pcb_on_black_background = np.zeros_like(pcb_bgr_image)
    pcb_on_black_background[binary_mask == 255] = pcb_bgr_image[binary_mask == 255]

    # Step 6: Enhance the PCB image using LAB color space
    # Convert the image to LAB color space for better contrast enhancement
    lab_image = cv2.cvtColor(pcb_on_black_background, cv2.COLOR_BGR2LAB)
    
    # Split the LAB image into its channels
    luminance_channel, a_channel, b_channel = cv2.split(lab_image)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_luminance = clahe.apply(luminance_channel)
    
    # Merge the enhanced L-channel back with the original A and B channels
    enhanced_lab_image = cv2.merge((enhanced_luminance, a_channel, b_channel))
    
    # Convert the enhanced LAB image back to BGR color space
    final_enhanced_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2BGR)

    # Step 7: Display results using Matplotlib
    plt.figure(figsize=(15, 10))

    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    # Thresholded image
    plt.subplot(2, 3, 2)
    plt.imshow(threshold_image, cmap="gray")
    plt.title("Thresholded Image")
    plt.axis("off")

    # Contoured edges
    plt.subplot(2, 3, 3)
    plt.imshow(cv2.cvtColor(contour_visualization, cv2.COLOR_BGR2RGB))
    plt.title("Contoured Edges")
    plt.axis("off")

    # Mask created from contours
    plt.subplot(2, 3, 4)
    plt.imshow(binary_mask, cmap="gray")
    plt.title("Mask from Contoured Edges")
    plt.axis("off")

    # Extracted PCB region
    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(pcb_on_black_background, cv2.COLOR_BGR2RGB))
    plt.title("Extracted PCB")
    plt.axis("off")

    # Enhanced PCB
    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(final_enhanced_image, cv2.COLOR_BGR2RGB))
    plt.title("Enhanced PCB")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Step 8: Save the final enhanced image
    try:
        cv2.imwrite(output_image_path, final_enhanced_image)
        print(f"Final image saved to {output_image_path}")
    except Exception as e:
        print(f"Error saving the image: {e}")

# Main function
if __name__ == "__main__":
    # Define the input image path and output image path
    input_image_path = r"C:\Users\16477\Documents\GitHub\AER850_project3\Project 3 Data\motherboard_image.JPEG"
    output_image_path = r"C:\Users\16477\Documents\GitHub\AER850_project3\Project 3 Data\hi.JPEG"

    # Call the masking function to process the image
    mask_image(input_image_path, output_image_path)
