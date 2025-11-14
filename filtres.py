import numpy as np 
import cv2
from matplotlib import pyplot as plt

def apply_gaussian_filters(image_path):
    # Load image
    img = cv2.imread(image_path)
    
    # Create subplots for different kernel sizes
    plt.figure(figsize=(15, 10))
    kernel_sizes = [3, 5, 9, 15]
    
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')
    
    # Apply Gaussian blur with different kernel sizes
    for i, ksize in enumerate(kernel_sizes, 2):
        blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)
        plt.subplot(2, 3, i)
        plt.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
        plt.title(f'Kernel {ksize}x{ksize}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def resize_and_gradients(image_path, new_width=300):
    # Load image
    img = cv2.imread(image_path)
    
    # Resize with aspect ratio preserved
    aspect_ratio = img.shape[1] / img.shape[0]
    new_height = int(new_width / aspect_ratio)
    resized = cv2.resize(img, (new_width, new_height))
    
    # Convert to grayscale for gradient calculation
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Compute X and Y gradients (Sobel operator)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    plt.title('Resized Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(np.abs(grad_x), cmap='gray')
    plt.title('Gradient X')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(np.abs(grad_y), cmap='gray')
    plt.title('Gradient Y')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def segmentation_and_filters(image_path):
    # Load image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply binary thresholding
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(blurred, cmap='gray')
    plt.title('Blurred')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(thresh, cmap='gray')
    plt.title('Binary Thresholding')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

image_path = 'mae2.jpg'  

print("1. Applying Gaussian filters with different kernel sizes:")
apply_gaussian_filters(image_path)

print("\n2. Resizing and computing gradients:")
resize_and_gradients(image_path)

print("\n3. Segmentation and filtering:")
segmentation_and_filters(image_path)


# ---- VIDEO PROCESSING ----

def process_video_frame(video_path):
    # Read the first frame of the video
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Unable to read video")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Compute gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(221)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title('Original Frame')
    plt.axis('off')
    
    plt.subplot(222)
    plt.imshow(np.abs(grad_x), cmap='gray')
    plt.title('Gradient X')
    plt.axis('off')
    
    plt.subplot(223)
    plt.imshow(np.abs(grad_y), cmap='gray')
    plt.title('Gradient Y')
    plt.axis('off')
    
    plt.subplot(224)
    plt.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    plt.title('Gaussian Blur')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def compare_filters(image_path):
    img = cv2.imread(image_path)
    
    # Apply different filters
    gaussian = cv2.GaussianBlur(img, (5, 5), 0)
    median = cv2.medianBlur(img, 5)
    bilateral = cv2.bilateralFilter(img, 9, 75, 75)
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(221)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(222)
    plt.imshow(cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB))
    plt.title('Gaussian Filter')
    plt.axis('off')
    
    plt.subplot(223)
    plt.imshow(cv2.cvtColor(median, cv2.COLOR_BGR2RGB))
    plt.title('Median Filter')
    plt.axis('off')
    
    plt.subplot(224)
    plt.imshow(cv2.cvtColor(bilateral, cv2.COLOR_BGR2RGB))
    plt.title('Bilateral Filter')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def detect_contours(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 100, 200)
    
    # Contour detection
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours
    result = img.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(edges, cmap='gray')
    plt.title('Edges')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Contours')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def process_webcam():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open webcam")
        return
        
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Gaussian blur + edges
            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            edges = cv2.Canny(blurred, 100, 200)
            
            # Display
            cv2.imshow('Original', frame)
            cv2.imshow('Edges', edges)
            
            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()


# Example usage
image_path = 'mae2.jpg'
video_path = 'Download.mp4'

print("4. Processing a video frame:")
process_video_frame(video_path)

print("\n5. Comparing filters:")
compare_filters(image_path)

print("\n6. Contour detection:")
detect_contours(image_path)
