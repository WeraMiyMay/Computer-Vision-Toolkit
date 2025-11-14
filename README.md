<div align="center">

# 🖼️ Image Processing & Computer Vision Toolkit  
### *Gaussian Blur • Gradients • Segmentation • Contours • Video Processing • Webcam Mode*

</div>

---

## 📌 Overview

This repository contains a collection of **image and video processing techniques** implemented with **OpenCV**, **NumPy**, and **Matplotlib**.  
It demonstrates fundamental computer vision operations such as:

- Gaussian filtering  
- Gradient detection (Sobel)  
- Image resizing  
- Thresholding and segmentation  
- Edge detection  
- Contour extraction  
- Video frame analysis  
- Real-time webcam processing  

The project is educational and showcases how different filters, transformations,  
and operations affect images and video data.

---

## 🎯 Project Purpose

This project was created to demonstrate how classical computer vision techniques work:

### ✔ Gaussian Filtering  
Smooths the image to remove noise and highlight large structures.

### ✔ Sobel Gradients  
Show how intensity changes across the image, useful for edge detection.

### ✔ Segmentation & Thresholding  
Allows separating foreground from background.

### ✔ Edge Detection  
Helps extract shape boundaries with Canny detection.

### ✔ Contours  
Allows identifying object outlines.

### ✔ Video Frame Processing  
Applies all methods to real video frames.

### ✔ Webcam Mode  
Demonstrates real-time processing with live video stream.

These techniques form the **foundation of many modern AI and computer vision systems**,  
so this project is a great practical introduction.

---

## 🛠️ Technologies Used

- **Python 3**
- **OpenCV**
- **NumPy**
- **Matplotlib**

---

## 🧪 Features Included

### 🔹 1. Gaussian Filters  
Compares blurring with kernel sizes: **3×3, 5×5, 9×9, 15×15**

### 🔹 2. Image Resizing + Gradients  
Resizes the image and computes **Sobel gradients** on X and Y axes.

### 🔹 3. Segmentation  
Converts image → grayscale → blur → thresholding.

### 🔹 4. Video Frame Analysis  
Loads the **first frame of a video** and extracts gradients + blur.

### 🔹 5. Filter Comparison  
Shows difference between:
- Gaussian blur  
- Median filter  
- Bilateral filter  

### 🔹 6. Contour Detection  
Extracts object edges and draws contours.

### 🔹 7. Webcam Processing  
Applies blur + edge detection in **real time**.

---

## ▶️ How to Run

Install required packages:

```bash
pip install opencv-python numpy matplotlib

image_path = 'your_image.jpg'
video_path = 'your_video.mp4'
