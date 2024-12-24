Media Processing Application
The Media Processing Application is an interactive, user-friendly web application developed using Streamlit that leverages advanced computer vision techniques for real-time image and video analysis. Designed for both educational and practical purposes, the application supports a wide range of media processing capabilities, offering insights into image and video data through visualizations and outputs.
This application combines the power of OpenCV, ImageAI, and other Python libraries to process images and videos with remarkable flexibility and ease. Whether you're looking to analyze RGB channels in images, detect and track objects in videos, or study motion patterns through optical flow, this application provides a robust platform for experimentation and use

Features
  •	Image Processing:
      o	Extract RGB channels and visualize them individually.
      o	Apply Gaussian Blur for smoothness.
      o	Generate pixel intensity histograms.
  •	Video Processing:
      o	Perform Object Detection using TinyYOLOv3.
      o	Track objects with custom bounding boxes and unique IDs.
      o	Analyze motion using Dense and Sparse Optical Flow techniques.
  •	Download Options:
      o	Save processed images as a ZIP file.
      o	Download processed videos in MP4 format.
Technologies Used
  •	Frontend: Streamlit
  •	Backend: OpenCV, ImageAI, NumPy, Matplotlib
  •	Programming Language: Python
  •	File Handling: BytesIO, tempfile, zipfile
License
This project is licensed under the MIT License. See the LICENSE file for more details.
