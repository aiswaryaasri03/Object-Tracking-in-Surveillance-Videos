# Media Processing Application

The **Media Processing Application** is an interactive, user-friendly web application developed using Streamlit. It leverages advanced computer vision techniques for real-time image and video analysis. Designed for both educational and practical purposes, the application supports a wide range of media processing capabilities, offering insights into image and video data through visualizations and outputs.

This application combines the power of OpenCV, ImageAI, and other Python libraries to process images and videos with remarkable flexibility and ease. Whether you're looking to:
- Analyze RGB channels in images,
- Detect and track objects in videos,
- Study motion patterns through optical flow,

This application provides a robust platform for experimentation and use.

## Features

### Image Processing
- Extract RGB channels and visualize them individually.
- Apply **Gaussian Blur** for smoothness.
- Generate pixel intensity histograms.

### Video Processing
- Perform object detection using **TinyYOLOv3**.
- Track objects with custom bounding boxes and unique IDs.
- Analyze motion using **Dense and Sparse Optical Flow** techniques.

### Download Options
- Save processed images as a **ZIP** file.
- Download processed videos in **MP4** format.


## Technologies Used
- **Frontend:** Streamlit
- **Backend:** OpenCV, ImageAI, NumPy, Matplotlib
- **Programming Language:** Python
- **File Handling:** BytesIO, tempfile, zipfile


## License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.
