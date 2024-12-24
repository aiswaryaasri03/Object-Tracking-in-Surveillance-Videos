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


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/media-processing-app.git
   cd media-processing-app
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the TinyYOLOv3 model and place it in the project directory for Object Detection:
   - [TinyYOLOv3 Model](https://sourceforge.net/projects/imageai.mirror/files/3.0.0-pretrained/tiny-yolov3.pt/download)

5. Download the following files and place it in the project directory for Object Tracking:
   - YOLOv4 Model
   - [YOLOv4 Weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights)
   - [Data File](https://github.com/pjreddie/darknet/blob/master/data/coco.names)


## Usage

1. Run the application:
   ```bash
   streamlit run app.py
   ```

2. Open the application in your browser.

3. Upload an image or video and select the desired processing option.

4. Download the processed outputs directly from the interface.


## Technologies Used
- **Frontend:** Streamlit
- **Backend:** OpenCV, ImageAI, NumPy, Matplotlib
- **Programming Language:** Python
- **File Handling:** BytesIO, tempfile, zipfile


## License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.
