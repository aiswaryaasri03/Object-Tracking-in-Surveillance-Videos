import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
from io import BytesIO
from imageai.Detection import VideoObjectDetection
import math
import os
import zipfile
import matplotlib.pyplot as plt
from object_detection import ObjectDetection

# Helper Functions
def convert_to_pil(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def process_image(image):
    """Process Image: RGB Channels, Gaussian Blur, and Histogram"""
    r, g, b = cv2.split(image)
    r_img = cv2.merge([r, np.zeros_like(r), np.zeros_like(r)])
    g_img = cv2.merge([np.zeros_like(g), g, np.zeros_like(g)])
    b_img = cv2.merge([np.zeros_like(b), np.zeros_like(b), b])
    side_by_side = np.hstack((r_img, g_img, b_img))
    side_by_side_resized = cv2.resize(side_by_side, (0, 0), fx=0.5, fy=0.5)  # Resize to half the original size
    blurred_img = cv2.GaussianBlur(image, (15, 15), 0)

    # Histogram
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fig, ax = plt.subplots()
    ax.hist(gray.ravel(), bins=256, range=[0, 256])
    ax.set_title("Pixel Intensity Histogram")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
    histogram_buffer = BytesIO()
    plt.savefig(histogram_buffer, format="PNG")
    histogram_buffer.seek(0)
    histogram_image = Image.open(histogram_buffer)

    return {
        "red_channel": convert_to_pil(r_img),
        "green_channel": convert_to_pil(g_img),
        "blue_channel": convert_to_pil(b_img),
        "rgb_channels_side_by_side": convert_to_pil(side_by_side_resized),
        "blurred": convert_to_pil(blurred_img),
        "histogram": histogram_image
    }

def download_images(images, filenames):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for img, filename in zip(images, filenames):
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            zf.writestr(filename, buffered.getvalue())
    zip_buffer.seek(0)
    return zip_buffer

def detect_objects_in_video(input_video_path, output_base_path):
    video_detector = VideoObjectDetection()
    video_detector.setModelTypeAsTinyYOLOv3()
    video_detector.setModelPath("tiny-yolov3.pt")  # Ensure the model file is in the same directory
    video_detector.loadModel()

    output_video_path = video_detector.detectObjectsFromVideo(
        input_file_path=input_video_path,
        output_file_path=output_base_path,
        frames_per_second=10,
        minimum_percentage_probability=30
    )

    return output_video_path

def track_objects_in_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_video = cv2.VideoWriter(
        temp_output.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height)
    )

    od = ObjectDetection()
    tracking_objects = {}
    track_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        center_points_cur_frame = []
        (class_ids, scores, boxes) = od.detect(frame)
        for box in boxes:
            (x, y, w, h) = box
            cx = int((x + x + w) / 2)
            cy = int((y + y + h) / 2)
            center_points_cur_frame.append((cx, cy))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

        for pt in center_points_cur_frame:
            same_object_detected = False
            for object_id, prev_pt in tracking_objects.items():
                distance = math.hypot(prev_pt[0] - pt[0], prev_pt[1] - pt[1])
                if distance < 35:
                    tracking_objects[object_id] = pt
                    same_object_detected = True
                    break

            if not same_object_detected:
                tracking_objects[track_id] = pt
                track_id += 1

        for object_id, pt in tracking_objects.items():
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)
            cv2.putText(frame, str(object_id), (pt[0] - 10, pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        output_video.write(frame)

    cap.release()
    output_video.release()
    return temp_output.name

def process_optical_flow(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define codec and create VideoWriter object
    output_path = "processed_video.mp4"
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Parameters for Farneback optical flow
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    hsv = np.zeros_like(prev_frame)
    hsv[..., 1] = 255

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        next_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Convert flow to HSV representation
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        # Convert HSV to BGR
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Write the frame
        out.write(bgr)

        prev_gray = next_gray

    cap.release()
    out.release()

    return output_path

def process_sparse_optical_flow(video_path, progress_bar=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error opening video file.")
        return None

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Temporary file for output
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Lucas-Kanade optical flow params
    lk_params = dict(winSize=(15, 15), maxLevel=2, 
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    ret, old_frame = cap.read()
    if not ret:
        st.error("Error reading the first frame.")
        return None

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    mask = np.zeros_like(old_frame)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1, st_status, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        good_new = p1[st_status == 1]
        good_old = p0[st_status == 1]

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

        img = cv2.add(frame, mask)
        out.write(img)
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cap.release()
    out.release()
    return output_path

# Streamlit UI
st.title("Media Processing Application")

# Upload Input
uploaded_file = st.file_uploader("Upload an Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

if uploaded_file is not None:
    file_type = uploaded_file.type.split('/')[0]

    if file_type == 'image':
        # Process Image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(convert_to_pil(img), caption="Uploaded Image", use_container_width=True)

        processing_option = st.radio("Select Processing Option:", ["RGB Channels", "Gaussian Blur", "Histogram"])

        if processing_option:
            processed_outputs = process_image(img)

            if processing_option == "RGB Channels":
                st.image(
                    [
                        processed_outputs["red_channel"],
                        processed_outputs["green_channel"],
                        processed_outputs["blue_channel"],
                        processed_outputs["rgb_channels_side_by_side"]
                    ],
                    caption=[
                        "Blue Channel",
                        "Green Channel",
                        "Red Channel",
                        "RGB Channels Side-by-Side"
                    ],
                    use_container_width=True
                )
            elif processing_option == "Gaussian Blur":
                st.image(
                    processed_outputs["blurred"],
                    caption="Blurred Image",
                    use_container_width=True
                )
            elif processing_option == "Histogram":
                st.image(
                    processed_outputs["histogram"],
                    caption="Pixel Intensity Histogram",
                    use_container_width=True
                )

            zip_buffer = download_images(
                [
                    processed_outputs["red_channel"],
                    processed_outputs["green_channel"],
                    processed_outputs["blue_channel"],
                    processed_outputs["rgb_channels_side_by_side"],
                    processed_outputs["blurred"],
                    processed_outputs["histogram"]
                ],
                [
                    "blue_channel.png", "green_channel.png", "red_channel.png",
                    "rgb_channels_side_by_side.png", "blurred.png", "histogram.png"
                ]
            )
            st.download_button(
                "Download All Outputs",
                data=zip_buffer,
                file_name="processed_images.zip",
                mime="application/zip"
            )

    elif file_type == 'video':
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(uploaded_file.read())
            temp_video_path = temp_video.name

        video_option = st.selectbox(
            "Select Video Processing Mode", 
            ["Object Detection", "Object Tracking", "Dense Optical Flow", "Sparse Optical Flow"]
        )

        if st.button("Process the Input"):
            output_video_path = None  # Initialize as None to avoid unhandled errors

            if video_option == "Object Detection":
                output_video_path = detect_objects_in_video(temp_video_path, "output_video")
            elif video_option == "Object Tracking":
                output_video_path = track_objects_in_video(temp_video_path)
            elif video_option == "Dense Optical Flow":
                output_video_path = process_optical_flow(temp_video_path)
            elif video_option == "Sparse Optical Flow":
                output_video_path = process_sparse_optical_flow(temp_video_path)

            if output_video_path:  # Ensure the output path is valid
                with open(output_video_path, "rb") as file:
                    st.download_button(
                        "Download Processed Video", 
                        data=file, 
                        file_name="processed_video.mp4", 
                        mime="video/mp4"
                    )
            else:
                st.error("Video processing failed. Please check your input and try again.")
