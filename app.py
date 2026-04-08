import os
import cv2
import numpy as np
import tempfile
from pathlib import Path

import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="Student / Face Counting System", layout="wide")
st.title("Student / Face Counting System using YOLO + OpenCV + Streamlit")

st.caption(
    "Camera image capture is supported natively. For video, upload a file here; "
    "native Streamlit camera input is for images, not recorded video."
)

# ----------------------------
# Model loading
# ----------------------------
@st.cache_resource
def load_yolo_model(model_path: str = "yolov8n.pt"):
    return YOLO(model_path)


@st.cache_resource
def load_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)


# ----------------------------
# Detection helpers
# ----------------------------
def detect_people(frame, model, conf_threshold: float, iou_threshold: float, line_width: int):
    results = model.predict(
        source=frame,
        conf=conf_threshold,
        iou=iou_threshold,
        classes=[0],
        verbose=False,
    )

    result = results[0]
    annotated = result.plot(line_width=line_width)
    count = len(result.boxes) if result.boxes is not None else 0
    return annotated, count


def detect_faces(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    annotated = frame.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            "Face",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    cv2.putText(
        annotated,
        f"Faces: {len(faces)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    return annotated, len(faces)


def process_frame(frame, mode, yolo_model, face_cascade, conf_threshold, iou_threshold, line_width):
    if mode == "People (YOLO)":
        return detect_people(frame, yolo_model, conf_threshold, iou_threshold, line_width)
    return detect_faces(frame, face_cascade)


# ----------------------------
# Image processing
# ----------------------------
def process_image(image_bytes, mode, yolo_model, face_cascade, conf_threshold, iou_threshold, line_width):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return None, 0

    annotated_bgr, count = process_frame(
        image,
        mode,
        yolo_model,
        face_cascade,
        conf_threshold,
        iou_threshold,
        line_width,
    )
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb, count


# ----------------------------
# Video processing
# ----------------------------
def process_video(video_bytes, original_name, mode, yolo_model, face_cascade, conf_threshold, iou_threshold, line_width):
    suffix = Path(original_name).suffix if original_name else ".mp4"
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_input.write(video_bytes)
    temp_input.close()

    cap = cv2.VideoCapture(temp_input.name)
    if not cap.isOpened():
        st.error("Could not open the uploaded video.")
        return None, None, None, temp_input.name

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_output.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        temp_output.name,
        fourcc,
        fps if fps > 0 else 20,
        (width, height),
    )

    best_frame = None
    best_count = -1

    frame_box = st.empty()
    progress_bar = st.progress(0)
    count_box = st.empty()

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_bgr, count = process_frame(
            frame,
            mode,
            yolo_model,
            face_cascade,
            conf_threshold,
            iou_threshold,
            line_width,
        )
        writer.write(annotated_bgr)

        if count > best_count:
            best_count = count
            best_frame = annotated_bgr.copy()

        frame_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        frame_box.image(frame_rgb, channels="RGB", use_container_width=True)
        count_box.metric("Current Count", count)

        idx += 1
        if total_frames > 0:
            progress_bar.progress(min(idx / total_frames, 1.0))

    cap.release()
    writer.release()

    best_image_path = None
    if best_frame is not None:
        best_image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
        cv2.imwrite(best_image_path, best_frame)

    return temp_output.name, best_image_path, best_count, temp_input.name


# ----------------------------
# Sidebar settings
# ----------------------------
st.sidebar.header("Settings")
mode = st.sidebar.selectbox("Detection mode", ["Faces (OpenCV)", "People (YOLO)"])
confidence = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.4, 0.05)
iou_threshold = st.sidebar.slider("IoU threshold", 0.1, 0.9, 0.5, 0.05)
line_width = st.sidebar.slider("Box width", 1, 10, 2, 1)
image_display_width = st.sidebar.slider("Display width", 300, 1200, 900, 50)

# Video upload limit note
st.sidebar.info(
    "To allow uploads up to 2 GB, set `.streamlit/config.toml` to:\n\n"
    "[server]\nmaxUploadSize = 2048"
)

yolo_model = load_yolo_model("yolov8n.pt")
face_cascade = load_face_detector()

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3 = st.tabs(["Image Upload", "Camera Image", "Video Upload"])

# ----------------------------
# Image upload tab
# ----------------------------
with tab1:
    st.subheader("Upload an Image")
    image_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        key="image_uploader",
    )

    if image_file:
        raw_bytes = image_file.read()
        col1, col2 = st.columns(2)

        with col1:
            st.write("Original Image")
            st.image(raw_bytes, width=image_display_width)

        with col2:
            st.write("Annotated Image")
            annotated_image, count = process_image(
                raw_bytes,
                mode,
                yolo_model,
                face_cascade,
                confidence,
                iou_threshold,
                line_width,
            )

            if annotated_image is not None:
                st.image(annotated_image, channels="RGB", width=image_display_width)
                st.success(f"Count detected: {count}")

                temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
                cv2.imwrite(temp_img, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                with open(temp_img, "rb") as f:
                    st.download_button(
                        label="Download annotated image",
                        data=f,
                        file_name="annotated_image.jpg",
                        mime="image/jpeg",
                    )
            else:
                st.error("Could not read the uploaded image.")


# ----------------------------
# Camera image tab
# ----------------------------
with tab2:
    st.subheader("Capture an Image from Camera")
    st.caption("This works well on mobile phones and laptops with camera permission enabled.")
    camera_image = st.camera_input("Take a picture")

    if camera_image is not None:
        raw_bytes = camera_image.getvalue()
        col1, col2 = st.columns(2)

        with col1:
            st.write("Captured Image")
            st.image(raw_bytes, width=image_display_width)

        with col2:
            st.write("Annotated Image")
            annotated_image, count = process_image(
                raw_bytes,
                mode,
                yolo_model,
                face_cascade,
                confidence,
                iou_threshold,
                line_width,
            )

            if annotated_image is not None:
                st.image(annotated_image, channels="RGB", width=image_display_width)
                st.success(f"Count detected: {count}")

                temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
                cv2.imwrite(temp_img, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                with open(temp_img, "rb") as f:
                    st.download_button(
                        label="Download captured annotated image",
                        data=f,
                        file_name="camera_annotated_image.jpg",
                        mime="image/jpeg",
                    )
            else:
                st.error("Could not process the captured image.")


# ----------------------------
# Video upload tab
# ----------------------------
with tab3:
    st.subheader("Upload a Video")
    st.caption(
        "Streamlit file uploads are limited by `server.maxUploadSize`. "
        "Set it to 2048 for a 2 GB limit."
    )

    video_file = st.file_uploader(
        "Choose a video",
        type=["mp4", "mov", "avi", "mkv"],
        key="video_uploader",
    )

    if video_file:
        video_bytes = video_file.getvalue()
        st.video(video_bytes)

        st.write("Processing video...")
        output_video_path, best_image_path, best_count, temp_input_path = process_video(
            video_bytes,
            video_file.name,
            mode,
            yolo_model,
            face_cascade,
            confidence,
            iou_threshold,
            line_width,
        )

        if output_video_path and os.path.exists(output_video_path):
            st.success("Video processed successfully.")

            with open(output_video_path, "rb") as f:
                st.download_button(
                    label="Download annotated video",
                    data=f,
                    file_name="annotated_video.mp4",
                    mime="video/mp4",
                )

            st.video(output_video_path)

        if best_image_path and os.path.exists(best_image_path):
            st.subheader("Frame With Highest Count")
            st.image(best_image_path, use_container_width=True)

            with open(best_image_path, "rb") as f:
                st.download_button(
                    label="Download counted face image",
                    data=f,
                    file_name="counted_faces.jpg",
                    mime="image/jpeg",
                )

            st.info(f"Highest count found in one frame: {best_count}")

        if temp_input_path and os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
            except Exception:
                pass
    else:
        st.info("Upload a video to detect and count faces or people.")


st.markdown(
    """
    ---
    **Note:** For a true in-app video recorder from the system camera, Streamlit's built-in `st.camera_input`
    handles images, not recorded video. The app above gives you camera image capture on mobile/desktop and
    large video upload support.
    """
)
