# Object-dimensions-detection-ai
A program that inputs images and videos and returns the 3 dimensions of the said object

-----

# Documentation: Real-Time 3D Object Measurement

### **Version 1.0**

**Script:** `realtime_measure_fixed.py`

-----

## 1\. Introduction and Overview

This document provides a detailed explanation of the `realtime_measure_fixed.py` script, a sophisticated computer vision application that performs real-time object detection, monocular depth estimation, and physical size measurement using a standard webcam.

### 1.1. Purpose of the Script

The primary goal of this script is to bridge the gap between a 2D image and the 3D world. By combining two powerful deep learning models—**YOLO** for identifying objects and **MiDaS** for estimating depth—it can overlay information onto a live video feed that includes:

  * **What** objects are in the scene (e.g., "person", "car").
  * **How far** away they are.
  * Their approximate **real-world width and height** in meters.

The application is designed for real-time performance, utilizing threading to ensure that the user interface remains smooth and responsive while heavy computational tasks are processed in the background. It also includes a user-friendly calibration feature, allowing it to adapt to different cameras and environments with a single keypress.

### 1.2. Core Technologies

The script is built upon a foundation of several key technologies:

  * **PyTorch**: An open-source machine learning framework that serves as the backbone for running the deep learning models. It provides the necessary tools for tensor computation and GPU acceleration.
  * **Ultralytics YOLO**: The script uses a YOLO (You Only Look Once) model provided by Ultralytics. YOLO is a state-of-the-art object detection model known for its exceptional speed and accuracy.
  * **MiDaS (intel-isl)**: A deep learning model for monocular depth estimation. It is trained to predict the depth of every pixel in an image from a single 2D input, creating a "depth map."
  * **OpenCV (cv2)**: A fundamental library for computer vision tasks. In this script, it's used for capturing video from the webcam, basic image processing (color conversion, drawing shapes and text), and displaying the final output.
  * **NumPy**: A library for numerical operations in Python, used extensively for handling image data and depth maps as multi-dimensional arrays.
  * **Python Threading**: Used to run the computationally intensive model inference in a separate process from the main camera feed loop, preventing lag and ensuring a smooth real-time experience.

-----

## 2\. Fundamental Concepts Explained

To understand how the script works, it's essential to grasp the core computer vision concepts it employs.

### 2.1. Object Detection with YOLO

**Object detection** is a computer vision task that involves identifying and locating objects within an image or video. It answers two questions: "What objects are here?" and "Where are they?".

**YOLO (You Only Look Once)** is a revolutionary family of object detection models. Unlike older models that would look at an image multiple times, YOLO processes the entire image in a single pass through its neural network. This makes it incredibly fast and suitable for real-time applications. The output of a YOLO model is a list of **bounding boxes**, where each box contains:

  * The coordinates of the box (`x1`, `y1`, `x2`, `y2`).
  * The class of the detected object (e.g., "person").
  * A confidence score indicating how certain the model is about its prediction.

This script uses the `ultralytics` Python library, which provides a simple and powerful interface to run various YOLO models.

### 2.2. Monocular Depth Estimation with MiDaS

**Depth estimation** is the task of creating a map that shows the distance of every point in a scene from the camera. When this is done using only a single camera (one "eye"), it's called **monocular depth estimation**. This is a challenging task because a 2D image inherently lacks depth information.

**MiDaS (Mixed-Data Supervision)** is a state-of-the-art model developed by Intel that excels at this. It's a neural network trained on a massive and diverse dataset of images with corresponding depth information. Given a regular 2D image, MiDaS outputs a **depth map**.

**Crucial Point: Relative vs. Absolute Depth**
MiDaS does **not** output depth in meters or feet. It produces a map of **relative, unscaled depth**. The values in the map correctly represent that one object is twice as far as another, but they don't have a real-world unit. They are arbitrary "depth units." This is the fundamental reason why the script **requires calibration**—to find the conversion factor that translates these arbitrary units into meters.

### 2.3. The Pinhole Camera Model and Size Estimation

To calculate the real-world size of an object, we rely on a simplified model of how a camera works, known as the **pinhole camera model**. This model uses the principles of similar triangles.

Imagine an object of real-world width `$W_{real}$` at a distance `$D$` from the camera. The camera's lens focuses an image of this object onto the camera's sensor. This image has a width in pixels, `$w_{px}$`. The relationship between these values is governed by the camera's **focal length**, `$f$`.

The core formula is:
$$\frac{W_{real}}{D} = \frac{w_{px}}{f}$$

Where `$f$` is the focal length expressed in pixels. If we know the distance to the object (`$D$`), its width in pixels (`$w_{px}$`), and the camera's focal length (`$f$`), we can rearrange the formula to solve for its real-world width:

$$W_{real} = \frac{w_{px} \times D}{f}$$

This is the central calculation the script uses to estimate object size.

  * `$w_{px}$` comes from the YOLO bounding box.
  * `$D$` comes from the MiDaS depth map (after calibration).
  * `$f$` is calculated from the camera's field of view.

### 2.4. Calibration: Tying It All Together

As mentioned, MiDaS gives us a relative depth value, let's call it `$d_{midas}$`. The true metric distance `$D$` is related to this by a scaling factor, `$S$`:

$$D = d_{midas} \times S$$

Our goal during calibration is to find this scale factor `$S$`. We do this by showing the camera an object for which we know the real-world width. Let's say we use a person, and we configure `KNOWN_OBJ_WIDTH_M = 0.45` meters.

When we press the 'c' key:

1.  The script finds the person in the frame (the `KNOWN_OBJ_CLASS`).
2.  YOLO gives us its width in pixels, `$w_{px}$`.
3.  MiDaS gives us its median relative depth, `$d_{midas}$`.
4.  We know the true width, `KNOWN_OBJ_WIDTH_M`.

We can now substitute our equation for `$D$` into the size estimation formula:

$$W_{real} = \frac{w_{px} \times (d_{midas} \times S)}{f}$$

Since we are looking at our known object, `$W_{real}$` is `KNOWN_OBJ_WIDTH_M`. We can now solve for the unknown scale factor `$S$`:

$$S = \frac{KNOWN\_OBJ\_WIDTH\_M \times f}{w_{px} \times d_{midas}}$$

This calculated value `$S$` is stored as `g_scale.depth_to_m`. From this point on, the script can convert any relative depth from MiDaS into an accurate distance in meters, allowing it to measure any detected object.

### 2.5. Real-Time Performance with Threading

Performing object detection and depth estimation on every frame is computationally expensive. If done sequentially in a single loop (capture -\> infer -\> display -\> repeat), the frame rate would be very low, resulting in a choppy, lagging video feed.

This script solves the problem using a **producer-consumer architecture** with two threads:

1.  **Main Thread (Producer)**: Its only jobs are to capture frames from the camera as fast as possible, put them into a `frame_queue`, and display processed results when they are ready. This thread is lightweight and always responsive to user input (like key presses).
2.  **Worker Thread (Consumer)**: This thread runs in the background. It continuously takes frames from the `frame_queue`, performs the heavy YOLO and MiDaS inference, annotates the frame with results, and places the final annotated image into a `result_queue`.

This separation ensures that the slow inference process does not block the main loop, leading to a smooth visual experience for the user. The `queue` data structure is thread-safe, handling the communication between the two threads automatically.

-----

## 3\. Code Deep Dive

This section breaks down the script, explaining each component in detail.

### 3.1. Imports and Libraries

  * `os`, `time`, `math`: Standard Python libraries for operating system interaction, time-related functions (like `time.time()`), and mathematical calculations (e.g., `tan`, `radians`).
  * `queue`: Provides the `Queue` class, essential for thread-safe communication between the main thread and the worker thread.
  * `torch`: The core deep learning framework. Used for loading models, moving data to the GPU (`.to(device)`), and running inference (`torch.no_grad()`).
  * `warnings`: Used to suppress non-critical `FutureWarning` messages from libraries.
  * `threading`: The library used to create and manage the background `processing_worker` thread.
  * `numpy as np`: A fundamental library for scientific computing, used here to represent images and depth maps as numerical arrays for efficient processing.
  * `cv2`: The OpenCV library. Used for camera capture (`VideoCapture`), image manipulation (`cvtColor`, `addWeighted`), and drawing (`rectangle`, `putText`).
  * `dataclasses`: Provides the `@dataclass` decorator for creating simple classes like `MeasureScale` with minimal boilerplate code.
  * `typing`: Provides type hints (e.g., `Optional`, `Tuple`) for improved code readability and maintainability.
  * `ultralytics.YOLO`: The specific class from the `ultralytics` package used to load and run YOLO models.
  * `PIL.Image`: Used as a fallback for MiDaS transforms that expect the Python Imaging Library (PIL) image format instead of a NumPy array.

### 3.2. Configuration Block (`---- CONFIG ----`)

This section contains global constants that allow you to easily tune the script's behavior without modifying the core logic.

  * `YOLO_MODEL`: Specifies which YOLO model file to use. `"yolo11n.pt"` is the "nano" version, which is very fast but less accurate. `"yolo11x.pt"` is the "extra large" version, which is much more accurate but significantly slower.
  * `MIDAS_MODEL_TYPE`: Determines which MiDaS model variant to load. `"DPT_Hybrid"` offers a good balance of speed and accuracy. `"DPT_Large"` is more accurate but slower, while `"MiDaS_small"` is the fastest but least accurate.
  * `CAM_INDEX`: The index of the camera to use. `0` is typically the default built-in webcam. If you have multiple cameras, you might use `1`, `2`, etc.
  * `CAM_WIDTH`, `CAM_HEIGHT`: The desired resolution for the camera feed. Higher resolutions provide more detail for the models but require more processing power.
  * `USE_DIRECTSHOW`: A Windows-specific flag that can help OpenCV connect to certain webcams more reliably.
  * `CAM_HFOV_DEG`: The **Horizontal Field of View** of your camera in degrees. This is a crucial parameter for accurately calculating the focal length. You may need to look up this specification for your webcam model. A typical value is around 70 degrees.
  * `FOCAL_LENGTH_PX`: If you know your camera's focal length in pixels, you can set it here to override the HFOV-based calculation for higher accuracy.
  * `KNOWN_OBJ_CLASS`, `KNOWN_OBJ_WIDTH_M`: These define the object used for calibration. By default, it looks for a `"person"` and assumes their shoulder width is approximately `0.45` meters. You could change this to `"bottle"` and `0.07` meters, for example.
  * `DEPTH_PERCENTILE_LOW`, `DEPTH_PERCENTILE_HIGH`: Used in the `robust_box_depth_stats` function to make depth measurements more stable by ignoring the closest 10% and farthest 10% of pixels within a bounding box, thus removing outliers.
  * `EMA_ALPHA`: The alpha parameter for the Exponential Moving Average filter. A value of `0.25` means that each new measurement contributes 25% to the updated value, while the previous history contributes 75%. This helps to smooth out jittery measurements.
  * `CONF_THRESH`, `IOU_THRESH`: Standard object detection thresholds. Detections with confidence below `0.25` are ignored. `IOU_THRESH` is used by the tracking algorithm to associate detections across frames.
  * `DRAW_DEPTH_OVERLAY`, `SHOW_INFO_OVERLAY`: Toggles for turning the visual depth map and the informational text on or off.

### 3.3. Utility Functions (`---- Utilities ----`)

  * **`compute_focal_from_hfov_px(img_width, hfov_deg)`**: Implements the geometric formula to calculate focal length (`$f$`) from the image width in pixels and the horizontal field of view. The formula is: `$f = (width / 2) / tan(hfov_{rad} / 2)$`.
  * **`clamp_xyxy(x1, y1, x2, y2, w, h)`**: A safety function that ensures the coordinates of a bounding box do not go outside the image boundaries, preventing potential crashes.
  * **`robust_box_depth_stats(depth_map, box)`**: A critical function for accurate measurement. Instead of naively averaging all depth values in an object's bounding box, it performs several steps to get a stable reading:
    1.  It extracts the patch of the `depth_map` corresponding to the bounding `box`.
    2.  It filters out any invalid depth values (`NaN`).
    3.  It calculates the 10th and 90th percentile depth values (`DEPTH_PERCENTILE_LOW/HIGH`).
    4.  It discards all depth values outside this range, effectively removing background or foreground noise that might be included in the box.
    5.  From the remaining clean data, it returns the **median** depth, which is less sensitive to outliers than the mean (average).
  * **`make_depth_colormap(depth)`**: A visualization helper that converts the single-channel, floating-point depth map into a colorized BGR image using OpenCV's `COLORMAP_MAGMA`, making it easy to interpret visually.
  * **`MeasureScale` Dataclass**: A simple container to hold the state of the calibration: the calculated `depth_to_m` scale factor and the time of the last calibration.

### 3.4. Core Classes (`DepthEstimator` and `YoloDetector`)

  * **`class DepthEstimator`**:

      * `__init__`: Initializes the MiDaS model. It uses `torch.hub.load` to automatically download and load the specified `model_type` and its associated `transforms`. It also moves the model to the GPU (`.to(device)`) if one is available.
      * `infer`: This method contains the logic for running depth estimation on a single frame. It's carefully designed to handle different behaviors of the MiDaS transform functions. It tries to apply the transform to a NumPy array first, and if that fails, it falls back to using a PIL Image. It then extracts the resulting tensor, ensures it has the correct dimensions, and passes it to the MiDaS model. The final output is a depth map, resized to match the original frame's dimensions.

  * **`class YoloDetector`**:

      * `__init__`: Initializes the YOLO model from the specified file path and moves it to the correct device.
      * `detect_or_track`: This is the method that runs the object detection. Crucially, it uses `model.track()` instead of `model.predict()`. The tracking feature allows the model to assign a consistent **ID** to each object as it moves across frames. This ID is essential for the smoothing logic, as it allows us to associate a new measurement with the previous measurements for the same object.

### 3.5. The Worker Thread (`---- Worker ----`)

  * **`processing_worker(yolo, midas)`**: This is the engine of the application and runs entirely on a background thread.
    1.  **Main Loop**: It runs in an infinite loop, waiting to pull a `frame` from the `frame_queue`.
    2.  **Inference**: It calls `yolo.detect_or_track()` and `midas.infer()` to get the detection results and the depth map for the current frame.
    3.  **Focal Length Calculation**: On the first run, it calculates the camera's focal length.
    4.  **Result Processing**:
          * It iterates through all detected bounding boxes from YOLO.
          * For each box, it calls `robust_box_depth_stats` to get a stable depth value (`med_depth`).
          * It retrieves the object's tracker ID (`tid`).
          * **Smoothing**: It uses the `tid` to look up the object's previous smoothed values in the `track_smoothers` dictionary. It then uses the `ema_update` function to apply an Exponential Moving Average to the depth, width, and height estimates. This makes the displayed measurements change smoothly over time rather than jumping around.
          * **Size Calculation**: It applies the core measurement formulas: `depth_m = med_depth * g_scale.depth_to_m` and `W_est = (w_px * depth_m) / g_focal_px`.
          * **Annotation**: It draws the bounding boxes and text labels (class name, confidence, ID, and estimated dimensions) onto the frame using OpenCV functions.
    5.  **Overlay Info**: It calculates the current FPS and draws the informational text overlay on the top-left of the screen.
    6.  **Output**: It puts the final annotated frame and other relevant data (like the calibration candidate) into the `result_queue` for the main thread to display.

### 3.6. The Main Execution Block (`---- Main ----`)

  * **`main()`**: This function orchestrates the entire application.
    1.  **Initialization**: It sets up the PyTorch device, loads the YOLO and MiDaS models (instantiating `YoloDetector` and `DepthEstimator`), and initializes the camera using `cv2.VideoCapture`.
    2.  **Start Worker**: It creates and starts the `processing_worker` thread, which immediately begins waiting for frames.
    3.  **Main Display Loop**:
          * It continuously reads a new `frame` from the camera.
          * It places the raw frame into the `frame_queue` for the worker to process.
          * It checks the `result_queue` for a finished, annotated frame from the worker. If one is available, it displays it using `cv2.imshow`.
          * **Key Handling**: It listens for keyboard input:
              * `'q'`: Breaks the loop and quits the application.
              * `'d'`: Toggles the depth map overlay.
              * `'i'`: Toggles the info text overlay.
              * `'c'`: Triggers the calibration logic. It uses the `calib_candidate` passed from the worker, calculates the new scale factor using the formula derived in section 2.4, and updates the global `g_scale`.
    4.  **Cleanup**: After the loop exits, it releases the camera resource (`cap.release()`) and closes all OpenCV windows (`cv2.destroyAllWindows()`).

-----

## 4\. How to Set Up and Run the Script

### 4.1. Prerequisites

  * Python 3.9 or newer.
  * `pip` package manager.
  * (Optional but highly recommended for good performance) An NVIDIA GPU with CUDA installed.

### 4.2. Installation

1.  Save the code as a single file named `realtime_measure_fixed.py`.
2.  Open a terminal or command prompt.
3.  Install the necessary Python libraries by running the following command:
    ```bash
    pip install ultralytics torch opencv-python Pillow
    ```
    *Note: If you have a CUDA-enabled GPU, it is highly recommended to visit the official PyTorch website and install the version of `torch` that corresponds to your CUDA toolkit version for optimal performance.*

### 4.3. Running the Application

1.  Navigate to the directory where you saved the script in your terminal.
2.  Run the script with the command:
    ```bash
    python realtime_measure_fixed.py
    ```
3.  **First-Time Run**: The first time you run the script, it will need to download the YOLO and MiDaS model weights. This requires an internet connection and may take a few minutes depending on your connection speed. Subsequent runs will be much faster as the models will be cached locally.
4.  A window titled "3D Object Measurement (fixed)" will appear, showing your webcam feed.

### 4.4. Usage Instructions

  * **To Calibrate**: Make sure an object of the `KNOWN_OBJ_CLASS` (a "person" by default) is clearly visible in the frame. Press the **`c`** key. You should see a confirmation message in the console that the scale has been set. After calibration, the `D` (depth), `W` (width), and `H` (height) measurements will be shown in meters.
  * **Toggle Depth Overlay**: Press the **`d`** key to turn the colorful depth map visualization on or off.
  * **Toggle Info Text**: Press the **`i`** key to turn the information text (FPS, device, etc.) on or off.
  * **To Quit**: Press the **`q`** key.

-----

## 5\. Conclusion and Potential Improvements

The `realtime_measure_fixed.py` script is a powerful demonstration of how multiple deep learning models can be combined to create a sophisticated real-world measurement tool. By leveraging YOLO for detection, MiDaS for depth, and a classic pinhole camera model, it provides a surprisingly effective solution using only a standard webcam.

### 5.1. Limitations

  * **Accuracy**: The accuracy is highly dependent on the quality of the models, the camera's calibration (HFOV), lighting conditions, and the distance of the object. It should be considered an estimation tool, not a precision instrument.
  * **Monocular Depth**: Depth estimation from a single camera is inherently ambiguous and can sometimes be inaccurate, especially with reflective or textureless surfaces.
  * **Calibration Object**: The calibration is only as good as the assumed width of the `KNOWN_OBJ_WIDTH_M`.

### 5.2. Potential Improvements

  * **Stereo Camera**: Replacing the monocular MiDaS model with a true stereo depth camera (like an Intel RealSense or ZED camera) would provide much more accurate, absolute depth data, removing the need for manual calibration.
  * **Advanced Smoothing**: While EMA is effective, more advanced filters like a Kalman filter could be implemented for even smoother and more predictive tracking of object dimensions.
  * **GUI Interface**: A graphical user interface (GUI) could be built using a library like PyQt or Tkinter to make the configuration and calibration process more user-friendly.
