## **Final Project Overview**

This project is an **AI-based Hand Gesture Recognition** system enhanced with **Trustworthiness Evaluation** mechanisms. It builds upon our midterm work, adding structured evaluations for **reliability**, **robustness**, and **transparency** using a combination of real-time perturbation tests and visual model explainability.

It combines:

* **Mediapipe** for accurate hand tracking and landmark detection  
* **MobileNet** for efficient deep learning-based gesture classification  
* **Custom-built perturbation modules** to simulate realistic visual distortions  
* **Grad-CAM visualizations** for interpreting model decisions and building user trust

The primary goal of this extension is to assess how trustworthy the AI system remains under diverse and imperfect conditions and whether users and developers can understand and rely on its decisions.

**Installation Instructions**

### **1\. Clone the Repository**

| git clone https://github.com/geetha-manogna/AI-Powered-Hand-Gesture-Recognition-for\-Real-Time-Interaction.git
 :---- |
cd AI-Powered-Hand-Gesture-Recognition-for\-Real-Time-Interaction](https://github.com/geetha-manogna/gesture-recognition-trustworthiness-evaluation) |
| :---- |

### **2\. Install Dependencies**

| pip install \-r requirements.txt |
| :---- |

Or manually:

| pip install opencv-python numpy tensorflow tf-keras-vis mediapipe matplotlib |
| :---- |

### **3\. Ensure Model Availability**

Make sure sign\_language\_mobilenet.h5 (trained on ASL dataset) is in the project root directory.

**How to Run the Trustworthy App**

| python app.py |
| :---- |

### **Controls:**

* 1 \= Normal mode (no perturbation)  
* 2 \= Dim Lighting simulation  
* 3 \= Partial Occlusion simulation  
* 4 \= Rotation simulation  
* g \= Toggle GradCAM visualization  
* q \= Quit the application

  **How It Works**

### **Hand Tracking:**

* Mediapipe detects 21 hand landmarks in real-time.  
* These landmarks are used to define a bounding box around the hand region.

  ### **Gesture Classification:**

* The extracted hand region is resized to 224x224 pixels.  
* The MobileNet model classifies the hand into one of 5 gesture classes (A–E).  
* The prediction is displayed on screen along with a confidence score.

  ### **Trust Extensions:**

* Users can apply **perturbations** (dim lighting, occlusion, rotation) to test robustness.  
* **GradCAM overlays** show which parts of the hand the model is focusing on during prediction.  
* **Low-confidence predictions (\<60%)** are marked as “Uncertain.”  
* All results are saved in robustness\_logs.csv for evaluation.  
  **Trustworthiness Principles Addressed**

  ### **Reliability & Robustness**

The system is tested under a range of real-world distortions to assess its ability to consistently recognize gestures:

* **Dim Lighting**: Reduced brightness simulates poor lighting environments  
* **Partial Occlusion**: Black rectangles mask parts of the hand  
* **Gesture Rotation**: Rotated frames simulate tilted hand orientations

Each mode helps verify how prediction accuracy changes. A confidence threshold is applied to identify low-confidence predictions, and all results are stored in robustness\_logs.csv for analysis.

### **Transparency & Explainability**

To address the "black-box" nature of deep learning, this project integrates **Grad-CAM** to:

* Visualize which areas of the hand image influenced the model’s decision  
* Help identify incorrect predictions and potential bias or model gaps

Users can toggle live Grad-CAM heatmaps using the g key to better understand predictions in real time.

## 📁 Folder Structure

```
AI-Powered-Hand-Gesture-Recognition/
├── app.py                     # Final version with trustworthiness features
├── sign_language_mobilenet.h5 # Trained MobileNet model
├── robustness_logs.csv        # Logs (auto-generated)
├── gradcam_*.jpg              # GradCAM visualizations (auto-generated)
├── README.md                  # Project documentation
├── requirements.txt           # Dependency list
```

**Trust Evaluation Outputs**

### **Prediction Logs:**

* All predictions are logged to robustness\_logs.csv  
* Format: mode,predicted\_class,confidence  
* Enables post-hoc analysis and plotting of performance under perturbations

  ### **GradCAM Visualizations:**

* Automatically saved as images: gradcam\_\<mode\>\_\<label\>.jpg  
* Provides visual interpretation of model’s attention area for each prediction  
  **Expected Output**

Once the application is running:

* Detected hands are shown with bounding boxes  
* Predicted gesture is displayed with a confidence score  
* Real-time updates reflect current frame and user inputs  
* GradCAM overlays appear when enabled, showing heatmaps on the hand region  
* Perturbation effects can be toggled live to observe their impact

These outputs help assess both **model performance** and **decision clarity** under varied visual conditions.

**Model Details**

### **Dataset Used:**

* **Source**: ASL Alphabet Dataset (Kaggle)  
* **Description**: Includes labeled images for each alphabet gesture.  
* **Subset Used**: The model is trained on a selected subset: A, B, C, D, and E.

  ### **Training Overview:**

* **Architecture**: MobileNet (pretrained on ImageNet)  
* **Transfer Learning**: Fine-tuned using hand gesture images from the ASL dataset.  
* **Input Size**: 224x224 RGB images  
* **Output**: 5-class softmax layer for the ASL labels A–E  
* **Loss Function**: Categorical Crossentropy  
* **Optimizer**: Adam  
* **Model Format**: Saved as sign\_language\_mobilenet.h5

  ### **Preprocessing:**

* Images are resized to (224, 224\)  
* Pixel values are normalized (divided by 255.0)

  ### **Runtime Model Usage:**

| import tensorflow as tf|
| :---- |
model \= tf.keras.models.load\_model("sign\_language\_mobilenet.h5") |
| :---- |

## **Citations & Acknowledgments**

* Mediapipe Hands API: [https://developers.google.com/mediapipe](https://developers.google.com/mediapipe)  
* TensorFlow MobileNet: [https://www.tensorflow.org/](https://www.tensorflow.org/)  
* GradCAM Visualization: [https://github.com/keisen/tf-keras-vis](https://github.com/keisen/tf-keras-vis)  
* OpenCV Library: [https://opencv.org/](https://opencv.org/)  
* ASL Dataset: [https://www.kaggle.com/datasets/grassknoted/asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)  
  
