# Indian Sign Language (ISL) Alphabet Recognition

An image-classification project that recognizes the 26 Indian Sign Language alphabet signs (A to Z) from a single hand image, compares a custom CNN with a MobileNetV2 transfer-learning model, audits the dataset for data leakage, and verifies the result in real time with a webcam.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Project Architecture](#3-project-architecture)
4. [Model Architectures](#4-model-architectures)
5. [Training Strategy](#5-training-strategy)
6. [Evaluation](#6-evaluation)
7. [Data-Leakage Audit](#7-data-leakage-audit)
8. [Real-Time Webcam Verification](#8-real-time-webcam-verification)
9. [Why the Webcam Result Differs from the Test Result](#9-why-the-webcam-result-differs-from-the-test-result)
10. [Improved Real-Time Pipeline](#10-improved-real-time-pipeline)
11. [Project Structure](#11-project-structure)
12. [Installation and Usage](#12-installation-and-usage)
13. [Limitations](#13-limitations)
14. [Future Work](#14-future-work)
15. [Conclusion](#15-conclusion)

---

## 1. Project Overview

**Goal:** classify a hand image into one of the 26 ISL alphabet letters and test the trained model live with a webcam.

**Main steps**
- Prepare, validate and split the dataset
- Explore the data and build an input pipeline with augmentation
- Train two models: a custom CNN and a MobileNetV2 transfer-learning model
- Evaluate with accuracy, precision, recall, F1-score and confusion matrices
- Audit the data for duplicates and rebuild a leak-free split
- Verify the final model with a webcam using MediaPipe hand detection
- Diagnose webcam failures and provide an improved real-time pipeline

**Tech stack:** Python, TensorFlow/Keras, NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, OpenCV, MediaPipe, split-folders.

---

## 2. Dataset

- **Source:** PRAICP-1000-IndiSignLang (one folder per letter).
- **Classes:** 26 (A to Z).
- **Added data:** the letters **J** and **Z** were recorded manually with a webcam and added to the dataset.
- **Image content:** a single hand, close to the camera, mostly on a plain light wall. Skin tone, hand size, rotation and image orientation vary.
- **Image sizes:** mixed resolutions and aspect ratios; every image is resized to 224 x 224 RGB before it enters a model.
- **Class balance:** mildly imbalanced. The smallest classes are H, V, W, U and D.
- **Important property:** J and Z are full webcam frames (dark room, colour cast), while the other letters are bright close-ups. This difference causes the webcam problem described in section 9.

**Splits**
| Split | Purpose |
|---|---|
| Train | learn the weights |
| Validation | tune training and select the best epoch |
| Test | final, untouched evaluation |

Two splits were used: a first stratified file-level split, and a leak-free split built on unique images after the duplicate audit (section 7).

---

## 3. Project Architecture

```
            PRAICP-1000-IndiSignLang.zip  +  webcam images (J, Z)
                               |
                               v
                  Extract and validate 26 classes
                               |
                               v
               Stratified split (train / val / test)
                               |
                               v
                   Exploratory data analysis
                               |
                               v
        tf.data pipeline: resize 224 x 224, batch, prefetch
                               |
              +----------------+----------------+
              v                                 v
        Model 1: Custom CNN          Model 2: MobileNetV2
      (trained from scratch)        (transfer learning, 2 stages)
              |                                 |
              +----------------+----------------+
                               v
        Evaluation: accuracy, precision, recall, F1,
                   confusion matrix, error analysis
                               |
                               v
        Data-leakage audit (hashing) -> leak-free re-split
                               |
                               v
          Final MobileNetV2 trained on the leak-free split
                               |
                               v
   Webcam verification: MediaPipe hand detection -> crop -> model
                               |
                               v
        Diagnosis of the domain gap -> improved real-time pipeline
```

---

## 4. Model Architectures

### 4.1 Model 1: Custom CNN
```
Input (224 x 224 x 3)
  -> Data augmentation (training only)
  -> Rescaling (1/255)
  -> Conv2D (32) -> BatchNorm -> ReLU -> MaxPool
  -> Conv2D (64) -> BatchNorm -> ReLU -> MaxPool
  -> Conv2D (128) -> BatchNorm -> ReLU -> MaxPool
  -> GlobalAveragePooling2D
  -> Dense (128, ReLU) -> Dropout
  -> Dense (26, softmax)
```
A small network built from scratch. It serves as the baseline.

### 4.2 Model 2: MobileNetV2 (transfer learning)
```
Input (224 x 224 x 3)
  -> Data augmentation (training only)
  -> Resize to the MobileNetV2 input size
  -> Rescaling to [-1, 1]
  -> MobileNetV2 base (ImageNet weights, alpha = 0.75)
  -> GlobalAveragePooling2D
  -> Dropout
  -> Dense (26, softmax)
```
The pre-trained base already knows general visual features, so only a small head needs to be learned first.

### 4.3 Data augmentation (training only)
Random rotation, zoom, translation and contrast. Brightness, colour and background changes are **not** included in the original version; this is a cause of the webcam gap (section 9).

---

## 5. Training Strategy

**Custom CNN**
- Optimizer: Adam
- Loss: sparse categorical cross-entropy
- Callbacks: ModelCheckpoint (best validation loss), EarlyStopping, ReduceLROnPlateau
- Trained on the CPU (TensorFlow does not support GPU on native Windows for recent versions), so training is slow.

**MobileNetV2 (two stages)**
1. **Feature extraction:** the base is frozen and only the new classifier head is trained.
2. **Fine-tuning:** the last 30 layers of the base are unfrozen (BatchNormalization layers stay frozen) and trained with a very low learning rate (1e-5).

The same two-stage recipe was repeated on the leak-free split to produce the final model.

---

## 6. Evaluation

Metrics used on the test set:
- Accuracy and loss
- Precision, recall and F1-score per class (classification report)
- Confusion matrix (rows = actual, columns = predicted)
- Visual inspection of misclassified images

**Observations**
- Transfer learning clearly outperforms the custom CNN and trains much faster on the same CPU.
- The custom CNN's validation curve is unstable in the early epochs (BatchNorm statistics and a high learning rate) and stabilizes as the learning rate decreases.
- The remaining CNN errors involve visually similar signs (for example L with H and V, N with S and T).
- The final MobileNetV2 model classifies the test set with no visible errors, which is why the data audit and the webcam test are important (sections 7 to 9).

---

## 7. Data-Leakage Audit

A near-perfect validation score on 26 look-alike signs is suspicious, so every image was hashed (SHA-256) to find exact duplicates.

**Findings**
- Identical images existed in more than one split (train/test, train/validation and validation/test), and the training set contained copies of its own images.
- The duplicates were concentrated in **J** and **Z**, because the webcam recording saved many consecutive, almost identical frames.
- Duplicates had consistent labels, but they inflate the evaluation.

**Fix**
- Keep one copy of every unique image.
- Re-split the unique images (stratified, seed 42).
- Verify that the overlap between train, validation and test is zero.

Note: exact-hash checks do not detect near-identical frames from the same recording session, so scores can still be optimistic.

---

## 8. Real-Time Webcam Verification

**Pipeline**
```
Webcam frame
   -> MediaPipe Hand Landmarker (21 landmarks)
   -> square hand crop with padding
   -> resize to 224 x 224, BGR -> RGB, raw 0-255 pixels
   -> classifier
   -> top-3 letters shown on screen
```
- The model contains its own resizing and rescaling, so it must receive raw 0 to 255 RGB pixels.
- The hand detector does not find a hand in every frame, so predictions can flicker or disappear.

**Result:** the model recognizes only some letters correctly in real time. A sample photo of the letter B was repeatedly predicted as **J**, with almost all probability on J and Z.

---

## 9. Why the Webcam Result Differs from the Test Result

| # | Likely cause | Evidence |
|---:|---|---|
| 1 | **Webcam shortcut for J and Z.** J and Z are the only classes recorded as full webcam frames; all other letters are bright close-ups. The model learns "dark room, colour cast = J or Z". | The B photo was predicted as J/Z; nearly all probability mass went to those two classes. |
| 2 | **Domain shift.** Training images show a close-up hand on a plain wall; webcam frames show a room, a forearm and a different scale. | Visual comparison of dataset images and webcam frames. |
| 3 | **Colour and lighting shift.** Augmentation had no brightness, colour or white-balance changes. | Webcam crops are darker with a colour tint. |
| 4 | **Optimistic test set.** The test images come from the same recording sessions as the training images. | Duplicate and near-duplicate frames. |
| 5 | **Aspect-ratio distortion.** Non-square images are squeezed to a square. | Mixed image sizes in the dataset. |
| 6 | **Look-alike signs.** Fist-like letters and V/W differ by one finger. | Error analysis. |
| 7 | **Weak real-time loop.** No smoothing, no confidence threshold, no "unknown" state; per-frame `predict()` is slow. | Flickering predictions. |
| 8 | **Static classifier for motion letters.** If J and Z involve movement, one frame cannot represent them. | Sign definition. |

**Another pitfall:** the helper `predict_isl_alphabet` has `model=mobilenet_finetuned_model` as its default argument, which is the **first-split model**. Always pass the model explicitly, for example `predict_isl_alphabet(path, model=final_model)`.

---

## 10. Improved Real-Time Pipeline

Recommended order:

1. **Hand-crop dataset:** run MediaPipe on every dataset image and save the same padded square crop that the live code uses. Skip exact duplicates.
2. **Webcam data collection for all 26 letters:** several sessions with different lighting, background, distance and hand tilt; save only frames that differ from the previous saved frame. Keep one session only for testing.
3. **Retraining:** train on hand crops plus webcam sessions, with strong colour and geometry augmentation (brightness, contrast, saturation, hue, per-channel white balance, zoom, translation). Validate on the held-out webcam session.
4. **Landmark classifier (optional, most robust):** convert the 21 landmarks to features (wrist at the origin, scaled by hand size) and train a Random Forest or similar model. It ignores background, skin tone and lighting.
5. **Real-time loop:**
   - same crop and padding as in training
   - exponential smoothing of probabilities
   - accept a letter only if the confidence and the margin to the second letter are high
   - require the same letter to stay on top for several consecutive frames
   - call `model(x, training=False)` instead of `model.predict()`
   - run as a script, not inside Jupyter

The standalone script `isl_realtime.py` implements this loop.

---

## 11. Project Structure

```
Indian Sign Language Detection/
|-- SignLanguageDetection.ipynb       # full workflow
|-- README.md
|-- isl_realtime.py                   # standalone real-time webcam script
|-- PRAICP-1000-IndiSignLang.zip      # original dataset archive
|-- dataset/                          # extracted data (one folder per letter)
|-- split_dataset/                    # train / val / test folders
|-- hand_crop_dataset/                # hand crops (improved pipeline)
|-- webcam_dataset/                   # your own webcam sessions
`-- models/
    |-- best_custom_cnn.keras
    |-- best_mobilenetv2.keras
    |-- best_mobilenetv2_finetuned.keras
    |-- isl_alphabet_final.keras      # final model
    |-- class_mapping.json            # index -> letter
    `-- hand_landmarker.task          # MediaPipe model
```

---

## 12. Installation and Usage

**Requirements**
```
tensorflow
numpy
pandas
matplotlib
seaborn
scikit-learn
opencv-python
mediapipe
split-folders
pillow
joblib
```

**Install**
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn opencv-python mediapipe split-folders pillow joblib
```

**Run the notebook**
1. Place `PRAICP-1000-IndiSignLang.zip` in the project folder and update the dataset path in the first cells.
2. Run the notebook from top to bottom (*Restart and Run All*) so the results are reproducible.
3. Training on a CPU is slow; for faster training use Google Colab with a GPU, and copy the saved models back to your PC.

**Run the real-time webcam script (on your own PC)**
```bash
python isl_realtime.py --model models/isl_alphabet_final.keras --classes models/class_mapping.json --landmarker models/hand_landmarker.task
```
Keys: `q` quit, `c` clear word, `b` backspace, `space` add a space.

> Webcam code does not work in Google Colab. Train and evaluate in Colab, then run the live script locally.

---

## 13. Limitations

- The model was trained mainly on close-up images on a plain background and does not yet generalize to arbitrary webcam conditions.
- J and Z came from a different capture setup than the other letters, which introduces a shortcut.
- The test set comes from the same sessions as the training data, so test scores are optimistic.
- A single frame cannot represent letters that involve motion.
- Hand detection fails in some frames, which interrupts predictions.
- Letters that differ by a single finger position are easily confused.

---

## 14. Future Work

- Record webcam data for all letters across multiple sessions, people, lighting conditions and backgrounds.
- Train on hand crops with colour and white-balance augmentation.
- Evaluate with a held-out webcam session and report accuracy per letter.
- Use a landmark-based classifier or combine it with the image model.
- Handle motion letters with a sequence model over landmarks.
- Add an "unknown / no sign" class and calibrate the confidence.
- Convert the model to TensorFlow Lite for faster and lighter real-time inference.
- Extend from letters to words and sentences.

---

## 15. Conclusion

This project shows that **transfer learning with MobileNetV2 is clearly better than a small custom CNN** for ISL alphabet recognition: it trains much faster, is more stable and reaches excellent results on the held-out test set. The project also shows that **data quality matters more than model choice**: duplicate frames inflated the first results, and the different capture conditions of J and Z created a shortcut that the model exploited.

In real-time webcam testing the model works only for some letters, because the webcam images differ strongly from the training images in background, scale, lighting and colour, and because "webcam look" became a hint for J and Z. A high offline score therefore proves that the model works on this dataset, not that it works on every camera.

To make the system reliable in practice, the recommended path is: record webcam data for every letter, train on matching hand crops with colour augmentation, evaluate on a separate webcam session, optionally use landmark features, and run the real-time loop with smoothing and confidence thresholds. The notebook and `isl_realtime.py` provide the foundation for these steps.