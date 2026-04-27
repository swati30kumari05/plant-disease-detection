# plant-disease-detection
Plant leaf disease detection using MobileNetV2 and transfer learning, classifying leaf images as healthy or diseased with high accuracy
# Plant Disease Detection using MobileNetV2

A deep learning project for classifying plant leaf images into **Healthy** or **Diseased** classes using **transfer learning with MobileNetV2**. The project uses TensorFlow/Keras, image preprocessing, data augmentation, fine-tuning, and evaluation with confusion matrix and classification metrics.[1][2]

## Project Overview

The goal of this project is to automatically identify whether a plant leaf is healthy or affected by disease from an input image. The system is based on the PlantVillage-style dataset, reorganized into two folders: healthy and diseased, and then split into training, validation, and test sets for supervised learning.[4][5]

The model is implemented in Google Colab using Python, TensorFlow, and Keras. MobileNetV2 is used as a lightweight convolutional neural network backbone because it is efficient and suitable for image classification tasks with limited computational resources.[6][7]

## Features

- Binary classification: Healthy vs Diseased leaf prediction.[5]
- Transfer learning using MobileNetV2 pretrained on ImageNet.[2]
- Data augmentation with random flip, rotation, and zoom.[8]
- Fine-tuning of upper MobileNetV2 layers for improved performance.[1]
- Evaluation using test accuracy, confusion matrix, precision, recall, and F1-score.[1]
- Model saved in `.keras` format for future inference and deployment.[3]

## Tech Stack

- Python
- TensorFlow / Keras[9]
- NumPy
- Matplotlib
- scikit-learn
- Google Colab

## Dataset

The project uses a PlantVillage-style plant leaf image dataset containing healthy and diseased leaf images. The original dataset was reorganized into binary classes and split into:

- `train/healthy`
- `train/diseased`
- `val/healthy`
- `val/diseased`
- `test/healthy`
- `test/diseased`

Images are loaded with `image_dataset_from_directory()` and resized to `160 x 160` pixels before training.[5]

## Model Architecture

The final model consists of:

- Input layer: `160 x 160 x 3`
- Data augmentation layer
- MobileNetV2 base model (`include_top=False`, `weights='imagenet'`)
- GlobalAveragePooling2D
- Dropout (`0.2`)
- Dense output layer with sigmoid activation

This architecture uses MobileNetV2 as the CNN feature extractor and a custom classifier head for binary prediction.[2][1]

## Training Strategy

Training is completed in two stages:[1]

1. **Feature Extraction Phase**
   - Freeze the MobileNetV2 base model
   - Train only the new classification head

2. **Fine-Tuning Phase**
   - Unfreeze the upper layers of MobileNetV2
   - Recompile with a lower learning rate
   - Continue training for better adaptation to plant leaf features

## Evaluation Metrics

The model is evaluated using:[1]

- Accuracy
- Loss
- Confusion Matrix
- Precision
- Recall
- F1-score

## Results

The fine-tuned model achieved high performance on the test set, with test accuracy around **98–99%** in the final run. Accuracy/loss plots, confusion matrix, and sample single-image predictions were also generated for analysis and reporting.[8]

## Project Structure

```text
plant-disease-detection/
├── plant_disease_detection.ipynb
├── README.md
├── fig_accuracy.png
├── fig_loss.png
├── confusion_matrix.png
├── sample_prediction.png
├── plant_binary_model_finetuned.keras
└── requirements.txt
```

## How to Run

1. Open the notebook in Google Colab.
2. Mount Google Drive if the saved model or figures are stored there.
3. Prepare the dataset in train/validation/test folder structure.
4. Run the notebook cells in order.
5. Train and fine-tune the model.
6. Save the final model using:

```python
model.save("/content/drive/MyDrive/plant_binary_model_finetuned.keras")
```

7. Reload later using:

```python
loaded_model = tf.keras.models.load_model(
    "/content/drive/MyDrive/plant_binary_model_finetuned.keras"
)
```

## Future Improvements

- Extend from binary classification to multi-class disease identification.[4]
- Build a Streamlit or web application for real-time prediction.
- Deploy on mobile or edge devices using model optimization techniques.[2]
- Use more diverse real-world field images for stronger generalization.[10]

## Acknowledgements

- TensorFlow / Keras documentation for model building, transfer learning, and model saving.[1][3][2]
- PlantVillage dataset and related plant disease detection research.[4]
