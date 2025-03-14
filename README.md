# UNet Implementation in TensorFlow

This repository contains an implementation of the UNet architecture for image segmentation using TensorFlow and Keras.

## 📌 About UNet

UNet is a convolutional neural network (CNN) primarily used for image segmentation tasks. It is widely applied in medical image segmentation, object detection, and other vision applications requiring precise pixel-wise classification. The architecture consists of an encoder (contracting path), a bottleneck, and a decoder (expanding path) with skip connections to retain spatial information.

## 🚀 Features
- Fully Convolutional Neural Network
- Skip connections for better spatial feature retention
- Batch Normalization for stable training
- Conv2DTranspose for upsampling
- Sigmoid activation for binary segmentation

## 🛠️ Installation
Ensure you have Python installed along with the required dependencies. You can install TensorFlow and other dependencies using:

```sh
pip install tensorflow
```

## 📂 File Structure
- `unet.py`: Contains the UNet model implementation
- `README.md`: Documentation for the repository

## 📜 Model Architecture
The UNet model consists of:
- **Encoder (Downsampling Path)**: Extracts features using Conv2D, BatchNormalization, and ReLU activation, followed by MaxPooling to reduce spatial size.
- **Bottleneck**: The deepest part of the network where the most abstract features are captured.
- **Decoder (Upsampling Path)**: Uses Conv2DTranspose to restore spatial dimensions while concatenating with skip connections for better feature retention.
- **Output Layer**: A `1x1` convolution with sigmoid activation for binary segmentation.

## 🏗️ How to Use
To run the model and see the summary, execute:

```sh
python unet.py
```

This will display the model architecture.

## 🖼️ Example Input Shape
The model accepts images of shape `(256, 256, 3)`, where:
- `256x256` is the image size
- `3` represents RGB channels

## 🏆 Model Summary
After running the script, the model summary will be displayed, showing:
- Number of layers
- Number of parameters
- Input/output shape

