# -U-Net-Image-Segmentation-Model-TensorFlow-Keras-

## 📌 Overview
This project implements a **U-Net** model for **binary image segmentation** using TensorFlow and Keras.  
U-Net is a convolutional neural network designed for precise pixel-wise segmentation, originally proposed for biomedical image analysis.  
Our implementation follows the classic encoder–decoder architecture with skip connections.

---

## 🧩 Architecture
The network consists of four key parts:

1. **Encoder (Contracting Path):**  
   Extracts features and progressively downsamples the image using convolution and max pooling layers.

2. **Bottleneck:**  
   Connects encoder and decoder; captures the deepest contextual information.

3. **Decoder (Expanding Path):**  
   Uses transposed convolutions to upsample feature maps and concatenates them with corresponding encoder features to recover spatial detail.

4. **Output Head:**  
   A 1×1 convolution produces the final segmentation mask using a sigmoid activation for binary classification.

---

## ⚙️ Model Components

### 🧱 `double_conv_block(x, num_f)`
Two consecutive 3×3 convolutions (stride=1, padding="same")  
Each followed by BatchNorm and ReLU activations.

### 🔽 `encoder_block(x, num_f)`
Applies `double_conv_block` for feature extraction, then a 2×2 MaxPooling for downsampling.  
Returns both:
- `x`: features before pooling (for skip connection)
- `p`: pooled features (input for the next stage)

### 🔼 `decoder_block(x, skip, num_f)`
Upsamples using a 2×2 `Conv2DTranspose`, concatenates with the corresponding skip connection,  
and refines using a `double_conv_block`.

### 🧬 `make_unet(input_shape, base_num_f, num_classes, final_act)`
Integrates all encoder, bottleneck, and decoder blocks into a complete U-Net model.  
- `input_shape`: image dimensions, e.g., (256, 256, 3)  
- `base_num_f`: base number of filters (default 32)  
- `num_classes`: 1 for binary segmentation  
- `final_act`: activation function (e.g., `"sigmoid"`)

---
