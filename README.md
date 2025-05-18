# 🎨 Neural Style Transfer & High-Resolution Image Upscaling

A deep learning-based project that unifies artistic styles across AI-generated images (from Midjourney, DALL·E, etc.) and upscales them for large-format printing (e.g., 4 ft x 4 ft) using super-resolution techniques like Real-ESRGAN.

---

## 🧠 Problem Statement

AI-generated images often vary in style and resolution, making them unsuitable for consistent presentation or high-quality large prints. This project aims to apply a **consistent artistic style (e.g., oil painting)** across diverse images and upscale them without losing quality.

---

## 🎯 Objective

- Apply **Neural Style Transfer** to maintain consistent "brushstroke" style across multiple images.
- Use **Real-ESRGAN** to upscale the images up to 4x for **large-format print quality**.
- Preserve **details, textures, and artistic fidelity** in the final high-resolution outputs.

---

## 📁 Dataset

### 1. Content Images
- A set of AI-generated images (portraits, landscapes, abstract).
- Sources: Midjourney, DALL·E, or similar tools.

### 2. Style Image
- A high-resolution oil painting (used as a reference for stylistic consistency).

---

## 📊 Exploratory Data Analysis (EDA)

- Image format and resolution distribution
- Color palette and edge histogram comparison
- Noise/artifact analysis (pre-upscaling)

---

## 🤖 Model Selection

### Neural Style Transfer:
- Pre-trained **VGG-19** model used to extract content and style features.
- Style loss + Content loss optimization.

### Super-Resolution:
- **Real-ESRGAN (Enhanced Super-Resolution GAN)**
- Based on RRDBNet architecture
- Pretrained model: `RealESRGAN_x4plus.pth`

---

## 🏗️ Model Architecture

### Neural Style Transfer (NST):
- Uses VGG-19 layers (`conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1`) for style features.
- Optimization done on the pixel space of the target image.

### Real-ESRGAN:
- RRDB blocks with residual-in-residual structure.
- Trained on high-quality image pairs.

---

## ⚙️ How to Use

### 1. Clone the repository

```bash
git clone https://github.com/Drishtichakarvarty/Neural_style_transfer
cd Neural_style_transfer


### 2. Install dependencies

```bash
pip install -r requirements.txt


### 3. Apply Neural Style Transfer

``` bash
python neural_style_transfer.py --content path/to/image.jpg --style path/to/style.jpg --output stylized.jpg

### 4. Upscale Image using Real-ESRGAN

```bash
python upscale.py --input stylized.jpg --output upscaled.jpg
