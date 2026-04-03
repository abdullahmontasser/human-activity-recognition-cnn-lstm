# Human Activity Recognition using CNN-LSTM

A deep learning project for recognizing human activities in videos using a hybrid CNN-LSTM architecture. This project combines the spatial feature extraction power of CNNs with the temporal modeling capabilities of LSTMs.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1gI4DUuzXhZDGE4iYvfHSRJQap4JsVLAG?usp=sharing)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Overview

This project implements a two-stage pipeline for video-based human activity recognition:

1. **Spatial Feature Extraction**: Uses pretrained ResNet50 (frozen weights) to extract high-level visual features from individual video frames
2. **Temporal Sequence Modeling**: Employs LSTM networks to capture motion patterns and temporal dependencies across frame sequences

### Architecture

```
Video → Sample Frames → ResNet50 (frozen) → Feature Vectors → LSTM → Activity Classification
```

## 🚀 Quick Start

### Run in Google Colab (Recommended)

Click the button below to open the notebook in Google Colab with GPU support:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1gI4DUuzXhZDGE4iYvfHSRJQap4JsVLAG?usp=sharing)

**Note**: Make sure to enable GPU: `Runtime → Change runtime type → T4 GPU`

### Local Setup

#### Prerequisites

```bash
pip install tensorflow opencv-python-headless scikit-learn matplotlib seaborn tqdm gdown
```

#### Dataset Structure

Organize your dataset in the following structure:

```
dataset/
├── train/
│   ├── Class1/
│   │   ├── video1.mp4
│   │   ├── video2.mp4
│   │   └── ...
│   ├── Class2/
│   └── ...
├── test/
│   ├── Class1/
│   └── ...
└── val/
    ├── Class1/
    └── ...
```

#### Running Locally

1. Clone this repository:
```bash
git clone https://github.com/abdullahmontasser/human-activity-recognition-cnn-lstm.git
cd human-activity-recognition-cnn-lstm
```

2. Launch Jupyter Notebook:
```bash
jupyter notebook human_activity_recognition.ipynb
```

3. Update dataset paths in the configuration cell:
```python
TRAIN_DIR = '/path/to/your/train'
TEST_DIR = '/path/to/your/test'
VAL_DIR = '/path/to/your/val'
```

## 📊 Dataset

This project supports any video classification dataset with the folder structure above. Examples include:
- **UCF101**: 101 action categories from YouTube
- **HMDB51**: 51 action classes
- **Kinetics**: Large-scale video dataset
- **Custom datasets**: Your own video collection

The notebook automatically selects the **top 5 classes** with the most videos for faster training.

## 🏗️ Model Architecture

### Stage 1: CNN Feature Extractor
- **Model**: ResNet50 (pretrained on ImageNet)
- **Input**: 224×224 RGB frames
- **Output**: 2048-dimensional feature vectors
- **Training**: Frozen (used as fixed feature extractor)

### Stage 2: LSTM Classifier
```
Input (20 frames × 2048 features)
    ↓
LSTM(256 units) + Dropout(0.5)
    ↓
LSTM(256 units) + Dropout(0.5)
    ↓
Dense(256, ReLU) + Dropout(0.5)
    ↓
Dense(num_classes, Softmax)
```

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Frames per video | 20 |
| Frame size | 224×224 |
| LSTM units | 256 |
| Dropout rate | 0.5 |
| Batch size | 8 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Max epochs | 30 |

## 📈 Results

The model achieves strong performance on activity recognition tasks:

- ✅ Per-class accuracy visualization
- ✅ Confusion matrix analysis
- ✅ Sample prediction inspection
- ✅ Training/validation curves

**Example Results** (varies by dataset):
```
Test Accuracy: ~85-95%
Training Time: ~10-15 minutes on Colab T4 GPU
```

## 📁 Project Structure

```
├── human_activity_recognition.ipynb  # Main Jupyter notebook
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
└── LICENSE                            # MIT License
```

## 🔧 Configuration

You can adjust these parameters in the notebook:

```python
NUM_FRAMES = 20          # Frames to extract per video
IMG_SIZE = 224           # Frame resolution
LSTM_UNITS = 256         # LSTM layer size
DROPOUT_RATE = 0.5       # Regularization
BATCH_SIZE = 8           # Training batch size
EPOCHS = 30              # Maximum training epochs
LEARNING_RATE = 0.001    # Adam learning rate
```

## 📊 Evaluation Metrics

The notebook provides comprehensive evaluation:
- Overall test accuracy
- Per-class precision, recall, F1-score
- Confusion matrix heatmap
- Training/validation loss and accuracy curves
- Sample predictions with confidence scores
- Error analysis (misclassification patterns)

## 🎓 Use Cases

This model can be applied to:
- 🎥 **Video surveillance**: Automatic detection of suspicious activities
- 🏃 **Sports analytics**: Analyzing player movements and tactics
- 🏥 **Healthcare**: Patient monitoring, fall detection
- 🎮 **Gaming**: Gesture recognition for interactive experiences
- 🏭 **Industrial safety**: Detecting unsafe worker behaviors

## 🚧 Future Improvements

Potential enhancements:
- [ ] Fine-tune ResNet50 layers (end-to-end training)
- [ ] Add attention mechanism for important frame selection
- [ ] Two-stream architecture (RGB + optical flow)
- [ ] 3D CNN for direct spatiotemporal learning
- [ ] Vision Transformer for sequence modeling
- [ ] Data augmentation (temporal jittering, frame dropout)
- [ ] Real-time inference optimization
- [ ] Model quantization for deployment

## 🤝 Contributing

Contributions are welcome! Please feel free to:
1. Fork the repository
2. Create a feature branch
3. Submit a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Abdullah Montasser**
- GitHub: [@abdullahmontasser](https://github.com/abdullahmontasser)


## 🙏 Acknowledgments

- TensorFlow and Keras teams for the amazing frameworks
- UCF for the UCF101 dataset
- Google Colab for free GPU resources

---

⭐ **If you find this project useful, please give it a star!** ⭐

🔗 **Colab Notebook**: [Open in Google Colab](https://colab.research.google.com/drive/1gI4DUuzXhZDGE4iYvfHSRJQap4JsVLAG?usp=sharing)
