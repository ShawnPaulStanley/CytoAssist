# CytoAssist

AI-Assisted FNAC Screening Tool - A decision-support system for cytology image screening using deep learning.

## Overview

CytoAssist is a ResNet18-based classification tool designed to assist in Fine Needle Aspiration Cytology (FNAC) screening. The system analyzes cytology images and classifies them as either Benign or Suspicious, providing visual explanations through Grad-CAM heatmaps.

**Disclaimer**: This tool is for educational and research purposes only. It does not provide medical diagnoses and should not be used for clinical decision-making.

## Features

- Binary classification of cytology images (Benign/Suspicious)
- Grad-CAM visualization for model interpretability
- Interactive web interface built with Streamlit
- Real-time inference with detailed progress logging
- Support for common image formats (PNG, JPG, JPEG)

## Prerequisites

- Python 3.8 or higher
- Git

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ShawnPaulStanley/CytoAssist.git
cd CytoAssist
```

2. Create and activate a virtual environment (recommended):

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install required dependencies:
```bash
pip install streamlit torch torchvision pillow numpy opencv-python matplotlib scikit-learn
```

## Running the Application

1. Navigate to the backend directory:
```bash
cd backend
```

2. Start the Streamlit application:
```bash
streamlit run app.py
```

3. Open your web browser and navigate to:
```
http://localhost:8501
```

The application will automatically load the pre-trained model (`cytoassist_resnet18.pth`) from the backend directory.

## Project Structure

```
CytoAssist/
├── backend/
│   ├── app.py                      # Main Streamlit application
│   └── cytoassist_resnet18.pth     # Pre-trained model weights
├── frontend/
│   ├── index.html                  # Frontend UI template
│   └── fonts/                      # Custom font files
└── README.md
```

## Usage

1. Launch the application using the instructions above
2. Use the file uploader in the sidebar to select a cytology image
3. The system will:
   - Display the uploaded image
   - Run the classification model
   - Show prediction results with confidence scores
   - Generate a Grad-CAM heatmap highlighting regions of interest
4. Review the results in the interactive interface

## Model Details

- Architecture: ResNet18 with custom classification head
- Input: 224x224 RGB images
- Normalization: ImageNet mean and standard deviation
- Output: Binary classification (Benign vs. Suspicious)
- Activation: Sigmoid output layer

## Technical Requirements

Key dependencies:
- streamlit
- torch
- torchvision
- pillow
- numpy
- opencv-python
- matplotlib
- scikit-learn

## Troubleshooting

**Model not found error:**
Ensure the `cytoassist_resnet18.pth` file is located in the `backend/` directory.

**Port already in use:**
If port 8501 is occupied, Streamlit will automatically use the next available port. Check the terminal output for the correct URL.

**Import errors:**
Verify all dependencies are installed in your active Python environment using:
```bash
pip list
```

## License

This project is for educational and research purposes only.

## Acknowledgments

Built with Streamlit, PyTorch, and torchvision.
