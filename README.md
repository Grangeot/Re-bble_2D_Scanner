# Re:bble 2D Scanner

An image processing tool that scans rubble shapes in 2D using phone pictures through aruco markers, image segmentation and perspective transformation.

## Overview

This script processes batch images containing ArUco markers and surrounding objects. It:

1. **Detects ArUco markers** using OpenCV (supports 4 marker sizes: XS, S, M, L)
2. **Segments objects** using Meta's SAM2 (Segment Anything Model 2) with prompt points
3. **Performs perspective transformation** to straighten the image based on marker position
4. **Extracts contours** and exports them in a standardized CSV format with meter-based coordinates

### Key Features

- Multi-scale ArUco marker detection (XS: 7×7, S: 4×4, M: 5×5, L: 6×6)
- Automatic pixel-to-meter conversion based on marker size
- SAM2-powered segmentation for accurate object detection
- Perspective transformation for straightened output
- Batch processing with detailed logging
- Contour simplification and center normalization
- GPU/MPS/CPU device auto-detection

## Requirements

### System Requirements
- Python 3.10 (recommended for SAM2 compatibility)
- CUDA 11.8+ (for GPU acceleration, optional)
- macOS with Apple Silicon (optional, uses MPS backend) or Linux/Windows with GPU support

### Python Dependencies

```
opencv-contrib-python>=4.8.0
numpy>=1.24.0
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.5.0
matplotlib>=3.7.0
sam2
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Grangeot/Re-bble_2D_Scanner.git
```

### 2. Create a Python Virtual Environment

```bash
# Create virtual environment with Python 3.10
python3.10 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 3. Install PyTorch

Choose the appropriate installation command based on your hardware:

**For GPU (CUDA 11.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For macOS (Apple Silicon):**
```bash
pip install torch torchvision torchaudio
```

**For CPU only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install SAM2

```bash
pip install git+https://github.com/facebookresearch/sam2.git
```

### 5. Download SAM2 Checkpoints

```bash
# Create checkpoints directory
mkdir -p checkpoints
cd checkpoints

# Download the base+ model (recommended for balance of speed/accuracy, ~3.5s per image)
wget https://dl.fbaipublicfiles.com/segment_anything_2/sam2_hiera_base_plus/sam2.1_hiera_base_plus.pt

# Alternative models:
# Large model (5s per image, better accuracy)
# wget https://dl.fbaipublicfiles.com/segment_anything_2/sam2_hiera_large/sam2.1_hiera_large.pt

# Tiny model (2.7s per image, faster)
# wget https://dl.fbaipublicfiles.com/segment_anything_2/sam2_hiera_tiny/sam2.1_hiera_tiny.pt

# Also download configs
cd ..
git clone https://github.com/facebookresearch/segment-anything-2.git && cp -r segment-anything-2/sam2/configs .
```

### 6. Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install opencv-contrib-python numpy Pillow matplotlib
```

## Configuration

Edit the script to customize these parameters:

```python
# Input/Output paths
input_folder = "../input/"      # Directory containing input images
output_folder = "../output/"    # Directory for output files
batch_number = "A"                  # Batch identifier

# Marker detection
MARKER_SIZE = 500                   # Marker size in pixels
MARKER_THICKNESS = 5                # Thickness of marker border visualization
query_spacing = 0.75                # Query point offset multiplier (0-1)

# Pixel-to-meter ratios (adjust based on your marker physical size)
pixel_to_meter_ratio_XS = 0.07 / MARKER_SIZE   # 7cm marker (A5)
pixel_to_meter_ratio_S = 0.10 / MARKER_SIZE    # 10cm marker (A5)
pixel_to_meter_ratio_M = 0.14 / MARKER_SIZE    # 14cm marker (A4)
pixel_to_meter_ratio_L = 0.2003 / MARKER_SIZE  # 20.03cm marker (A3)

# SAM2 model selection
# Options: sam2.1_hiera_tiny (fast), sam2.1_hiera_base_plus (balanced), sam2.1_hiera_large (accurate)
sam2_checkpoint = "checkpoints/sam2.1_hiera_base_plus.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
```

## Usage

### Prepare Your Files

1. Create input directory structure:
```bash
mkdir -p ../input/
mkdir -p ../output/
```

2. Place your JPG images in `../input/`

### Run the Script

```bash
python "Re-bble_2D_Scanner.py"
```

The script will:
- Process all JPG images in the input folder
- Generate output files
- Create a processing log

### Output Files

After processing, you'll find:

- **Marked-up images**: `1_detected_markers_query_points_segmented_outline_[MARKER]_[INDEX].jpg`
  - Shows detected markers, SAM query points, and segmented outlines
  
- **Straightened images**: `2_straightened_photo_only_[MARKER]_[INDEX].jpg`
  - Perspective-corrected images aligned with the marker

- **Processing log**: `processing_log_[TIMESTAMP].csv`
  - Detailed log of all processed images including errors and timing

- **Contours data**: `all_contours_[TIMESTAMP].csv`
  - Extracted object contours in meter-based coordinates
  - Format: `MarkerID, Point1_X,Point1_Y; Point2_X,Point2_Y; ...`

## Understanding the Output CSV

### Contours CSV Format

Each row contains:
- **Marker ID**: `[SIZE]_[BATCH]_[ID]_[INDEX]` (e.g., `L_A_15_1`)
- **Points**: Semicolon-separated coordinates in meters, relative to object center
- **Format**: `x1,y1;x2,y2;x3,y3;...`

Example:
```
L_A_15_1,-0.25,0.30;-0.20,0.35;-0.15,0.33;-0.20,0.28
```

## Marker Sizes

The script supports four ArUco marker dictionary sizes:

| Size | Dictionary | Physical Size (Standard) | Use Case |
|------|-----------|------------------------|----------|
| XS   | 7×7_250   | 7cm (A5)               | Small objects |
| S    | 4×4_250   | 10cm (A5)              | Small-medium |
| M    | 5×5_250   | 14cm (A4)              | Medium objects |
| L    | 6×6_250   | 20cm (A3)              | Large objects/rubble |

**Important**: Update `pixel_to_meter_ratio_*` values based on your actual printed marker sizes.

## Troubleshooting

### Issue: ArUco markers not detected
1. Ensure markers are printed clearly with good contrast
2. Check `parameters.adaptiveThreshWinSizeMin/Max` values
3. Adjust `query_spacing` parameter
4. Verify marker dictionary matches printed markers

### Issue: Segmentation inaccurate
1. Try adjusting `query_spacing` (0.5-1.0)
2. Switch to larger SAM2 model for better accuracy
3. Ensure good image lighting and contrast


## Advanced Configuration

### Disable specific debug output
Uncomment lines in the save images section to skip intermediate visualizations:
```python
#cv2.imwrite(...)  # Comment out to skip saving
```

### Stop after first image
For testing, uncomment:
```python
break  #! Stop after processing the first image
```

### Adjust ArUco detection sensitivity
```python
parameters.adaptiveThreshWinSizeMin = 3      # Smaller = more sensitive
parameters.adaptiveThreshWinSizeMax = 23     # Larger = broader search
parameters.minMarkerPerimeterRate = 0.03     # Minimum marker size ratio
```

## License

MIT License
Maxence Grangeot - EPFL

## Authors

[Add author information]

## Aknowlegdements

If you use this tool in research, please cite:

- **SAM2**: Ravi et al. 2024, "SAM 2: Segment Anything in Images and Videos"
- **OpenCV**: Bradski, G. et al., OpenCV library, https://opencv.org/
- **ArUco**: Garrido-Jurado, S., et al. (2014), "Automatic generation and detection of highly reliable fiducial markers under occlusion"
