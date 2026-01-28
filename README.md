# Re:bble 2D Scanner

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18404477.svg)](https://doi.org/10.5281/zenodo.18404477)

An image processing tool that scans rubble shapes in 2D using phone pictures through  markers detection, image segmentation and perspective transformation.

![Detecting 2D Rubble shape using phone pictures](<visuals/MAXENCE GRANGEOT_RE-BBLE_2D_SCANNER_IMG-01_LR.png>)

## Overview

This script processes batch images containing ArUco markers and surrounding objects. It:

1. **Detects scale markers** using OpenCV 
2. **Segments rubble shapes** using Meta's Segment Anything Model 2
3. **Performs perspective transformation** to straighten the image based on camera angle
4. **Extracts contours** and exports them in a CSV format with meter-based coordinates

## Usage

### Using the custom markers

The custom markers used for the detection, naming and image straigthening are provided in the [aruco markers folder](<aruco markers>). 
The script for 2D scanning supports four ArUco marker dictionary sizes:

| Size | Dictionary | Marker Size | Printed Support Size |
|------|-----------|------------------------|----------|
| S    | 4×4_250   | 10cm                   | A5
| M    | 5×5_250   | 14cm                   | A4
| L    | 6×6_250   | 20cm                   | A3

**Important**: Update `pixel_to_meter_ratio_*` values based on your actual printed marker sizes.

### Prepare Your Files

1. Take a single picture per rubble piece, as vertical as possible above it. The entire perimeter of each unit should be visible in the picture in order to correctly segment each image. 

2. Transfer the phone pictures to your computer and place your JPG images in `../input/`

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
svgwrite (optional, for aruco marker generation)
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

## Generating ArUco Markers

The script `Re-bble_Aruco Generator.py` is provided if you need to generate your own ArUco markers. The generated markers are stored in the `aruco_markers` directory. 
To generate printable files from them, I used Indesign datamerge. Feel free to use other methods.

### Instructions to Generate and Print ArUco Markers:
1. **Run the Script**: Execute the `Re-bble_Aruco Generator.py` script. This will create the markers and save them in the specified output directory.
2. **Locate the Markers**: After running the script, navigate to the `aruco_markers` directory. You will find subdirectories for each marker size (extra small, small, medium, large).
3. **Printing the Markers**: Open the SVG files in a compatible viewer or web browser. You can print them directly from there. Ensure that the print settings maintain the original size of the markers for accurate detection during scanning.

## License

[MIT License](./LICENSE)
Copyright (c) 2025 Maxence Grangeot EPFL

## Context and use case

This tool was developped as part of the [PhD Research](https://go.epfl.ch/rubble-reuse) of Maxence Grangeot at SXL, EPFL
It was used to help prefabricated walls from concrete rubble to drastically reduce the environmental impact of concrete structures.
Re:bble Prefa walls:
![alt text](<visuals/Maxence GRANGEOT_SXL-CRCL-EPFL_prototype-07_prefabrication_HD_image-126.jpg>)
Re:bble Tower - a two-storey demonstrator from concrete rubble
![alt text](<visuals/Maxence GRANGEOT_SXL-CRCL-EPFL_prototype-07_tower_HD_image_3_no columns brighter.jpg>)
Re:bble Pavilion - a public installation to showcase this innovative construction method
![alt text](<visuals/Maxence GRANGEOT_SXL-CRCL-EPFL_prototype-08_image_11.jpg>)

## Aknowlegdements

If you use this tool in research, please cite:

- **SAM2**: Ravi et al. 2024, "SAM 2: Segment Anything in Images and Videos"
- **OpenCV**: Bradski, G. et al., OpenCV library, https://opencv.org/
- **ArUco**: Garrido-Jurado, S., et al. (2014), "Automatic generation and detection of highly reliable fiducial markers under occlusion"


## How to cite

To cite the software follow the following BibTeX entry:

```bibtex
@software{Re-bbleScanner2025,
    title = {{Re:bble 2D Scanner}},
    author = {Maxence Grangeot},
    year = {2025},
    doi = {10.5281/zenodo.18404477},
    url = {https://github.com/Grangeot/Re-bble_2D_Scanner}
}
```

or the associated Zenodo DOI:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18404477.svg)](https://doi.org/10.5281/zenodo.18404477)


## Disclaimer
This software is provided "as-is" as a prototype, without any warranties, express or implied, including but not limited to fitness for a particular purpose or non-infringement. The user assumes full responsibility for the use of the software, and we are not liable for any damages, losses, or misuse arising from its use. By using this software, you agree to these terms.