#requires environment with sam2 installed (phd_3-10)
import cv2
from cv2 import aruco
import numpy
import os
import time
import csv
import sys
import torch
import matplotlib.pyplot as  plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from datetime import datetime

print("Script initialized")

input_folder = "../input/muro"
output_folder = "../output/muro"
batch_number = "A"
marker_count = {}
all_contours = []

# Constants for image dimensions and marker size
MARKER_SIZE = 500 #pixels. Rubble size approx 80cm diameter for 17cm marker printed on A3
MARKER_THICKNESS = 5  # Parameter for detected marker thickness
query_spacing = 0.75 #Multiplier for spacing of query points from centroid of marker to corners

# Define the pixel to meter ratio based on the marker size
pixel_to_meter_ratio_XS = 0.07 / MARKER_SIZE  # Assuming 10.0cm marker printed on A5 #! Measure to be sure
pixel_to_meter_ratio_S = 0.10 / MARKER_SIZE  # Assuming 10.0cm marker printed on A5 #! Measure to be sure
pixel_to_meter_ratio_M = 0.14 / MARKER_SIZE  # Assuming 14.0cm marker printed on A4 #! Measure to be sure
pixel_to_meter_ratio_L = 0.2003 / MARKER_SIZE  # Assuming 20.03cm marker printed on A3 #! Measure to be sure

# Load the dictionary that was used to generate the markers.
aruco_dict_XS = aruco.getPredefinedDictionary(aruco.DICT_7X7_250) #DICT_7X7_250
aruco_dict_S = aruco.getPredefinedDictionary(aruco.DICT_4X4_250) #DICT_4X4_250
aruco_dict_M = aruco.getPredefinedDictionary(aruco.DICT_5X5_250) #DICT_5X5_250
aruco_dict_L = aruco.getPredefinedDictionary(aruco.DICT_6X6_250) #DICT_6X6_250

# Initialize the detector parameters using default values
parameters = aruco.DetectorParameters()
# Adjust parameters if needed
parameters.adaptiveThreshWinSizeMin = 3
parameters.adaptiveThreshWinSizeMax = 23
parameters.minMarkerPerimeterRate = 0.03

# Initialize ArucoDetector objects for each dictionary
detector_XS = cv2.aruco.ArucoDetector(aruco_dict_XS, parameters)
detector_S = cv2.aruco.ArucoDetector(aruco_dict_S, parameters)
detector_M = cv2.aruco.ArucoDetector(aruco_dict_M, parameters)
detector_L = cv2.aruco.ArucoDetector(aruco_dict_L, parameters)

# select the device for computation
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print("mps device")

numpy.random.seed(3)

# Load the SAM2 model
#sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt" # 5s per image 
#model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml" # 5s per image
#sam2_checkpoint = "checkpoints/sam2.1_hiera_tiny.pt" # 2.7s per image 
#model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml" # 2.7s per image
sam2_checkpoint = "checkpoints/sam2.1_hiera_base_plus.pt" # 3.5s per image 
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml" # 3.5s per image

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)

# Function to draw borders around detected markers
def draw_borders(image, corners, thickness):
    for corner in corners:
        corner = corner[0]
        for i in range(4):
            start_point = tuple(map(int, corner[i]))
            end_point = tuple(map(int, corner[(i + 1) % 4]))
            cv2.line(image, start_point, end_point, (0, 255, 0), thickness)
    return image

# Function to draw prompt points on the image
def draw_prompt_points(image, points, radius=10, color=(0, 0, 255)):
    for point in points:
        cv2.circle(image, tuple(map(int, point)), radius, color, -1)
    return image

# Function to create a transparent background image
def create_transparent_image(width, height):
    return numpy.zeros((height, width, 4), dtype=numpy.uint8)

# Function to draw borders around detected markers on a transparent background
def draw_borders_transparent(corners, thickness, width, height):
    image = create_transparent_image(width, height)
    for corner in corners:
        corner = corner[0]
        for i in range(4):
            start_point = tuple(map(int, corner[i]))
            end_point = tuple(map(int, corner[(i + 1) % 4]))
            cv2.line(image, start_point, end_point, (0, 255, 0, 255), thickness)
    return image

# Function to draw prompt points on a transparent background
def draw_prompt_points_transparent(points, radius, color, width, height):
    image = create_transparent_image(width, height)
    for point in points:
        cv2.circle(image, tuple(map(int, point)), radius, color, -1)
    return image

# Function to draw contours on a transparent background
def draw_contours_transparent(contours, image_width, image_height):
    image = create_transparent_image(image_width, image_height)
    cv2.drawContours(image, contours, -1, (255, 0, 0, 255), 2)
    return image

# Function to segment the shape surrounding the ArUco marker using SAM
def segment_shape_sam(image, corners, spacing):
    # Calculate the centroid of the corners
    centroid = numpy.mean(corners[0][0], axis=0)
    
    # Calculate vectors from the centroid to each corner
    vectors = corners[0][0] - centroid
    
    # Offset the points outward by adding the scaled vectors to the original points
    offset_corners = corners[0][0] + vectors*spacing
    
    # Use the offset corners as point prompts
    point_prompts = numpy.array(offset_corners, dtype="float32")
    point_labels = numpy.ones(point_prompts.shape[0])  # All points are foreground points

    predictor.set_image(image)

    masks, _, _ = predictor.predict(
        point_coords=point_prompts,
        point_labels=point_labels,
        multimask_output=False,
    )

    masks.shape  # (number_of_masks) x H x W
    
    return masks[0], point_prompts

def clean_segmented_image(segmented_image):
    # Find contours in the segmented image
    contours, _ = cv2.findContours(segmented_image.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create an empty mask with the correct data type
    cleaned_image = numpy.zeros_like(segmented_image, dtype="uint8")
    
    # Draw the largest contour on the mask
    cv2.drawContours(cleaned_image, [largest_contour], -1, 1, thickness=cv2.FILLED)
    
    return cleaned_image

def overlay_contour(image, mask):
    contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (255, 0, 0), 2)
    return image

# Function to find the marker closest to the center of the image
def find_closest_marker(corners, image_center):
    min_distance = float('inf')
    closest_marker_index = -1
    for i, corner in enumerate(corners):
        centroid = numpy.mean(corner[0], axis=0)
        distance = numpy.linalg.norm(centroid - image_center)
        if distance < min_distance:
            min_distance = distance
            closest_marker_index = i
    return closest_marker_index

def simplify_contour(contour, max_points=300):
    epsilon = 0.001 * cv2.arcLength(contour, True)
    simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
    while len(simplified_contour) > max_points:
        epsilon *= 1.05
        simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
    return simplified_contour

def convert_points_to_meters(points, pixel_to_meter_ratio):
    return points * pixel_to_meter_ratio
    
def center_contour(contour):
    contour = numpy.array(contour, dtype=numpy.float32)  # Ensure the contour is in the correct format
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx, cy = 0, 0
    centered_contour = contour - [cx, cy]
    return centered_contour

# Function to save all contours to a single CSV file
def save_all_contours_as_csv(all_contours, output_path, max_points=300):
    with open(output_path, "w", newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for marker_key, index, contours in all_contours:
            # Choose ratio based on marker prefix
            if str(marker_key).startswith("XS"):
                ratio = pixel_to_meter_ratio_XS
            if str(marker_key).startswith("S"):
                ratio = pixel_to_meter_ratio_S
            elif str(marker_key).startswith("M"):
                ratio = pixel_to_meter_ratio_M
            elif str(marker_key).startswith("L"):
                ratio = pixel_to_meter_ratio_L
            else:
                ratio = pixel_to_meter_ratio_S
            contour_points_list = []
            # Use only the largest contour to avoid duplicate export of similar points
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                simplified_contour = simplify_contour(largest_contour, max_points)
                simplified_contour = convert_points_to_meters(simplified_contour, ratio)
                simplified_contour = center_contour(simplified_contour)
                contour_points = ";".join([f"{point[0][0]},{-point[0][1]}" for point in simplified_contour])
                contour_points_list.append(contour_points)
            csv_writer.writerow([f"{marker_key}_{index}"] + contour_points_list)

# Function to overlay multiple images
def overlay_images(base_image, overlays):
    for overlay in overlays:
        alpha = overlay[:, :, 3] / 255.0
        for c in range(0, 3):
            base_image[:, :, c] = base_image[:, :, c] * (1 - alpha) + overlay[:, :, c] * alpha
    return base_image

# Function to draw marker IDs on the image
def draw_marker_ids(image, corners, ids):
    if ids is not None:
        for i, corner in enumerate(corners):
            centroid = numpy.mean(corner[0], axis=0)
            marker_id = ids[i][0]
            cv2.putText(image, str(marker_id), tuple(map(int, centroid)), cv2.FONT_HERSHEY_SIMPLEX, 
                        5.5, (255, 255, 255), 2, cv2.LINE_AA)
    return image

# Quick fix for specific ID misinterpretation
def fix_marker_id(prefix, marker_id):
    if prefix == "S" and marker_id == 108:
        return "L", 22  # Correct the prefix and ID
    return prefix, marker_id

# Get the list of image files to process
image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".JPG") or f.endswith(".jpg")])
total_images = len(image_files)

print(f"Start processing of {total_images} images...")

# Get the current date and time for file suffix
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

# Open the log file for writing as CSV
log_file_path = os.path.join(output_folder, f"processing_log_{current_datetime}.csv")
with open(log_file_path, "w", newline='') as log_file:
    log_writer = csv.writer(log_file)
    log_writer.writerow(["Filename", "Detected Markers", "Size_Marker ID_Index", "Segmented", "Processing Time (s)", 
                         "Tranformed Upper Left Corner", "Tranformed Upper Right Corner", "Tranformed Lower Left Corner", "Tranformed Lower Right Corner"])
    
    for index, filename in enumerate(image_files):
        start_time = time.time()
        try:
            im = cv2.imread(os.path.join(input_folder, filename))
            
            # Ensure image is in portrait mode
            if im.shape[0] < im.shape[1]:
                im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            # Get dynamic dimensions from the image
            image_height, image_width = im.shape[:2]
            
            sys.stdout.write(f"[{index + 1:03d}/{total_images:03d}] Processing {filename} [{image_width}x{image_height}]...")
            sys.stdout.flush()

            # Convert the image to grayscale
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            
            # Detect markers using the new detectors
            corners_XS, ids_XS, _ = detector_XS.detectMarkers(gray)
            corners_S, ids_S, _ = detector_S.detectMarkers(gray)
            corners_M, ids_M, _ = detector_M.detectMarkers(gray)
            corners_L, ids_L, _ = detector_L.detectMarkers(gray)
            
            all_corners = []
            all_ids = []
            all_prefix = []
            if ids_XS is not None:
                for i in range(len(ids_XS)):
                    all_corners.append(corners_XS[i])
                    all_ids.append(ids_XS[i])
                    all_prefix.append("XS")
            if ids_S is not None:
                for i in range(len(ids_S)):
                    all_corners.append(corners_S[i])
                    all_ids.append(ids_S[i])
                    all_prefix.append("S")
            if ids_M is not None:
                for i in range(len(ids_M)):
                    all_corners.append(corners_M[i])
                    all_ids.append(ids_M[i])
                    all_prefix.append("M")
            if ids_L is not None:
                for i in range(len(ids_L)):
                    all_corners.append(corners_L[i])
                    all_ids.append(ids_L[i])
                    all_prefix.append("L")
            
            # Extract the corners if any markers were detected
            if len(all_corners) > 0:
                # Find the marker closest to the center of the image
                image_center = numpy.array([image_width / 2, image_height / 2])
                closest_marker_index = find_closest_marker(all_corners, image_center)
                
                src_points = numpy.array(all_corners[closest_marker_index][0], dtype="float32")
                marker_id = all_ids[closest_marker_index][0]
                prefix = all_prefix[closest_marker_index]
                
                # Apply the quick fix for the specific ID issue
                prefix, marker_id = fix_marker_id(prefix, marker_id)
                
                marker_key = f"{prefix}_{batch_number}_{marker_id}"
                
                # Handle duplicate marker IDs per marker type
                if marker_key not in marker_count:
                    marker_count[marker_key] = 0
                marker_count[marker_key] += 1
                index = marker_count[marker_key]
                
                # Visualize detected markers on a transparent background
                im_with_markers = draw_borders_transparent(all_corners, MARKER_THICKNESS, image_width, image_height)
                
                # Draw the four corners of the marker as circles on a transparent background
                im_with_corners = draw_prompt_points_transparent(src_points, 10, (0, 0, 255, 255), image_width, image_height)
                
                # Segment the shape surrounding the ArUco marker on the original image using SAM
                binary_segmented, point_prompts = segment_shape_sam(im, [all_corners[closest_marker_index]], query_spacing)
                
                # Draw the 4 query points for the segmentation on a transparent background
                im_with_query_points = draw_prompt_points_transparent(point_prompts, 10, (0, 0, 255, 255), image_width, image_height)
                
                # Clean the segmented image to retain only the biggest shape
                cleaned_segmented = clean_segmented_image(binary_segmented)
                
                # Draw the segmented outline on a transparent background
                contours, _ = cv2.findContours(cleaned_segmented.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                im_with_segmented_outline = draw_contours_transparent(contours, image_width, image_height)
                
                # Add simplified contours output file
                simplified_contours = [simplify_contour(c) for c in contours]
                im_with_simplified_outline = draw_contours_transparent(simplified_contours, image_width, image_height)
                
                # Define the destination points for homography
                dst_points = numpy.array([
                    [image_width/2 - MARKER_SIZE/2, image_height/2 - MARKER_SIZE/2],  # Top-left
                    [image_width/2 + MARKER_SIZE/2, image_height/2 - MARKER_SIZE/2],  # Top-right
                    [image_width/2 + MARKER_SIZE/2, image_height/2 + MARKER_SIZE/2],  # Bottom-right
                    [image_width/2 - MARKER_SIZE/2, image_height/2 + MARKER_SIZE/2]   # Bottom-left
                ], dtype="float32")
                
                # Compute the homography matrix
                H = cv2.findHomography(src_points, dst_points, cv2.RANSAC)
                
                # Apply the homography transformation to the original image
                imDef = cv2.warpPerspective(im, H[0], (im.shape[1], im.shape[0]))
                
                # Apply the homography transformation to the segmented image
                segmented_transformed = cv2.warpPerspective(cleaned_segmented, H[0], (im.shape[1], im.shape[0]))
                
                # Get the coordinates of the corners of the transformed image
                original_corners = numpy.array([
                    [0, 0],  # Upper left
                    [image_width, 0],  # Upper right
                    [image_width, image_height],  # Lower right
                    [0, image_height]  # Lower left
                ], dtype="float32")
                transformed_corners = cv2.perspectiveTransform(numpy.array([original_corners]), H[0])[0]
                upper_left = numpy.round(transformed_corners[0].tolist(), 1)
                upper_right = numpy.round(transformed_corners[1].tolist(), 1)
                lower_right = numpy.round(transformed_corners[2].tolist(), 1)
                lower_left = numpy.round(transformed_corners[3].tolist(), 1)
                
                # Draw prompt points on the detected markers image
                im_with_prompts = draw_prompt_points(im_with_markers.copy(), point_prompts)
                
                # Overlay the marker outline, query points, and segmented outline on the source image
                overlays = [im_with_markers, im_with_query_points, im_with_segmented_outline]
                im_with_contour = overlay_images(im.copy(), overlays)

                # Draw marker IDs on the image
                im_with_contour = draw_marker_ids(im_with_contour, all_corners, all_ids)
                
                # Save the images for results and troubleshooting
                #cv2.imwrite(os.path.join(output_folder, f"0_source_image_{marker_key}_{index}.jpg"), im)
                cv2.imwrite(os.path.join(output_folder, f"1_detected_markers_query_points_segmented_outline_{marker_key}_{index}.jpg"), im_with_contour)
                cv2.imwrite(os.path.join(output_folder, f"2_straightened_photo_only_{marker_key}_{index}.jpg"), imDef)
                #cv2.imwrite(os.path.join(output_folder, f"3_segmented_transformed_{marker_key}_{index}.jpg"), (segmented_transformed * 255).astype("uint8"))
                #cv2.imwrite(os.path.join(output_folder, f"4_detected_markers_{marker_key}_{index}.png"), im_with_markers)
                #cv2.imwrite(os.path.join(output_folder, f"5_corners_{marker_key}_{index}.png"), im_with_corners)
                #cv2.imwrite(os.path.join(output_folder, f"6_query_points_{marker_key}_{index}.png"), im_with_query_points)
                #cv2.imwrite(os.path.join(output_folder, f"7_segmented_outlines_{marker_key}_{index}.png"), im_with_segmented_outline)
                #cv2.imwrite(os.path.join(output_folder, f"8_segmented_outline_simplified_{marker_key}_{index}.png"), im_with_simplified_outline)

                # Extract contours from the segmented transformed image
                new_transformed_contours = []
                for c in contours:
                    pts = c.reshape(-1,1,2).astype(numpy.float32)
                    transformed = cv2.perspectiveTransform(pts, H[0])
                    new_transformed_contours.append(transformed)
                
                # Save the contours to the list of all contours with marker_key and index
                all_contours.append((marker_key, index, new_transformed_contours))
                
                # Log success and print on same line
                processing_time = time.time() - start_time
                log_writer.writerow([filename, "Detected", f"{marker_key}_{index}", "Segmented", f"{processing_time:.2f}", 
                                     upper_left.tolist(), upper_right.tolist(), lower_left.tolist(), lower_right.tolist()])
                sys.stdout.write(f" Successful detection and segmentation in {processing_time:.1f} seconds\n")
                sys.stdout.flush()
            else:
                # Log failure due to no markers found and print on same line
                processing_time = time.time() - start_time
                log_writer.writerow([filename, "No markers found", "", "", f"{processing_time:.2f}", "", "", "", ""])
                sys.stdout.write(f" Failed detection in {processing_time:.1f} seconds\n")
                sys.stdout.flush()
        except Exception as e:
            # Log failure due to an exception and print on same line
            processing_time = time.time() - start_time
            log_writer.writerow([filename, "Failed", "", str(e), f"{processing_time:.2f}", "", "", "", ""])
            sys.stdout.write(f" Failed detection: {str(e)} in {processing_time:.1f} seconds\n")
            sys.stdout.flush()
        
        #break  #! Stop after processing the first image: comment out for full batch processing

# Save all contours to a single CSV file
all_contours_csv_path = os.path.join(output_folder, f"all_contours_{current_datetime}.csv")
save_all_contours_as_csv(all_contours, all_contours_csv_path)
