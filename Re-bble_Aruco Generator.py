import cv2
import numpy as np
import os
import csv
import svgwrite

# Define the dictionary and output directory
aruco_dict_XS = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_250)
aruco_dict_S = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
aruco_dict_M = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
aruco_dict_L = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
output_dir = 'aruco_markers'

# Create the output directories if they don't exist
os.makedirs(os.path.join(output_dir), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'extra small'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'small'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'medium'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'large'), exist_ok=True)

# CSV file paths
csv_file_path_XS = os.path.join(output_dir, 'extra small', 'markers_XS.csv')
csv_file_path_S = os.path.join(output_dir, 'small', 'markers_S.csv')
csv_file_path_M = os.path.join(output_dir, 'medium', 'markers_M.csv')
csv_file_path_L = os.path.join(output_dir, 'large', 'markers_L.csv')

def save_marker_as_svg(marker_image, file_path):
    size = marker_image.shape[0]
    dwg = svgwrite.Drawing(file_path, profile='tiny', size=(size, size))
    for y in range(size):
        for x in range(size):
            if marker_image[y, x] == 0:
                dwg.add(dwg.rect(insert=(x, y), size=(1, 1), fill='black'))
    dwg.save()

# Generate and save each marker, and write to CSV
with open(csv_file_path_XS, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['@File Path', 'Marker ID'])
    
    for marker_id in range(250):
        # Extra mall Size Markers (7x7_250)
        marker_image_XS = np.zeros((9, 9), dtype=np.uint8)
        marker_image_XS = cv2.aruco.generateImageMarker(aruco_dict_S, marker_id, 9, marker_image_XS, 1)
        file_path_XS = os.path.join(output_dir, 'extra small', f'marker_XS_{marker_id}.svg')
        save_marker_as_svg(marker_image_XS, file_path_XS)

        absolute_file_path = os.path.abspath(file_path_XS)
        formatted_id = f'RUBBLE XS_{marker_id:03}'
        csv_writer.writerow([absolute_file_path, formatted_id])

with open(csv_file_path_S, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['@File Path', 'Marker ID'])
    
    for marker_id in range(250):
        # Small Size Markers (4x4_250)
        marker_image_S = np.zeros((6, 6), dtype=np.uint8)
        marker_image_S = cv2.aruco.generateImageMarker(aruco_dict_S, marker_id, 6, marker_image_S, 1)
        file_path_S = os.path.join(output_dir, 'small', f'marker_S_{marker_id}.svg')
        save_marker_as_svg(marker_image_S, file_path_S)

        absolute_file_path = os.path.abspath(file_path_S)
        formatted_id = f'RUBBLE S_{marker_id:02}'
        csv_writer.writerow([absolute_file_path, formatted_id])

with open(csv_file_path_M, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['@File Path', 'Marker ID'])
    
    for marker_id in range(250):
        # Medium Size Markers (5x5_250)
        marker_image_M = np.zeros((7, 7), dtype=np.uint8)
        marker_image_M = cv2.aruco.generateImageMarker(aruco_dict_M, marker_id, 7, marker_image_M, 1)
        file_path_M = os.path.join(output_dir, 'medium', f'marker_M_{marker_id}.svg')
        save_marker_as_svg(marker_image_M, file_path_M)

        absolute_file_path = os.path.abspath(file_path_M)
        formatted_id = f'RUBBLE M_{marker_id:02}'
        csv_writer.writerow([absolute_file_path, formatted_id])

with open(csv_file_path_L, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['@File Path', 'Marker ID'])
    
    for marker_id in range(250):
        # Large Size Markers (6x6_250)
        marker_image_L = np.zeros((8, 8), dtype=np.uint8)
        marker_image_L = cv2.aruco.generateImageMarker(aruco_dict_L, marker_id, 8, marker_image_L, 1)
        
        file_path_L = os.path.join(output_dir, 'large', f'marker_L_{marker_id}.svg')
        save_marker_as_svg(marker_image_L, file_path_L)
        
        absolute_file_path = os.path.abspath(file_path_L)
        formatted_id = f'RUBBLE L_{marker_id:02}'
        csv_writer.writerow([absolute_file_path, formatted_id])
