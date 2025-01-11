import os
import re
import numpy as np
import SimpleITK as sitk
import cv2

import matplotlib.pyplot as plt
from scipy import ndimage

from skimage import measure
from skimage._shared.filters import gaussian
from skimage.filters import threshold_otsu

import pyvista as pv  # pip install pyvista


def load_images_from_directory(jpg_filenames):
    """
    Load JPG images from a 'files' subdirectory, convert each to grayscale,
    and stack them into a 3D numpy volume (shape: [Z, Y, X]).
    """
    # current_file_directory = os.path.dirname(os.path.abspath(__file__))
    # directory_files_path = os.path.join(current_file_directory, "files")
    #
    # # Gather JPG filenames
    # # Sort by filenames (modify key as needed)
    # jpg_filenames = sorted([
    #     os.path.join(directory_files_path, f)
    #     for f in os.listdir(directory_files_path)
    #     if f.lower().endswith('.jpg')
    # ], key=lambda x: int(os.path.basename(x).split("_")[0]))

    # for j in jpg_filenames:
    #     print(j)

    # List to hold each processed 2D slice
    slices_3d = []

    for i, j in enumerate(jpg_filenames[4:len(jpg_filenames)]):
        # Load and process an image
        #sharp_mask, visualization = detect_sharp_regions(j)

        # Display or save the results
        #cv2.imwrite(f"{i}.jpg", visualization)

        # 1. Read and convert to grayscale
        image = cv2.imread(j)
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1) Determine the center and radius
        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2
        radius = max(center_x, center_y)  # for safety margin

        # 2) Create a blurred version
        #    - The kernel size and sigma (20) can be tuned
        blurred = cv2.GaussianBlur(image, (7, 7), 50)

        # 3) Build a distance map from the center (Euclidean distance)
        #    dist_map[y, x] = distance of pixel (x, y) from center
        y_indices, x_indices = np.indices((h, w))
        dist_map = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)

        # 4) Normalize the distance to [0, 1] where
        #    dist = 0 => center => alpha=0 => 0% blurred, 100% original
        #    dist >= radius => edges => alpha=1 => 100% blurred
        alpha = dist_map / radius
        alpha = np.clip(alpha, 0, 1)  # Ensure we don't exceed [0,1]

        # If the image is color, we need 3 channels of alpha.
        # If the image is grayscale, you can skip this expansion.
        if len(image.shape) == 3 and image.shape[2] == 3:
            alpha_3d = cv2.merge([alpha, alpha, alpha])
            # Blend: final = alpha * blurred + (1 - alpha)* original
            # Make sure everything is float to avoid clip/round errors
            blurred_float = blurred.astype(np.float32)
            image_float = image.astype(np.float32)
            final_float = alpha_3d * blurred_float + (1 - alpha_3d) * image_float

            # Convert back to uint8
            final = final_float.astype(np.uint8)

        else:
            # For grayscale images
            blurred_float = blurred.astype(np.float32)
            image_float = image.astype(np.float32)
            final_float = alpha * blurred_float + (1 - alpha) * image_float
            final = final_float.astype(np.uint8)

        gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)

        # 2. Compute Laplacian
        #    - ksize=3 (or 5) is the kernel size, can be tuned
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=5)
        # Take absolute value to get magnitude
        laplacian_abs = np.absolute(laplacian)

        # 3. Normalize the result (0 to 255 range) for easier thresholding
        laplacian_norm = cv2.normalize(laplacian_abs, None, 0, 255, cv2.NORM_MINMAX)
        laplacian_norm = np.uint8(laplacian_norm)

        # 4. Threshold to create a binary mask
        #    - The threshold value (here 30) is empirical; adjust to your image
        _, focus_mask = cv2.threshold(laplacian_norm, 30, 255, cv2.THRESH_BINARY)

        # 5. (Optional) Morphological operations to clean up noise
        #    - You can open/close the mask to remove small bright/dark spots
        kernel = np.ones((2, 2), np.uint8)
        focus_mask_clean = cv2.morphologyEx(focus_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        focus_mask_clean = cv2.morphologyEx(focus_mask_clean, cv2.MORPH_CLOSE, kernel, iterations=5)

        # 6. Create output (isolate in-focus regions)
        #    - We'll create a 3-channel mask so we can multiply it with the color image
        focus_mask_3d = cv2.merge([focus_mask_clean, focus_mask_clean, focus_mask_clean])
        in_focus_only = cv2.bitwise_and(final, focus_mask_3d)

        # Invert the binary mask
        #focus_mask_clean_inverted = cv2.bitwise_not(focus_mask_clean)

        # Append to list (as a 2D array)
        slices_3d.append(focus_mask_clean)
        #slices_3d.append(in_focus_only)

        # #Show results
        # cv2.imshow("Original Image", image)
        # cv2.imshow("Focus Mask", focus_mask_clean)
        # cv2.imshow("In-Focus Regions", in_focus_only)
        # cv2.imshow("Blurred result", final)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


    # Stack along a new z-axis => shape: (num_slices, height, width)
    volume = np.stack(slices_3d, axis=0)

    return volume


def create_mesh(volume, threshold, spacing):
    """
    Extract a surface mesh from the 3D volume using marching cubes.
    """
    # skimage.measure.marching_cubes (or marching_cubes_lewiner)
    verts, faces, normals, values = measure.marching_cubes(
        volume,
        level=threshold,
        spacing=spacing
    )
    return verts, faces, normals, values


def analyze_run(jpg_filenames):
    # 1) Load the processed 3D volume (in-focus only)
    volume = load_images_from_directory(jpg_filenames)

    # 2) Compute Otsu's threshold from the entire volume
    thresh = threshold_otsu(volume.ravel())
    #print(f"Otsu's threshold = {thresh}")

    # 3) Create mesh with marching_cubes
    #    Adjust spacing as needed (dz, dy, dx)
    verts, faces, normals, values = create_mesh(volume, thresh, spacing=(15,1/3,1/3))

    # 4) Convert faces to the format PyVista expects
    faces_expanded = np.column_stack([np.full(len(faces), 3, dtype=np.int32), faces]).ravel()

    # 5) Create PyVista mesh
    mesh = pv.PolyData(verts, faces_expanded)

    # Suppose 'volume' is a 3D binary array
    voxel_count = np.count_nonzero(volume)

    # If each voxel's physical size is dx × dy × dz:
    dx, dy, dz = (1/3, 1/3, 15)  # example
    voxel_volume = dx * dy * dz

    # Total volume in physical units:
    volume_estimate = voxel_count * voxel_volume
    #print(f"Volume = {volume_estimate} (same units^3)")
    print(f"MESH VOLUME = {mesh.volume}")

    # 6) Visualize in PyVista
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='blue', opacity=0.6, show_edges=True)
    plotter.add_axes()
    plotter.show_grid()
    plotter.show()

    return volume_estimate

def main():
    volume_estimates = []

    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    directory_files_path = os.path.join(current_file_directory, "all_data_first", "1_F-75")

    # ASSUMPTION: all folders for the current dish will have the same amount of runs (images) in them
    amount_of_runs = len([name for name in os.listdir(directory_files_path)])

    folder_names = sorted(
        [name for name in os.listdir(os.path.join(current_file_directory, "all_data_first"))],
        key=lambda x: int(os.path.basename(x).split("_")[0]))

    #print(folder_names)

    for run in range(1):
        run_file_names = []
        for folder_name in folder_names:
            run_file_names.append(
                os.path.join(
                    current_file_directory, "all_data_first", folder_name,
                    os.listdir(
                        os.path.join(
                            current_file_directory, "all_data_first", folder_name
                        )
                    )[run])
            )
        #print(run_file_names)
        current_run_volume = analyze_run(run_file_names)
        #print(f"RUN: {run+1}, VOLUME: {current_run_volume}")
        volume_estimates.append(current_run_volume)

if __name__ == "__main__":
    main()


    #load_images_from_directory()

