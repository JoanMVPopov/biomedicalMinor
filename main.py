import os
import re
import numpy as np
import SimpleITK as sitk
import cv2

import matplotlib.pyplot as plt
import vtk
from scipy import ndimage
from scipy.spatial.distance import pdist, squareform

from skimage import measure
from skimage._shared.filters import gaussian
from skimage.filters import threshold_otsu

import pyvista as pv  # pip install pyvista
from skimage.morphology import binary_closing, ball


def get_feret_diameter(contour):
    """
    Calculates the maximum distance between any two points on the contour (Feret diameter).

    Parameters:
    contour: numpy array of contour points from cv2.findContours

    Returns:
    max_distance: The Feret diameter
    (point1, point2): The two points that are furthest apart
    """
    # Reshape contour to 2D array of points
    points = contour.reshape(-1, 2)

    # Calculate pairwise distances between all points
    distances = pdist(points)
    distance_matrix = squareform(distances)

    # Find the maximum distance and its indices
    max_idx = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)
    max_distance = distance_matrix[max_idx] / 3

    # Get the actual points
    point1 = tuple(points[max_idx[0]])
    point2 = tuple(points[max_idx[1]])

    return max_distance, (point1, point2)


def get_equivalent_diameter(contour):
    """
    Calculates the diameter of a circle with the same area as the contour.

    Parameters:
    contour: numpy array of contour points from cv2.findContours

    Returns:
    diameter: The equivalent circular diameter
    center: The centroid of the contour
    """
    # Calculate area
    area = cv2.contourArea(contour)

    # Calculate equivalent diameter
    diameter = np.sqrt(4 * area / np.pi) / 3

    # Calculate centroid
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        center = (cx, cy)
    else:
        center = None

    return diameter, center


def get_minimum_enclosing_circle(contour):
    """
    Finds the minimum enclosing circle of the contour.

    Parameters:
    contour: numpy array of contour points from cv2.findContours

    Returns:
    diameter: The diameter of the minimum enclosing circle
    (center, radius): The center point and radius of the circle
    """
    center, radius = cv2.minEnclosingCircle(contour)
    diameter = 2 * radius / 3
    return diameter, (center, radius)


def visualize_diameters(image, contour, area_ratio):
    """
    Visualizes different diameter measurements on the image.

    Parameters:
    image: Original image
    contour: numpy array of contour points from cv2.findContours

    Returns:
    annotated_image: Image with diameter measurements drawn
    """
    # Create a copy for visualization
    vis_image = image.copy()
    if len(vis_image.shape) == 2:
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)

    # Draw contour
    cv2.drawContours(vis_image, [contour], -1, (0, 255, 0), 2)

    # Draw Feret diameter
    feret_diameter, (point1, point2) = get_feret_diameter(contour)
    cv2.line(vis_image, point1, point2, (255, 0, 0), 2)
    cv2.putText(vis_image, f'Feret: {feret_diameter:.1f}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Draw equivalent circular diameter
    equiv_diameter, center = get_equivalent_diameter(contour)
    if center is not None:
        radius = int(equiv_diameter / 2)
        cv2.circle(vis_image, center, radius, (0, 0, 255), 2)
        cv2.putText(vis_image, f'Equivalent: {equiv_diameter:.1f}',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Draw minimum enclosing circle
    min_diameter, (min_center, min_radius) = get_minimum_enclosing_circle(contour)
    center_point = (int(min_center[0]), int(min_center[1]))
    cv2.circle(vis_image, center_point, int(min_radius), (255, 255, 0), 2)
    cv2.putText(vis_image, f'Min enclosing: {min_diameter:.1f}',
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Add area ratio text
    cv2.putText(vis_image, f'Area ratio: {area_ratio:.1f}%',
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return vis_image


# Example usage in your existing code:
def analyze_contour_in_image(image):
    """
    Analyzes contour diameters in an image.

    Parameters:
    image: Input image (grayscale or color)

    Returns:
    Dictionary containing different diameter measurements
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Find contours
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate different diameters
    feret_diameter, feret_points = get_feret_diameter(largest_contour)
    equiv_diameter, center = get_equivalent_diameter(largest_contour)
    min_diameter, (min_center, min_radius) = get_minimum_enclosing_circle(largest_contour)

    # Calculate the area ratio
    contour_area = cv2.contourArea(largest_contour)
    circle_area = np.pi * (min_radius ** 2)
    area_ratio = (contour_area / circle_area) * 100  # Convert to percentage

    # Visualize
    annotated_image = visualize_diameters(image, largest_contour, area_ratio)


    return {
        'feret_diameter': feret_diameter,
        'equivalent_diameter': equiv_diameter,
        'minimum_enclosing_diameter': min_diameter,
        'area_ratio': area_ratio,
        'annotated_image': annotated_image
    }


def load_images_from_directory(jpg_filenames):
    """
    Load JPG images from a 'files' subdirectory, convert each to grayscale,
    and stack them into a 3D numpy volume (shape: [Z, Y, X]).
    """
    # List to hold each processed 2D slice
    slices_3d = []

    # first few images usually don't have cytoplasm in focus
    # maybe we can skip them??
    #for i, j in enumerate(jpg_filenames[4:len(jpg_filenames)]):

    ############
    # Diameter is calculated in a naive manner
    # Only take F +/-15 and F0 since those seem to provide best contrast between zygote border and bg
    diameter = 0.0
    area_ratio = 0.0

    for i, j in enumerate(jpg_filenames):
        image = cv2.imread(j)
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # final = blur_region_outside_of_radius(image)
        # final = color_region_outside_of_radius(image)


        final = gradual_color_region_outside_radius(image)

        #final = image

        gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)

        # Otsu's threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # compute Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=5)
        # Take absolute value to get magnitude
        laplacian_abs = np.absolute(laplacian)

        laplacian_norm = cv2.normalize(laplacian_abs, None, 0, 255, cv2.NORM_MINMAX)
        laplacian_norm = np.uint8(laplacian_norm)

        # 4. Threshold to create a binary mask
        _, focus_mask = cv2.threshold(laplacian_norm, 80, 255, cv2.THRESH_BINARY)

        # First, let's apply morphological operations to connect nearby points
        # Create a kernel for dilation - adjust size based on point spacing
        kernel = np.ones((5, 5), np.uint8)

        # Dilate the image to connect nearby points
        dilated = cv2.dilate(focus_mask, kernel, iterations=3)

        # Apply closing to fill small holes
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

        # Find the contour of the connected region
        contours, _ = cv2.findContours(closed,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Create an empty image for the outline
        outline_image = np.zeros_like(gray)


        # Draw only the largest contour
        # Draw and fill contours
        if contours:
            #largest_contour = max(contours, key=cv2.contourArea)

            # Close first and last contour, needed for proper 3d mesh
            if i == 0 or i == len(jpg_filenames) - 1:
                cv2.drawContours(outline_image, contours, -1, 128, -1)  # -1 thickness means fill

            else:
                # First fill the inside area with white (255)
                cv2.drawContours(outline_image, contours, -1, 255, -1)  # -1 thickness means fill

                # Create a copy of the filled contour
                contour_only = np.zeros_like(gray)
                # Draw the contour line itself in gray (128)
                cv2.drawContours(contour_only, contours, -1, 128, 2)

                # Combine: where contour is drawn (128), use that value; otherwise keep the fill values
                outline_image = np.where(contour_only == 128, 128, outline_image)

            ############################################################
            ## DIAMETER APPROXIMATION
            ###########################################################

            # Get diameter measurements
            results = analyze_contour_in_image(outline_image)

            # print(results)

            if results:
                # print(f"Slice {i} measurements:")
                # print(f"Feret diameter: {results['feret_diameter']:.2f}")
                # print(f"Equivalent diameter: {results['equivalent_diameter']:.2f}")
                # print(f"Minimum enclosing diameter: {results['minimum_enclosing_diameter']:.2f}")

                # Take F+/-15 and F0 as diameter for embryo
                if i >= 4 and i <= 6:
                    if results['area_ratio'] > area_ratio:
                        diameter = results['feret_diameter']

                # Display annotated image
                cv2.imshow(f"Diameter measurements - Slice {i}", results['annotated_image'])
                cv2.waitKey(0)

            ##############################################################

        # morphological operations to clean up noise
        # kernel = np.ones((2, 2), np.uint8)
        # focus_mask_clean = cv2.morphologyEx(focus_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        # focus_mask_clean = cv2.morphologyEx(focus_mask_clean, cv2.MORPH_CLOSE, kernel, iterations=5)

        # 6. Create output (isolate in-focus regions)
        focus_mask_3d = cv2.merge([closed, closed, closed])
        in_focus_only = cv2.bitwise_and(final, focus_mask_3d)

        # Invert the binary mask
        #focus_mask_clean_inverted = cv2.bitwise_not(focus_mask_clean)



        # Append to list (as a 2D array)
        # slices_3d.append(focus_mask_clean)
        # slices_3d.append(gray)
        slices_3d.append(outline_image)



        ###############################
        # VISUALIZATION
        # purely experimental,
        # only run this with A SMALL SUBSET
        # of runs
        ##############################

        # #Show results
        # cv2.imshow("Thresh", thresh)
        # cv2.imshow("Original Image", image)

        image_name = j.split("all_data_first")[-1]

        cv2.imshow(f"Focus Mask {image_name}", closed)
        cv2.imshow(f"In-Focus Regions {image_name}", in_focus_only)
        cv2.imshow(f"OUTLINE {image_name}", outline_image)

        # cv2.imshow("Blurred result", final)
        cv2.imshow(f"GRAY {image_name}", gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # Stack along a new z-axis => shape: (num_slices, height, width)
    volume = np.stack(slices_3d, axis=0)

    return volume, diameter


def gradual_color_region_outside_radius(image):
    # 1) Determine the center and radius
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    radius = max(center_x, center_y) + 200  # for safety margin

    # 2) Calculate the average color
    if len(image.shape) == 3 and image.shape[2] == 3:
        avg_color = np.mean(image, axis=(0, 1))
    else:
        avg_color = np.mean(image)

    # 3) Build a distance map from the center
    y_indices, x_indices = np.indices((h, w))
    dist_map = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)

    # 4) Create alpha channel for gradual transition
    alpha = dist_map / radius
    alpha = np.clip(alpha, 0, 1)

    # 5) Apply gradual transition to average color
    if len(image.shape) == 3 and image.shape[2] == 3:
        # For color images
        alpha_3d = cv2.merge([alpha, alpha, alpha])

        # Convert to float for calculations
        image_float = image.astype(np.float32)
        avg_color_image = np.full_like(image_float, avg_color)

        # Blend original image with average color image
        final_float = (1 - alpha_3d) * image_float + alpha_3d * avg_color_image

        # Convert back to uint8
        final = final_float.astype(np.uint8)
    else:
        # For grayscale images
        image_float = image.astype(np.float32)
        avg_color_image = np.full_like(image_float, avg_color)

        final_float = (1 - alpha) * image_float + alpha * avg_color_image
        final = final_float.astype(np.uint8)

    return final

def color_region_outside_of_radius(image):
    # 1) Determine the center and radius
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    radius = max(center_x, center_y)  # for safety margin

    # 2) Build a distance map from the center (Euclidean distance)
    y_indices, x_indices = np.indices((h, w))
    dist_map = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)

    # 3) Create a mask where:
    #    True (1) = pixels to make black
    #    False (0) = pixels to keep original
    mask = dist_map / radius >= 1  # Everything at or beyond radius distance becomes black

    # 2) Calculate the average color of the image
    if len(image.shape) == 3 and image.shape[2] == 3:
        # For color images - calculate mean for each channel
        avg_color = np.mean(image, axis=(0, 1))
    else:
        # For grayscale images
        avg_color = np.mean(image)

    # If the image is color, handle 3 channels
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Make a copy to avoid modifying original
        result = image.copy()
        # Set pixels where mask is True to black ([0,0,0])
        #result[mask] = [0, 0, 0]
        result[mask] = avg_color

        return result
    else:
        # For grayscale images
        result = image.copy()
        # Set pixels where mask is True to black (0)
        #result[mask] = 0
        result[mask] = avg_color

        return result

def blur_region_outside_of_radius(image):
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

        return final

    else:
        # For grayscale images
        blurred_float = blurred.astype(np.float32)
        image_float = image.astype(np.float32)
        final_float = alpha * blurred_float + (1 - alpha) * image_float
        final = final_float.astype(np.uint8)

        return final


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

def interpolate_slices(volume, num_intermediate_slices):
    """
    Interpolates between adjacent slices of a 3D volume.

    Parameters:
    - volume (np.ndarray): The input 3D volume (Z, Y, X).
    - num_intermediate_slices (int): The number of intermediate slices to add between each pair of slices.

    Returns:
    - interpolated_volume (np.ndarray): The interpolated 3D volume.
    """
    z, y, x = volume.shape
    new_volume = []

    for i in range(z - 1):
        # Add the current slice
        new_volume.append(volume[i])

        # Linearly interpolate intermediate slices
        for j in range(1, num_intermediate_slices + 1):
            alpha = j / (num_intermediate_slices + 1)  # Interpolation factor
            #print(alpha)
            interpolated_slice = (1 - alpha) * volume[i] + alpha * volume[i + 1]
            #interpolated_slice = np.where(interpolated_slice < 255/2, 0.0, 255.0)
            #print(np.unique(interpolated_slice))
            new_volume.append(interpolated_slice)

    # Add the last slice
    new_volume.append(volume[-1])

    return np.stack(new_volume, axis=0)

# Visualization Function
def visualize_slices(original_volume, interpolated_volume, num_intermediate_slices, slice_idx):
    """
    Visualizes a specific slice from the original and interpolated volumes.

    Parameters:
    - original_volume (np.ndarray): The original 3D volume.
    - interpolated_volume (np.ndarray): The interpolated 3D volume.
    - slice_idx (int): Index of the original slice to visualize.
    """

    current_idx = slice_idx
    end_idx = slice_idx + (num_intermediate_slices + 1)*2

    while True:
        # Display the current slice
        slice_to_show = interpolated_volume[current_idx].astype(np.uint8)
        cv2.imshow(f"Slice {current_idx}", slice_to_show)

        # Wait for user input
        key = cv2.waitKey(0)

        if key == ord('q'):  # Quit on 'q'
            break
        elif key == ord('n') and current_idx + 1 <= end_idx:  # Next slice on 'n'
            cv2.destroyWindow(f"Slice {current_idx}")
            current_idx += 1
        elif key == ord('p') and current_idx > slice_idx:  # Previous slice on 'p'
            cv2.destroyWindow(f"Slice {current_idx}")
            current_idx -= 1

    # Clean up
    cv2.destroyAllWindows()

    #Plot the interpolated slice
    # plt.subplot(1, 2, 2)
    # plt.imshow(interpolated_volume[interpolated_idx], cmap='gray')
    # plt.title(f"Interpolated Slice {interpolated_idx}")
    # plt.axis('off')
    #
    # plt.tight_layout()
    # plt.show()


def analyze_run(jpg_filenames):
    # 1) Load the processed 3D volume (in-focus only)
    volume_resampled, diameter = load_images_from_directory(jpg_filenames)

    # Interpolation parameters
    #num_intermediate_slices = 1  # Number of intermediate slices between each pair of original slices

    # Interpolate the volume
    #interpolated_volume = interpolate_slices(volume_resampled, num_intermediate_slices)

    #############
    # SLICE VISUALIZATION
    # This visualizes a few slices, purely experimental, feel free to change
    # Only run this with a small subset
    #############

    #original_slice_idx = 0
    #visualize_slices(volume_resampled, interpolated_volume,  num_intermediate_slices, original_slice_idx)

    # # Check shapes
    # print(f"Original volume shape: {volume_resampled.shape}")
    # print(f"Interpolated volume shape: {interpolated_volume.shape}")

    # Now 'volume_resampled' is your new 3D array with finer sampling along Z.
    #volume_resampled = binary_closing(volume_resampled, ball(4))

    # 3) Compute Otsu's threshold (already done in your code)

    # Maybe unnecessary
    thresh = threshold_otsu(volume_resampled.ravel())

    # 4) Create mesh with marching_cubes
    verts, faces, normals, values = create_mesh(volume_resampled, 128, spacing=(15.0, 1/3, 1/3))

    # 5) Convert faces, create PyVista mesh, etc. (same as before)
    faces_expanded = np.column_stack([np.full(len(faces), 3, dtype=np.int32), faces]).ravel()
    mesh = pv.PolyData(verts, faces_expanded)

    # # Optional cleanup / smoothing
    # mesh = mesh.smooth_taubin(n_iter=100)
    # mesh = mesh.fill_holes(hole_size=1000)
    #
    # 6) Compute volume from the *resampled* binary array
    voxel_count = np.count_nonzero(volume_resampled)
    dx, dy, dz = (1/3, 1/3, 15.0)
    voxel_volume = dx * dy * dz
    volume_estimate = voxel_count * voxel_volume

    #################
    ## VISUALIZATION
    # This creates a very cool looking 3d plot
    # Hopefully, you should see the outline of the embryo
    # However, a lot of holes are present so it looks kinda off...
    # ONLY RUN WITH A SMALL SUBSET of runs selected
    ###############

    #7) Visualize in PyVista
    # plotter = pv.Plotter()
    # plotter.add_mesh(mesh, color='blue', opacity=1, show_edges=True)
    # plotter.add_axes()
    # plotter.show_grid()
    # plotter.show()

    sphere_volume_estimate = (4/3) * np.pi * ((diameter/2)**3)

    return volume_estimate/(10**4), mesh.volume/(10**4), sphere_volume_estimate/(10**4)


def main():
    volume_estimates_numpy = []
    volume_estimates_pyvista_mesh = []
    volume_estimates_sphere = []

    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    directory_files_path = os.path.join(current_file_directory, "all_data_first", "1_F-75")

    # ASSUMPTION: all folders for the current dish will have the same amount of runs (images) in them

    #########################################
    # Uncomment this if you want all the runs
    amount_of_runs = len([name for name in os.listdir(directory_files_path)])
    # amount_of_runs = 10

    runs = range(amount_of_runs-10, amount_of_runs)
    #runs = range(0, 10)
    ##########################################

    folder_names = sorted(
        [name for name in os.listdir(os.path.join(current_file_directory, "all_data_first")) if ".gitkeep" not in name],
        key=lambda x: int(os.path.basename(x).split("_")[0]))

    gitkeep_index = -1

    for run in runs:
        run_file_names = []
        for folder_name in folder_names:
            current_folder_run_file = os.path.join(
                    current_file_directory, "all_data_first", folder_name,
                    os.listdir(
                        os.path.join(
                            current_file_directory, "all_data_first", folder_name
                        )
                    )[run]
                )

            if ".gitkeep" not in current_folder_run_file:
                run_file_names.append(current_folder_run_file)

        # if run only traversed .gitkeep from each folder, len will be 0
        if len(run_file_names) == 0:
            gitkeep_index = run
            continue

        current_run_volume_numpy, current_run_volume_mesh, current_run_sphere_volume_estimate = analyze_run(run_file_names)
        print(f"RUN: {run+1}, NUMPY VOLUME: {current_run_volume_numpy} 10^4 µm")
        print(f"RUN: {run + 1}, MESH VOLUME: {current_run_volume_mesh} 10^4 µm")
        print(f"RUN: {run + 1}, SPHERE VOLUME: {current_run_sphere_volume_estimate} 10^4 µm")
        volume_estimates_numpy.append(current_run_volume_numpy)
        volume_estimates_pyvista_mesh.append(current_run_volume_mesh)
        volume_estimates_sphere.append(current_run_sphere_volume_estimate)

    runs_list = list(runs)
    if gitkeep_index != -1:
        runs_list.pop(gitkeep_index)

    # Create the plot
    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.plot(runs_list, volume_estimates_numpy, label="NUMPY Volume progression")  # Plot the line
    plt.plot(runs_list, volume_estimates_pyvista_mesh, label="PYVISTA MESH Volume progression")  # Plot the line
    plt.plot(runs_list, volume_estimates_sphere, label="SPHERE Volume progression")  # Plot the line

    # Add labels and title
    plt.xlabel("Runs")  # Label for X-axis
    plt.ylabel("Volume in 10^4 µm")  # Label for Y-axis
    plt.title("Volume progression")  # Title of the chart

    # Add grid and legend
    plt.grid(True)  # Add a grid to the background
    plt.legend(loc="upper right")  # Add a legend in the top-right corner

    plt.savefig("volume_estimates_progression.png")

if __name__ == "__main__":
    main()
