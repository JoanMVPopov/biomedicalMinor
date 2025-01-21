import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pyvista as pv

from scipy.spatial.distance import pdist, squareform
from skimage import measure
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler


def get_parameter_options():
    return {
        'laplacian_k_size': [1,3,5,7,9],
        'laplacian_threshold': [40,50,60,70,80,90],
        'dilation_kernel_size': [(1,1), (2,2), (3,3), (4,4), (5,5)],
        'dilation_iterations': [1,2,3,4,5,6]
    }
    # return {
    #     'laplacian_k_size': [1, 3],
    #     'laplacian_threshold': [40],
    #     'dilation_kernel_size': [(1, 1), (2, 2)],
    #     'dilation_iterations': [1]
    # }


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


def visualize_diameters(image, contour, area_ratio, avg_distance_to_enclosing):
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
        # cv2.circle(vis_image, center, radius, (0, 0, 255), 2)
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

    # Add average distance to enclosing text
    cv2.putText(vis_image, f'Avg distance to enclosing: {avg_distance_to_enclosing:.1f}',
                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return vis_image


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

    avg_distance_to_enclosing = calc_contour_to_circle_perimeter_distance(largest_contour, min_center[0], min_center[1], min_radius)

    # Visualize
    annotated_image = visualize_diameters(image, largest_contour, area_ratio, avg_distance_to_enclosing)

    return {
        'feret_diameter': feret_diameter,
        'equivalent_diameter': equiv_diameter,
        'minimum_enclosing_diameter': min_diameter,
        'area_ratio': area_ratio,
        'avg_distance_to_enclosing': avg_distance_to_enclosing,
        'annotated_image': annotated_image
    }

def calc_contour_to_circle_perimeter_distance(contour, center_x, center_y, radius):
    """
    Calculate average distance from contour points to the perimeter of their enclosing circle.
    A smaller average means a tighter fit.

    Parameters:
    contour: numpy array of shape (N, 1, 2) containing contour points

    Returns:
    float: average distance
    numpy.ndarray: all distances
    tuple: circle parameters (center_x, center_y, radius)
    """
    # Convert contour to normal array of points
    contour_points = contour.reshape(-1, 2)

    # For each point, calculate distance to center
    distances_to_center = np.sqrt(
        (contour_points[:, 0] - center_x) ** 2 +
        (contour_points[:, 1] - center_y) ** 2
    )

    # The distance to circle perimeter is the difference between
    # the radius and the distance to center
    distances_to_perimeter = radius - distances_to_center

    # Sort distances in descending order and select the top 10
    top_distances = np.sort(distances_to_perimeter)[-10:][::-1]

    # Calculate average distance
    avg_distance = np.mean(top_distances)

    return avg_distance



def load_images_from_directory(jpg_filenames, params, show_focus_contours = False):
    """
    Load JPG images from a 'files' subdirectory, convert each to grayscale,
    and stack them into a 3D numpy volume (shape: [Z, Y, X]).
    """
    # List to hold each processed 2D slice
    slices_3d = []
    color_data = []

    diameter = 0.0
    adte_min = np.inf

    diameter_options = []
    adte_list = []
    ratio_list = []

    for i, j in enumerate(jpg_filenames):
        image = cv2.imread(j)

        final = gradual_color_region_outside_radius(image)

        gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)

        # Otsu's threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # compute Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=params['laplacian_k_size'])
        # Take absolute value to get magnitude
        laplacian_abs = np.absolute(laplacian)

        laplacian_norm = cv2.normalize(laplacian_abs, None, 0, 255, cv2.NORM_MINMAX)
        laplacian_norm = np.uint8(laplacian_norm)

        # 4. Threshold to create a binary mask
        _, focus_mask = cv2.threshold(laplacian_norm, params['laplacian_threshold'], 255, cv2.THRESH_BINARY)

        # First, let's apply morphological operations to connect nearby points
        # Create a kernel for dilation - adjust size based on point spacing
        kernel = np.ones(params['dilation_kernel_size'], np.uint8)

        # Dilate the image to connect nearby points
        dilated = cv2.dilate(focus_mask, kernel, iterations=params['dilation_iterations'])

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
            # Close first and last contour, needed for proper 3d mesh
            # if i == 0 or i == len(jpg_filenames) - 1:
            #     cv2.drawContours(outline_image, contours, -1, 128, -1)  # -1 thickness means fill

            # else:
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

            if results:
                # Take F-15, F0, F15, F30 into consideration for embryo diameter
                if i >= 4 and i <= 7:

                    adte_list.append((-1) * results['avg_distance_to_enclosing'])
                    ratio_list.append(results['area_ratio'])

                    diameter_options.append(
                        {
                            "adte": (-1) * results['avg_distance_to_enclosing'],
                            "ratio": results['area_ratio'],
                            "eq": results['equivalent_diameter']
                        }
                    )

                # Display annotated image
                if show_focus_contours:
                    cv2.imshow(f"Diameter measurements - Slice {i}", results['annotated_image'])
                    cv2.waitKey(0)

        # 6. Create output (isolate in-focus regions)
        # focus_mask_3d = cv2.merge([closed, closed, closed])
        # in_focus_only = cv2.bitwise_and(final, focus_mask_3d)

        focus_mask_3d = cv2.merge([outline_image, outline_image, outline_image])
        in_focus_only = cv2.bitwise_and(image, focus_mask_3d)

        slices_3d.append(outline_image)
        color_data.append(in_focus_only)

        if show_focus_contours:
            image_name = j.split("all_data_first")[-1]

            cv2.imshow(f"Original Image {image_name}", image)
            cv2.imshow(f"Focus Mask {image_name}", closed)
            cv2.imshow(f"In-Focus Regions {image_name}", in_focus_only)
            cv2.imshow(f"Outline {image_name}", outline_image)

            # cv2.imshow("Blurred result", final)
            cv2.imshow(f"Gray {image_name}", gray)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    numpy_array_for_minmax_analysis = np.column_stack((adte_list, ratio_list))

    scaler = MinMaxScaler()

    # apply MinMaxScaler to each column
    scaled_data = scaler.fit_transform(numpy_array_for_minmax_analysis)  # Scale each column to [0, 1]

    row_sums = scaled_data.sum(axis=1)

    for i, row in enumerate(row_sums):
        diameter_options[i]['final_score'] = row

    sorted_options = sorted(diameter_options, key=lambda x: x['final_score'], reverse=True)

    diameter = sorted_options[0]['eq']

    # Stack along a new z-axis => shape: (num_slices, height, width)
    volume = np.stack(slices_3d, axis=0)
    color_volume = np.stack(color_data, axis=0)

    return volume, color_volume, diameter


def gradual_color_region_outside_radius(image):
    # 1) Determine the center and radius
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    radius = max(center_x, center_y) + 100  # for safety margin

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


def create_colored_mesh(volume, color_volume, threshold, spacing):
    """
    Extract a surface mesh from the 3D volume and apply colors from the color volume.
    """
    # Extract surface using marching cubes
    verts, faces, normals, values = measure.marching_cubes(
        volume,
        level=threshold,
        spacing=spacing
    )

    # For each vertex, get the color from the color volume
    colors = []
    for vert in verts:
        # Convert vertex coordinates to indices in the color volume
        z, y, x = np.round(vert / spacing).astype(int)
        # Ensure indices are within bounds
        z = np.clip(z, 0, color_volume.shape[0] - 1)
        y = np.clip(y, 0, color_volume.shape[1] - 1)
        x = np.clip(x, 0, color_volume.shape[2] - 1)
        # Get color at this position and normalize to 0-1 range
        color = color_volume[z, y, x] / 255.0
        colors.append(color)

    return verts, faces, normals, np.array(colors)

def analyze_run(jpg_filenames, params, show_3d_model = False, show_focus_contours = False):
    # 1) Load the processed 3D volume (in-focus only)
    volume_resampled, color_volume, diameter = load_images_from_directory(jpg_filenames, params, show_focus_contours)

    mesh_volume = 0

    try:
        plotter = pv.Plotter()

        # Store meshes for each layer
        layer_meshes = []

        print("Creating separate meshes for each layer...")

        # Create separate mesh for each layer
        for z in range(volume_resampled.shape[0]):
            # Create a mask for this layer
            layer_mask = np.zeros_like(volume_resampled)
            layer_mask[z] = volume_resampled[z]

            try:
                # Create mesh for this layer
                verts, faces, normals, colors = create_colored_mesh(
                    layer_mask,
                    color_volume[z:z + 1],  # Take corresponding color slice
                    128,
                    spacing=(15.0, 1/3, -1/3)
                )

                # Create PyVista mesh for this layer
                faces_expanded = np.column_stack([np.full(len(faces), 3, dtype=np.int32), faces]).ravel()
                mesh = pv.PolyData(verts, faces_expanded)
                mesh.point_data['colors'] = colors[:, :3]

                layer_meshes.append(mesh)

            except Exception as e:
                print(f"Error creating mesh for layer {z}: {str(e)}")
                #layer_meshes.append(None)

        # Initialize list to track visibility status
        visibility_status = [True] * len(layer_meshes)

        # Add all meshes initially
        mesh_actors = []
        for i, mesh in enumerate(layer_meshes):
            if mesh is not None:
                actor = plotter.add_mesh(
                    mesh,
                    scalars='colors',
                    rgb=True,
                    name=f'layer_{i}'
                )
                mesh_actors.append(actor)
            else:
                mesh_actors.append(None)

        print("Calculating checkbox positions...")

        # Calculate checkbox positions
        n_layers = len(layer_meshes)
        checkbox_height = 0.05  # Height of each checkbox
        start_position = 0.95 - (n_layers * checkbox_height)  # Start from top with some margin

        # Add checkboxes for each layer
        for i in range(n_layers):
            if layer_meshes[i] is not None:
                try:
                    plotter.add_checkbox_button_widget(
                        lambda state, layer=i: callback(state, layer, visibility_status, mesh_actors, plotter),
                        value=True,  # Initially checked
                        position=(10, int(start_position * 800 + (i * checkbox_height * 800))),
                        # Convert to pixel coordinates
                        size=15,
                        border_size=2,
                        color_on='grey',
                        color_off='white',
                        background_color='black'
                    )
                except Exception as e:
                    print(e)

                # Add text label for each checkbox
                plotter.add_text(
                    f"Focal Plane {i}",
                    position=(40, int(start_position * 800 + (i * checkbox_height * 800))),
                    font_size=10,
                    color='black'
                )

        # Add title
        plotter.add_text(
            "Toggle Focal Planes",
            position=(10, 820),
            font_size=12,
            color='black'
        )

        plotter.add_axes()
        plotter.show_grid()

        print("Showing 3D model...")

        # Show the interactive window
        plotter.show()


    except Exception as e:
        with open("output.txt", "a") as file:
            file.write("Mesh could not be created due to an error, likely related to the given iso value (128). Nevertheless, mesh volume will be set to 0. Moving on...\n")
    finally:
        sphere_volume_estimate = (4 / 3) * np.pi * ((diameter / 2) ** 3)

        return mesh_volume / (10 ** 4), sphere_volume_estimate / (10 ** 4), diameter


def callback(state, layer_idx, visibility_status, mesh_actors, plotter):
    visibility_status[layer_idx] = state
    if mesh_actors[layer_idx] is not None:
        mesh_actors[layer_idx].SetVisibility(state)
    plotter.render()


def average_every_n(data, n=10):
    # Calculate how many complete groups of n we have
    length = len(data)
    groups = length // n
    # Truncate the data to fit complete groups
    truncated = data[:groups * n]
    # Reshape and calculate mean
    return np.mean(np.reshape(truncated, (-1, n)), axis=1)


def do_runs(runs_range, folder_names, current_file_directory, params, do_plots=False,
            show_3d_model=False, show_focus_contours=False):
    volume_estimates_pyvista_mesh = []
    volume_estimates_sphere = []

    diameter_list = []

    pairwise_abs_diff = []
    current_to_first_abs_diff = []

    gitkeep_index = -1

    for run in runs_range:
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

        print(f"RUN: {run+1}")

        # The current algorithm does not account for expansion (happens ~ at the start of day 5)
        # Just in case, set dilation to a higher number to connect components
        if run == 600:
            print("Switching to expansion mode, dilation set to 6")
            params['dilation_iterations'] = 6

        current_run_volume_mesh, current_run_sphere_volume_estimate, diameter = analyze_run(run_file_names, params,
                                                                                            show_3d_model, show_focus_contours)
        with open("output.txt", "a") as file:
            # file.write(f"RUN: {run + 1}, MESH VOLUME: {current_run_volume_mesh} 10^4 µm\n")
            # file.write(f"RUN: {run + 1}, SPHERE VOLUME: {current_run_sphere_volume_estimate} 10^4 µm\n")
            file.write(f"RUN: {run + 1}, DIAMETER: {diameter}\n")

        if len(diameter_list) > 0:
            diameter_penalty = 0
            pairwise_abs_diff_to_append = (abs(diameter - diameter_list[-1]))**2
            current_to_first_abs_diff_to_append = (abs(diameter - diameter_list[0]))**2

            if diameter < 100:
                diameter_penalty = abs(diameter - 100)**3
            elif diameter > 150:
                diameter_penalty = abs(diameter - 150)**3

            pairwise_abs_diff.append(pairwise_abs_diff_to_append + diameter_penalty)
            current_to_first_abs_diff.append(current_to_first_abs_diff_to_append + diameter_penalty)

        diameter_list.append(diameter)
        volume_estimates_pyvista_mesh.append(current_run_volume_mesh)
        volume_estimates_sphere.append(current_run_sphere_volume_estimate)

    runs_list = list(runs_range)
    if gitkeep_index != -1:
        runs_list.pop(gitkeep_index)

    if do_plots:
        # Volume progression
        progression_plots(runs_list, volume_estimates_sphere, "SPHERE Volume progression",
                          "Runs", "Volume in 10^4 µm", "Volume progression",
                          f"volume_progression_lks_{params['laplacian_k_size']}_lt_{params['laplacian_threshold']}_dks_{params['dilation_kernel_size']}_di_{params['dilation_iterations']}.png")

        # Calculate averages
        averaged_pyvista = average_every_n(volume_estimates_pyvista_mesh, 5)
        averaged_sphere = average_every_n(volume_estimates_sphere, 5)

        # Create x-axis values (each point represents 5 runs)
        averaged_runs = np.arange(len(averaged_sphere)) * 5

        # Volume 5-run average progression
        progression_plots(averaged_runs, averaged_sphere, "SPHERE Volume progression",
                          "Runs (averaged every 5)", "Volume in 10^4 µm", "Volume progression (5-run averages)",
                          f"volume_progression_averaged_lks_{params['laplacian_k_size']}_lt_{params['laplacian_threshold']}_dks_{params['dilation_kernel_size']}_di_{params['dilation_iterations']}.png")

        # Diameter progression
        progression_plots(runs_list, diameter_list, "Diameter progression",
                          "Runs", "Diameter in µm", "Diameter progression",
                          f"diameter_progression_lks_{params['laplacian_k_size']}_lt_{params['laplacian_threshold']}_dks_{params['dilation_kernel_size']}_di_{params['dilation_iterations']}.png")

        # Calculate averages
        averaged_diameter = average_every_n(np.array(diameter_list), 5)

        # Create x-axis values (each point represents 10 runs)
        averaged_runs = np.arange(len(averaged_diameter)) * 5

        # Diameter 5-run average progression
        progression_plots(averaged_runs, averaged_diameter, "Diameter progression",
                          "Runs (averaged every 5)", "Diameter in µm", "Diameter progression (5-run averages)",
                          f"diameter_progression_averaged_lks_{params['laplacian_k_size']}_lt_{params['laplacian_threshold']}_dks_{params['dilation_kernel_size']}_di_{params['dilation_iterations']}.png")

    return pairwise_abs_diff, current_to_first_abs_diff, diameter_list


def progression_plots(runs_x_values, progression_y_values, plot_label, x_label, y_label, title, image_name):
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(runs_x_values, progression_y_values, label=plot_label)

    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # Add grid and legend
    plt.grid(True)
    plt.legend(loc="upper right")

    path_dir = os.getcwd()
    save_path = os.path.join(path_dir, "output-images", image_name)

    plt.savefig(save_path)
    plt.close()


def evaluate_parameter_combinations(runs, folder_names, current_file_directory, do_plots=False,
                                    show_3d_model=False, show_focus_contours=False):

    parameters = get_parameter_options()

    grid = ParameterGrid(parameters)

    options = []
    pairwise_total_sum_list = []
    current_to_first_total_sum_list = []

    # Iterate through all possible combinations (or hardcoded ones)
    for params in grid:
        print("------------------------\n")
        print(f"CURRENT PARAMS: {params}\n")
        try:
            with open("output.txt", "a") as file:   
                file.write("------------------------\n")
                file.write(f"CURRENT PARAMS: {params}\n")

            pairwise_abs_diff, current_to_first_abs_diff, _ = do_runs(runs, folder_names,
                                                                current_file_directory, params,
                                                                      do_plots, show_3d_model,
                                                                      show_focus_contours)

            # First one records distance between the current run diameter and last run's diameter
            # Should be as little as possible because we need consistency for days 0 through 3/4
            # e.g. 100, 101, 99, 100, 102 is ok; 100, 120, 94, 111 is not
            # times -1 will invert minmax scaling to maxmin, closer to 0 is better
            pairwise_total_sum = -1 * np.sum(pairwise_abs_diff)
            # This records distance between current and first distance in the list
            # That way, if we have consistency but we are consistently very far from the initial diameter
            # Then something went wrong, so we should penalize the result
            # 100, 130, 131, 129, 125 is not ok
            current_to_first_total_sum = -1 * np.sum(current_to_first_abs_diff)

            with open("output.txt", "a") as file:
                file.write(f"PAIRWISE TOTAL SUM (closer to 0 is better): {pairwise_total_sum}\n")
                file.write(f"CURRENT TO FIRST TOTAL SUM (closer to 0 is better): {current_to_first_total_sum}\n")

            pairwise_total_sum_list.append(pairwise_total_sum)
            current_to_first_total_sum_list.append(current_to_first_total_sum)

            options.append(
                {
                    'pairwise_total_sum': pairwise_total_sum,
                    'current_to_first_total_sum': current_to_first_total_sum,
                    'params': params
                }
            )
        except Exception as e:
            with open("output.txt", "a") as file:
                file.write(f"ERROR processing params {params}: {str(e)}\n")

    numpy_array_for_minmax_analysis = np.column_stack((pairwise_total_sum_list, current_to_first_total_sum_list))

    scaler = MinMaxScaler()

    # apply MinMaxScaler to each column
    scaled_data = scaler.fit_transform(numpy_array_for_minmax_analysis)  # Scale each column to [0, 1]
    # multiply by 100, looks better
    scaled_data_100 = scaled_data * 100

    # sum the results for each row
    # it's essentially like a leaderboard (places decided by sum of the two features per each row)
    # this way we can take both results into account
    # e.g.:
    # f1 f2
    # 100 25 -> 125 (2nd place)
    # 50 60 -> 110 (3rd place)
    # 70 80 -> 150 (1st place)
    row_sums = scaled_data_100.sum(axis=1)

    for i, row in enumerate(row_sums):
        options[i]['final_score'] = row

    sorted_options = sorted(options, key=lambda x: x['final_score'], reverse=True)

    with open("output.txt", "a") as file:
        # Output top 10 best
        for i in range(10):
            file.write(f"CHOICE {i+1}\nSCORE: {sorted_options[i]['final_score']}:\n"
                f"SCORE: {sorted_options[i]['final_score']}\nPARAMS: {sorted_options[i]['params']}\n")

    return sorted_options


if __name__ == "__main__":

    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    directory_files_path = os.path.join(current_file_directory, "all_data_first", "1_F-75")

    # ASSUMPTION: all folders for the current dish will have the same amount of runs (images) in them
    amount_of_runs = len([name for name in os.listdir(directory_files_path)])

    folder_names = sorted(
        [name for name in os.listdir(os.path.join(current_file_directory, "all_data_first")) if ".gitkeep" not in name],
        key=lambda x: int(os.path.basename(x).split("_")[0]))

    ##################################################################################################
    ## Use this if you want to iterate over possible combinations
    ## And find the best ones according to the defined metrics

    ## Only calculate the first 50 runs for each combination
    ## Could increase, but it seems to be enough
    # runs = range(0, 50)
    #
    # sorted_options = evaluate_parameter_combinations(runs, folder_names, current_file_directory,
    #                                                 do_plots=False, show_3d_model=False, show_focus_contours=False)
    #
    # runs = range(0, amount_of_runs)
    #
    # Get plots for top 10 combinations
    # for i in range(10):
    #     do_runs(runs, folder_names, current_file_directory, sorted_options[i]['params'],
    #     do_plots=True, show_3d_model=False, show_focus_contours=False)

    ###################################################################################################

    # Combinations derived from our tests, hardcoded
    grid = [
        {'dilation_iterations': 2, 'dilation_kernel_size': (4, 4), 'laplacian_k_size': 9, 'laplacian_threshold': 60}
        # 2 Other possible combinations below
        # {'dilation_iterations': 3, 'dilation_kernel_size': (2, 2), 'laplacian_k_size': 9, 'laplacian_threshold': 50},
        # {'dilation_iterations': 3, 'dilation_kernel_size': (3, 3), 'laplacian_k_size': 9, 'laplacian_threshold': 60},
    ]

    runs = range(0, 2)

    for param_combination in grid:
        do_runs(runs, folder_names, current_file_directory, param_combination,
                do_plots=True, show_3d_model=False, show_focus_contours=False)
