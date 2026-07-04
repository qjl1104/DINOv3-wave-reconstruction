import numpy as np
import sys

# --- Script Configuration ---
# The original calibration file from which to read data.
input_npz_file = 'D:\Research\wave_reconstruction_project\camera_calibration\params\stereo_calib_params_from_matlab_full.npz'
# The final text file where all data will be written.
output_txt_file = 'D:\Research\wave_reconstruction_project\camera_calibration\params\stereo_calib_params_full_in_one.txt'
# --------------------------

try:
    # Load the .npz file
    calib_data = np.load(input_npz_file)
    print(f"Successfully loaded '{input_npz_file}'.")

    # Open the output text file in write mode
    with open(output_txt_file, 'w') as f:
        # Iterate through all the arrays in the .npz file
        for key in calib_data.files:
            # Write the parameter name as a header
            f.write(f"--- Parameter: {key} ---\n")

            # Get the array data
            data = calib_data[key]

            # Use a dynamic format to handle integers and floats appropriately
            if np.issubdtype(data.dtype, np.integer):
                fmt = '%d'
            else:
                # Use scientific notation for floats to preserve precision
                fmt = '%.18e'

            # For 3D arrays (like the maps), reshape to 2D to save with savetxt
            if data.ndim == 3:
                data_to_save = data.reshape(data.shape[0], -1)
            else:
                data_to_save = data

            # Save the array data to the text file with a comma delimiter
            np.savetxt(f, data_to_save, fmt=fmt, delimiter=',')

            # Add extra newlines for better separation between parameters
            f.write("\n\n")

    print(f"Success! All calibration parameters have been saved to '{output_txt_file}'.")

except FileNotFoundError:
    print(f"Error: The input file '{input_npz_file}' was not found.")
    print("Please make sure it is in the same directory as this script.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit(1)