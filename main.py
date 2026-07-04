# main.py
import os
from camera_calibration import calibrate
from particle_processing import (
    _01_preprocess as preprocess,
    _02_particle_detection as detection,
    _03_trajectory_tracking_2d as tracking_2d,
    _04_trajectory_matching as matching,
    _05_reconstruction_3d as reconstruction_3d
)
from wave_modeling import (
    _08_training as training_pinn 
)
# Import model and data prep for PINN if running training from main
from wave_modeling.model_definition import PINN_Wave 
from wave_modeling.data_preparation import prepare_pinn_data_from_trajectories


# --- 配置参数 ---
# 标定相关 (假设标定图片为BMP)
CALIB_LEFT_IMG_DIR = "data/calibration_images/left/" # 应包含.bmp 文件
CALIB_RIGHT_IMG_DIR = "data/calibration_images/right/" # 应包含.bmp 文件
CALIB_PARAMS_DIR = "camera_calibration/params/"
CALIB_PARAMS_FILE = os.path.join(CALIB_PARAMS_DIR, "stereo_calib_params.npz")
CHESSBOARD_COLS = 9 
CHESSBOARD_ROWS = 6 
SQUARE_SIZE_MM = 25 

# 图像序列处理相关 (假设原始波浪图片为BMP)
RAW_IMG_LEFT_DIR = "data/left_images/" # 应包含.bmp 文件
RAW_IMG_RIGHT_DIR = "data/right_images/" # 应包含.bmp 文件
PREPROCESSED_LEFT_DIR = "data/preprocessed/left/" # 预处理后输出为.png
PREPROCESSED_RIGHT_DIR = "data/preprocessed/right/" # 预处理后输出为.png
DETECTIONS_LEFT_FILE = "data/detections/detections_left.pkl"
DETECTIONS_RIGHT_FILE = "data/detections/detections_right.pkl"
TRAJ_2D_LEFT_FILE = "data/trajectories/trajectories_2d_left.pkl"
TRAJ_2D_RIGHT_FILE = "data/trajectories/trajectories_2d_right.pkl"
MATCHED_PAIRS_FILE = "data/trajectories/matched_pairs_2d.pkl"
TRAJ_3D_FILE = "data/trajectories/trajectories_3d.pkl"

# 深度学习相关
MODEL_SAVE_PATH = "wave_modeling/saved_models/pinn_wave_final.pth"
PINN_EPOCHS = 5000 
PINN_LR = 1e-3
PINN_LAMBDA_PHYSICS = 1e-2


def run_calibration_stage():
    print("-" * 30)
    print("STAGE 1: CAMERA CALIBRATION")
    print("-" * 30)
    if not os.path.exists(CALIB_PARAMS_FILE) or \
       input(f"Calibration file {CALIB_PARAMS_FILE} exists. Re-calibrate? (y/N): ").lower() == 'y':
        if not os.path.exists(CALIB_LEFT_IMG_DIR) or not os.path.exists(CALIB_RIGHT_IMG_DIR):
            print(f"Error: Calibration image directories not found: {CALIB_LEFT_IMG_DIR} or {CALIB_RIGHT_IMG_DIR}")
            return False
        calibrate.calibrate_stereo_camera(
            CALIB_LEFT_IMG_DIR, CALIB_RIGHT_IMG_DIR,
            (CHESSBOARD_COLS, CHESSBOARD_ROWS), SQUARE_SIZE_MM,
            CALIB_PARAMS_DIR
        )
        print("Camera calibration finished.")
    else:
        print("Skipping calibration, using existing parameters.")
    return True

def run_particle_processing_stage():
    print("\n" + "-" * 30)
    print("STAGE 2: PARTICLE PROCESSING (DETECTION, TRACKING, 3D RECONSTRUCTION)")
    print("-" * 30)

    if not os.path.exists(CALIB_PARAMS_FILE):
        print(f"Error: Calibration parameters file {CALIB_PARAMS_FILE} not found. Cannot proceed with particle processing.")
        return False

    print("\nStep 2.1: Image Preprocessing...")
    if not os.path.exists(RAW_IMG_LEFT_DIR) or not os.path.exists(RAW_IMG_RIGHT_DIR):
        print(f"Error: Raw image directories not found: {RAW_IMG_LEFT_DIR} or {RAW_IMG_RIGHT_DIR}")
        return False
    preprocess.run_preprocessing(
        CALIB_PARAMS_FILE, RAW_IMG_LEFT_DIR, RAW_IMG_RIGHT_DIR,
        PREPROCESSED_LEFT_DIR, PREPROCESSED_RIGHT_DIR
    )
    
    print("\nStep 2.2: Particle Detection...")
    if not os.path.exists(PREPROCESSED_LEFT_DIR) or not os.path.exists(PREPROCESSED_RIGHT_DIR):
        print(f"Error: Preprocessed image directories not found: {PREPROCESSED_LEFT_DIR} or {PREPROCESSED_RIGHT_DIR}")
        return False
    detection.run_detection(
        PREPROCESSED_LEFT_DIR, PREPROCESSED_RIGHT_DIR,
        DETECTIONS_LEFT_FILE, DETECTIONS_RIGHT_FILE
    )

    print("\nStep 2.3: 2D Trajectory Tracking...")
    if not os.path.exists(DETECTIONS_LEFT_FILE) or not os.path.exists(DETECTIONS_RIGHT_FILE):
        print(f"Error: Detection files not found: {DETECTIONS_LEFT_FILE} or {DETECTIONS_RIGHT_FILE}")
        return False
    tracking_2d.run_tracking(
        DETECTIONS_LEFT_FILE, DETECTIONS_RIGHT_FILE,
        TRAJ_2D_LEFT_FILE, TRAJ_2D_RIGHT_FILE
    )

    print("\nStep 2.4: Trajectory Matching...")
    if not os.path.exists(TRAJ_2D_LEFT_FILE) or not os.path.exists(TRAJ_2D_RIGHT_FILE):
        print(f"Error: 2D trajectory files not found: {TRAJ_2D_LEFT_FILE} or {TRAJ_2D_RIGHT_FILE}")
        return False
    matching.run_trajectory_matching(
        TRAJ_2D_LEFT_FILE, TRAJ_2D_RIGHT_FILE,
        CALIB_PARAMS_FILE, MATCHED_PAIRS_FILE
    )

    print("\nStep 2.5: 3D Trajectory Reconstruction...")
    if not os.path.exists(MATCHED_PAIRS_FILE):
        print(f"Error: Matched pairs file not found: {MATCHED_PAIRS_FILE}")
        return False
    reconstruction_3d.run_3d_reconstruction(
        MATCHED_PAIRS_FILE, CALIB_PARAMS_FILE, TRAJ_3D_FILE
    )
    print("Particle processing stage finished.")
    return True

def run_wave_modeling_stage():
    print("\n" + "-" * 30)
    print("STAGE 3: WAVE MODELING (PINN TRAINING)")
    print("-" * 30)
    
    if not os.path.exists(TRAJ_3D_FILE):
        print(f"Error: 3D trajectories file {TRAJ_3D_FILE} not found. Cannot train PINN.")
        return False

    # Ensure helper scripts are accessible for training_pinn.train_pinn_model
    # This might require them to be in PYTHONPATH or for main.py to be in the project root
    # and imports within training_pinn.py to be relative if needed.
    # For simplicity, assuming they are structured to be importable.
    training_pinn.train_pinn_model(
        TRAJ_3D_FILE,
        MODEL_SAVE_PATH,
        epochs=PINN_EPOCHS,
        lr=PINN_LR,
        lambda_physics=PINN_LAMBDA_PHYSICS
    )
    print("Wave modeling (PINN training) stage finished.")
    return True

def run_inference_stage():
    print("\n" + "-" * 30)
    print("STAGE 4: INFERENCE AND VISUALIZATION")
    print("-" * 30)
    
    # Import here to avoid circular dependencies if model_definition is also imported at top level
    from wave_modeling._09_inference_visualization import infer_and_visualize_wave_surface
    
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Error: Trained model {MODEL_SAVE_PATH} not found. Cannot run inference.")
        return

    time_to_plot = 50.0 
    x_plot_range = (-5, 5) 
    y_plot_range = (-5, 5)
    
    infer_and_visualize_wave_surface(
        MODEL_SAVE_PATH, 
        time_to_plot,
        x_range=x_plot_range,
        y_range=y_plot_range
    )
    print("Inference and visualization finished.")


if __name__ == "__main__":
    DO_CALIBRATION = True
    DO_PARTICLE_PROCESSING = True
    DO_WAVE_MODELING = True
    DO_INFERENCE = True # Set to True to run inference after training

    calibration_ok = True
    if DO_CALIBRATION:
        calibration_ok = run_calibration_stage()
    
    particle_processing_ok = True
    if DO_PARTICLE_PROCESSING:
        if not calibration_ok and not os.path.exists(CALIB_PARAMS_FILE):
            print("Skipping particle processing due to missing calibration parameters.")
        else:
            particle_processing_ok = run_particle_processing_stage()

    wave_modeling_ok = True
    if DO_WAVE_MODELING:
        if not particle_processing_ok and not os.path.exists(TRAJ_3D_FILE):
            print("Skipping wave modeling due to missing 3D trajectories.")
        else:
            wave_modeling_ok = run_wave_modeling_stage()
            
    if DO_INFERENCE:
        if not wave_modeling_ok and not os.path.exists(MODEL_SAVE_PATH):
             print("Skipping inference due to missing trained model.")
        else:
            run_inference_stage()

    print("\nFull pipeline execution attempt finished.")