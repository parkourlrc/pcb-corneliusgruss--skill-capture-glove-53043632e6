import json
import yaml
import numpy as np

def rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def invert_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv

def main():
    KALIBR_PATH = "software/configs/calibration/kalibr/calib-camchain-imucam.yaml"
    
    with open(KALIBR_PATH, "r") as f:
        kalibr = yaml.safe_load(f)
        
    T_cam0_imu = np.array(kalibr["cam0"]["T_cam_imu"])
    T_cam1_imu = np.array(kalibr["cam1"]["T_cam_imu"])
    
    T_imu_cam0 = invert_T(T_cam0_imu)
    T_imu_cam1 = invert_T(T_cam1_imu)
    
    INTENT_PATH = "software/configs/calibration/cad_intent.json"
    with open(INTENT_PATH, "r") as f:
        cad_intent = json.load(f)
        
    T_mtcp_imu = np.eye(4)
    R_imu = np.array([
        [1,  0,  0],  # IMU X -> MTCP X
        [0,  0, -1],  # IMU Z (Right) -> MTCP -Y
        [0,  1,  0]   # IMU Y (Up) -> MTCP Z
    ])
    T_mtcp_imu[:3, :3] = R_imu
    T_mtcp_imu[:3, 3] = np.array(cad_intent["imu"]["relative_translation_mm"]) / 1000.0
    
    T_mtcp_cam0 = T_mtcp_imu @ T_imu_cam0
    T_mtcp_cam1 = T_mtcp_imu @ T_imu_cam1
    
    # Optional override from CAD for Camera Origins to enforce housing offsets
    if "relative_translation_mm" in cad_intent["cam0"]:
        T_mtcp_cam0[:3, 3] = np.array(cad_intent["cam0"]["relative_translation_mm"]) / 1000.0
    if "relative_translation_mm" in cad_intent["cam1"]:
        T_mtcp_cam1[:3, 3] = np.array(cad_intent["cam1"]["relative_translation_mm"]) / 1000.0

    # ----------------------------------------------------
    # GENERATE SENSORS VIA UNIFIED GRAPH WALKER
    # ----------------------------------------------------
    
    # Pre-populate matrices with strictly Kalibr-verified physical matrices!
    matrices = {
        "mtcp": np.eye(4),
        "imu": T_mtcp_imu,
        "cam0": T_mtcp_cam0,
        "cam1": T_mtcp_cam1
    }
    
    config = {
        "description": "Unified absolute transformations relative to the strictly defined Mount Tool Center Point (MTCP). X=Forward, Y=Left, Z=Up.",
        "T_mtcp_imu": T_mtcp_imu.tolist(),
        "T_mtcp_cam0": T_mtcp_cam0.tolist(),
        "T_mtcp_cam1": T_mtcp_cam1.tolist()
    }

    def process_node(node_name):
        if node_name in matrices:
            return matrices[node_name]
            
        cfg = cad_intent[node_name]
        
        p_pos_name = cfg["parent_pos"]
        p_rot_name = cfg["parent_rot"]
        T_parent_pos = process_node(p_pos_name)
        T_parent_rot = process_node(p_rot_name)
        
        roll, pitch, yaw = np.radians(cfg.get("relative_rotation_rpy_deg", [0,0,0]))
        
        # Intersection Target Solver Override
        if "target_intersection_depth_mm" in cfg:
            depth = cfg["target_intersection_depth_mm"] / 1000.0
            
            # 1. Compute where the Parent's optical beam is at `depth`
            P_target = T_parent_rot[:3, 3] + T_parent_rot[:3, :3] @ np.array([0, 0, depth])
            
            # 2. Compute where THIS sensor physically sits
            T_local_pos = np.eye(4)
            T_local_pos[:3, 3] = np.array(cfg.get("relative_translation_mm", [0,0,0])) / 1000.0
            O_node = (T_parent_pos @ T_local_pos)[:3, 3]
            
            # 3. Vector from Sensor to Target
            V_world = P_target - O_node
            
            # 4. Transform Target Vector into the Parent Orientation's Local Frame
            R_parent_inv = T_parent_rot[:3, :3].T
            V_local = R_parent_inv @ V_world
            dx_loc, dy_loc, dz_loc = V_local
            
            # 5. Inverse Kinematics for Z-Y-X Body Gimbal to aim at V_local
            pitch = np.arctan2(-dy_loc, dz_loc)
            yaw = np.arctan2(-dx_loc, dz_loc)
            roll = 0.0
            
        if p_rot_name == "mtcp" or p_rot_name == "imu":
            R_yaw = rot_z(yaw)
            R_pitch = rot_y(pitch)
            R_roll = rot_x(roll)
            R_local = R_yaw @ R_pitch @ R_roll
        else:
            R_pitch = rot_x(pitch)
            R_yaw = rot_y(-yaw) 
            R_roll = rot_z(roll)
            R_local = R_yaw @ R_pitch @ R_roll
            
        T_local_pos = np.eye(4)
        T_local_pos[:3, 3] = np.array(cfg.get("relative_translation_mm", [0,0,0])) / 1000.0
        final_origin = (T_parent_pos @ T_local_pos)[:3, 3]
        
        final_rot = T_parent_rot[:3, :3] @ R_local
        
        final_T = np.eye(4)
        final_T[:3, :3] = final_rot
        final_T[:3, 3] = final_origin
        
        matrices[node_name] = final_T
        config[f"T_mtcp_{node_name}"] = final_T.tolist()
        return final_T

    for key in cad_intent.keys():
        if isinstance(cad_intent[key], dict) and "parent_pos" in cad_intent[key]:
            process_node(key)

    out_path = "software/configs/calibration/mtcp_sensors.json"
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)
        
    print(f"\nSuccessfully generated unified {out_path}.")
    print(f"Cam0 Origin in MTCP (mm): {T_mtcp_cam0[:3, 3] * 1000}")
    print(f"Cam1 Origin in MTCP (mm): {T_mtcp_cam1[:3, 3] * 1000}")
    
if __name__ == "__main__":
    main()
