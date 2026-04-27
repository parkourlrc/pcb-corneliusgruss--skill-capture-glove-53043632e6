import json
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

def build_kinematic_tree(intent):
    config = {
        "description": "THEORETICAL CAD INTENT ONLY. DO NOT USE FOR SLAM. Use for Fusion 360 geometry."
    }
    
    # Root World Frame
    matrices = {"mtcp": np.eye(4)}
    
    # R_base rotates from MTCP into OpenCV Camera Frame
    R_base = np.array([
        [ 0,  0, 1],
        [-1,  0, 0],
        [ 0, -1, 0]
    ])
    
    def process_node(node_name):
        if node_name in matrices:
            return matrices[node_name]
            
        cfg = intent[node_name]
        
        # Recursively build parents if not solved
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
            print(f"[*] Dynamically solved Optical Targeting for {node_name}: Pitch {np.degrees(pitch):.2f}°, Yaw {np.degrees(yaw):.2f}°")
            
        if p_rot_name == "mtcp" or p_rot_name == "imu":
            R_yaw = rot_z(yaw)
            R_pitch = rot_y(pitch)
            R_roll = rot_x(roll)
            R_local = R_yaw @ R_pitch @ R_roll
        else:
            # Parent is Camera-like Body (Z=Fwd, X=Right, Y=Down).
            # Pitch is rotation around X-axis. Yaw is rotation around Y-axis. Roll is rotation around Z-axis.
            R_pitch = rot_x(pitch)
            R_yaw = rot_y(-yaw) # -yaw preserves + Left mapping matching MTCP
            R_roll = rot_z(roll)
            R_local = R_yaw @ R_pitch @ R_roll
        
        # Apply Base Mapping if crossing the boundary from MTCP into Camera Frame space
        if p_rot_name == "mtcp" and node_name.startswith("cam"):
            R_local = R_local @ R_base
            
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
        
    for key in intent.keys():
        if isinstance(intent[key], dict) and "parent_pos" in intent[key]:
            process_node(key)

    return config

def main():
    intent_file = "software/configs/calibration/cad_intent.json"
    with open(intent_file, "r") as f:
        intent = json.load(f)
        
    config = build_kinematic_tree(intent)
    
    out_path = "software/configs/calibration/theoretical_mtcp.json"
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)
        
    print(f"Generated {out_path} from cad_intent.json for CAD modeling!")

if __name__ == "__main__":
    main()
