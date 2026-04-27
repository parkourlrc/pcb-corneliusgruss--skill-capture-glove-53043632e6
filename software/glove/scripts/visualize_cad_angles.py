import numpy as np
import matplotlib.pyplot as plt

def plot_vector(ax, origin, vector, color, label, length=100.0):
    point = origin + vector * length
    ax.quiver(origin[0], origin[1], origin[2], 
              vector[0]*length, vector[1]*length, vector[2]*length,
              color=color, label=label, arrow_length_ratio=0.1, linewidth=2)
    return point

def main():
    # 1. Exact Extrinsics from Kalibr
    R_cam0_imu = np.array([
        [-0.4192096,  0.0140686,  0.9077804],
        [-0.0843432, -0.9961593, -0.0235110],
        [ 0.9039632, -0.0864211,  0.4187861]
    ])
    R_cam1_imu = np.array([
        [-0.0234876,  0.0085003,  0.9996879],
        [-0.6769727, -0.7359448, -0.0096476],
        [ 0.7356331, -0.6769881,  0.0230401]
    ])
    
    # Kalibr t is the position of the IMU in the camera frame (meters).
    t_kalibr_cam0 = np.array([0.0755848, 0.0708566, 0.0272288]) * 1000.0
    t_kalibr_cam1 = np.array([0.0476767, -0.0501389, -0.0623717]) * 1000.0
    
    # Position of camera in IMU frame = - R_imu_cam @ t_kalibr
    t_cam0 = -R_cam0_imu.T @ t_kalibr_cam0
    t_cam1 = -R_cam1_imu.T @ t_kalibr_cam1
    
    # Extract Camera Optical Axes (Z-axis) in IMU frame
    dir_cam0 = R_cam0_imu[2, :]  
    dir_cam1 = R_cam1_imu[2, :]  

    # 2. ToF Ray Calculations
    
    # TOP TOF
    b_top = 20.0  # placed 20mm physically BELOW the camera lens
    d_top = 200.0 
    cross_angle_rad = np.arctan2(b_top, d_top) # ~5.7 degrees Upwards 
    
    up_cam0 = -R_cam0_imu[1, :] # standard vision up-vector in IMU (negative Y in optical)
    tof0_origin = t_cam0 - up_cam0 * b_top # physical offset below main lens
    
    cross_point = t_cam0 + dir_cam0 * d_top
    dir_tof0 = cross_point - tof0_origin
    dir_tof0 /= np.linalg.norm(dir_tof0)

    # BOTTOM TOF: Orthogonal Optimization
    # We want dir_tof1 to be EXACTLY orthogonal to dir_tof0
    # constraint: dot(dir_tof1, dir_tof0) == 0.
    # We start with the pitch of cam1 (dir_cam1), and rotate it around cam1's Local UP axis
    # until dir_tof1 dot dir_tof0 = 0.
    
    up_cam1 = -R_cam1_imu[1, :]
    
    def rot_axis(axis, deg):
        rad = np.radians(deg)
        c, s = np.cos(rad), np.sin(rad)
        vx, vy, vz = axis
        return np.array([
            [c + vx*vx*(1-c), vx*vy*(1-c) - vz*s, vx*vz*(1-c) + vy*s],
            [vy*vx*(1-c) + vz*s, c + vy*vy*(1-c), vy*vz*(1-c) - vx*s],
            [vz*vx*(1-c) - vy*s, vz*vy*(1-c) + vx*s, c + vz*vz*(1-c)]
        ])

    optimal_yaw = 0.0
    min_dot = 1000.0
    for yaw in np.linspace(-90, 90, 1800):
        test_dir = rot_axis(up_cam1, yaw) @ dir_cam1
        test_dir /= np.linalg.norm(test_dir)
        d = abs(np.dot(dir_tof0, test_dir))
        if d < min_dot:
            min_dot = d
            optimal_yaw = yaw
            
    dir_tof1 = rot_axis(up_cam1, optimal_yaw) @ dir_cam1
    dir_tof1 /= np.linalg.norm(dir_tof1)
    
    # Bottom ToF offset: move it to the right by 20mm
    right_cam1 = R_cam1_imu[0, :]
    tof1_origin = t_cam1 + right_cam1 * 20.0

    print("\n--- CAD REFERENCE ANGLES ---")
    print(f"Top ToF intersects cam0 axis {d_top}mm away (with {b_top}mm physical offset below lens).")
    print(f"Angle offset: {np.degrees(cross_angle_rad):.1f} degrees.")
    
    print(f"\nTo achieve PERFECT 90.0 degree orthogonality between the two ToF sensors:")
    print(f"The Bottom ToF should be yawed outward by exactly: {optimal_yaw:.2f} degrees")
    print(f"relative to the optical axis of cam1.")
    print(f"Dot product of the two ToF rays: {np.dot(dir_tof0, dir_tof1):.5f}")

    # 3. Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw IMU Origin
    ax.scatter([0], [0], [0], color='k', s=100, label='IMU Center')
    plot_vector(ax, np.zeros(3), np.array([1, 0, 0]), 'r', 'IMU +X', 30)
    plot_vector(ax, np.zeros(3), np.array([0, 1, 0]), 'g', 'IMU +Y', 30)
    plot_vector(ax, np.zeros(3), np.array([0, 0, 1]), 'b', 'IMU +Z', 30)

    plot_vector(ax, t_cam0, dir_cam0, 'c', 'cam0 (Top)', length=150)
    plot_vector(ax, t_cam1, dir_cam1, 'm', 'cam1 (Bottom)', length=150)
    plot_vector(ax, tof0_origin, dir_tof0, 'orange', f'ToF0', length=150)
    plot_vector(ax, tof1_origin, dir_tof1, 'purple', f'ToF1 (Orthogonal, Yaw={optimal_yaw:.1f}deg)', length=150)
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title("Glove V3: Mathematically Optimal Tof Geometry")
    ax.set_xlim([-100, 200])
    ax.set_ylim([-100, 200])
    ax.set_zlim([-100, 200])
    ax.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
