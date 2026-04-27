import json
import numpy as np
import matplotlib.pyplot as plt

def plot_frame(ax, T, name, scale=30.0):
    origin = T[:3, 3] * 1000.0  # Convert to mm
    R = T[:3, :3]
    
    # Plot Origin point
    ax.scatter([origin[0]], [origin[1]], [origin[2]], color='k', s=50)
    ax.text(origin[0], origin[1], origin[2] + 10, name, color='k')
    
    # X axis (Red)
    x_axis = R[:, 0]
    ax.quiver(origin[0], origin[1], origin[2], 
              x_axis[0]*scale, x_axis[1]*scale, x_axis[2]*scale,
              color='r', arrow_length_ratio=0.1)
              
    # Y axis (Green)
    y_axis = R[:, 1]
    ax.quiver(origin[0], origin[1], origin[2], 
              y_axis[0]*scale, y_axis[1]*scale, y_axis[2]*scale,
              color='g', arrow_length_ratio=0.1)
              
    # Z axis (Blue)
    z_axis = R[:, 2]
    ax.quiver(origin[0], origin[1], origin[2], 
              z_axis[0]*scale, z_axis[1]*scale, z_axis[2]*scale,
              color='b', arrow_length_ratio=0.1)

def main():
    with open("software/configs/calibration/mtcp_sensors.json", "r") as f:
        config = json.load(f)
        
    T_mtcp_imu = np.array(config["T_mtcp_imu"])
    T_mtcp_cam0 = np.array(config["T_mtcp_cam0"])
    T_mtcp_cam1 = np.array(config["T_mtcp_cam1"])
    T_mtcp_tof0 = np.array(config["T_mtcp_tof0"])
    T_mtcp_tof1 = np.array(config["T_mtcp_tof1"])
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot MTCP Root Frame
    plot_frame(ax, np.eye(4), "MTCP Origin\n(Robot Attachment)", scale=50.0)
    
    # Plot IMU Frame
    plot_frame(ax, T_mtcp_imu, "IMU Chip", scale=30.0)
    
    # Plot Camera Frames
    plot_frame(ax, T_mtcp_cam0, "Cam0 (Top)", scale=30.0)
    plot_frame(ax, T_mtcp_cam1, "Cam1 (Bottom)", scale=30.0)
    
    # Plot ToF Frames
    plot_frame(ax, T_mtcp_tof0, "ToF0 (Top)", scale=20.0)
    plot_frame(ax, T_mtcp_tof1, "ToF1 (Bottom)", scale=20.0)

    
    # Draw Optical Viewing Rays (Cameras look down their local +Z axis in Kalibr convention)
    def plot_optical_ray(ax, T, color, length=150.0):
        origin = T[:3, 3] * 1000.0
        # Z axis is the optical forward vector in Kalibr OpenCv mapping
        optical_vector = T[:3, :3][:, 2]
        ax.quiver(origin[0], origin[1], origin[2],
                  optical_vector[0]*length, optical_vector[1]*length, optical_vector[2]*length,
                  color=color, alpha=0.5, linestyle='--')

    plot_optical_ray(ax, T_mtcp_cam0, 'c')
    plot_optical_ray(ax, T_mtcp_cam1, 'm')
    
    plot_optical_ray(ax, T_mtcp_tof0, 'orange', length=200.0)
    plot_optical_ray(ax, T_mtcp_tof1, 'purple', length=200.0)
    
    ax.set_xlabel('MTCP X (mm) [Forward]')

    ax.set_ylabel('MTCP Y (mm) [Left]')
    ax.set_zlabel('MTCP Z (mm) [Up]')
    ax.set_title("Glove V3 Unified System Geometry")
    ax.set_xlim([-100, 150])
    ax.set_ylim([-100, 150])
    ax.set_zlim([-100, 150])
    
    # Set a good viewing angle
    ax.view_init(elev=30, azim=-45)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
