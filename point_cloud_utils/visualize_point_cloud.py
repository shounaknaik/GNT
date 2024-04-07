import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_camera(ax, pose, size=0.1, color='r'):
    """
    Plot a camera symbolized by a pyramid.
    
    Parameters:
    - ax: Axes3D object
    - pose: 3D position of the camera
    - size: Size of the camera symbol
    - color: Color of the camera symbol
    """
    u, v, w = size * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    verts = [pose + u, pose - 0.5 * u - 0.5 * v, pose - 0.5 * u + 0.5 * v, pose + w]
    poly = Poly3DCollection([verts], color=color, alpha=0.5)
    ax.add_collection3d(poly)

# Read point cloud data
points_data = points_data = np.loadtxt('points3D.txt', skiprows=1, delimiter=' ',usecols=(1,2,3))
print(points_data)


# Read camera poses
with open('transforms_train_colmap.json', 'r') as f:
    transforms_data = json.load(f)

# Extract camera poses and orientations
frames = transforms_data["frames"]
poses = [np.array(frame["transform_matrix"]) for frame in frames]
# import pdb
# pdb.set_trace()
# temp = poses[0]
# print("h")
# print(poses[0])
# print(poses[0][:3,3])
# rotation = poses[0][:3,:3]
# translation = poses[0][:3,3]
# new_translation = rotation.T @ -translation
# poses = [np.linalg.inv(pose) for pose in poses]
# print(poses[0])
# print(poses[0][:3,3])
# print(poses[0]@ temp)
translations = [pose[:3,3] for pose in poses]
rotations = [pose[:3,:3] for pose in poses]
# camera_poses = [np.array(transform['transform']['translation']) for transform in transforms_data]
# camera_orientations = [np.array(transform['transform']['rotation']) for transform in transforms_data]

# Visualize point cloud
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_data[:,0], points_data[:,1], points_data[:,2], c='r', marker='o', label='Points',s=0.1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_xlim([-5, 5])  # Adjust the limits of the x-axis
ax.set_ylim([-5, 5])  # Adjust the limits of the y-axis
ax.set_zlim([-5, 5])  # Adjust the limits of the z-axis


# Visualize camera poses
camera_poses = np.array(translations)
# print(translations)
# ax.scatter(camera_poses[:10000,0], camera_poses[:10000,1], camera_poses[:10000,2], c='b', marker='o', label='Cameras')
# print(camera_poses)
for pose, orientation in zip(camera_poses, rotations):
    plot_camera(ax, pose, size=0.1, color='b')


plt.legend()
plt.show()
