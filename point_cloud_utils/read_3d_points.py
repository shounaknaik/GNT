import numpy as np
import collections
import struct
import json


Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])



def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_points3d_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for point_line_index in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id, xyz=xyz, rgb=rgb,
                error=error, image_ids=image_ids,
                point2D_idxs=point2D_idxs)
    
    return points3D

class Image(BaseImage):
    def qvec2rotmat(self):

        qvec = self.qvec
        return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
	da = da / np.linalg.norm(da)
	db = db / np.linalg.norm(db)
	c = np.cross(da, db)
	denom = np.linalg.norm(c)**2
	t = ob - oa
	ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
	tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
	if ta > 0:
		ta = 0
	if tb > 0:
		tb = 0
	return (oa+ta*da+ob+tb*db) * 0.5, denom

def rotmat(a, b):
	a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
	v = np.cross(a, b)
	c = np.dot(a, b)
	# handle exception for the opposite direction input
	if c < -1 + 1e-10:
		return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def read_cameras_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    images = {}
    full_poses_dict = {}

    c2w_dict={}

    up = np.zeros(3)
    # The c2w +y would be along the image
    # If we add all these vectors, it should point up in the colmap coordinate frame.
    # We can then find the rotation between this +z in colmap coordinate frame and +z in nerf coordinate frame.


    # find a central point they are all looking at
    num_points2D_max = -1
    best_image = None
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for image_index in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_name] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)

            full_pose = Image.qvec2rotmat(images[image_name])
            full_pose = np.hstack((full_pose,tvec.reshape(-1,1)))
            full_pose = np.vstack((full_pose,np.array([0,0,0,1])))

            c2w = np.linalg.inv(full_pose)

            c2w[0:3,2] *= -1 # flip the y and z axis
            c2w[0:3,1] *= -1
            c2w = c2w[[1,0,2,3],:]
            c2w[2,:] *= -1 # flip whole world upside down

            c2w_dict[image_name] = c2w

            


            up += c2w[0:3,1]
            if(num_points2D > num_points2D_max):
                num_points2D_max = num_points2D
                best_image = image_name

    

    up = up / np.linalg.norm(up)
    print("up vector was", up)
    R = rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
    R = np.pad(R,[0,1])
    R[-1, -1] = 1

    # for key in c2w_dict.keys():
    #     c2w_dict[key] = np.matmul(R, c2w_dict[key]) # rotate up to be the z axis

    print("computing center of attention...")
    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    #Z is the direction of the camera, thus take the 2 column index from pose
    for key in c2w_dict.keys():
        mf = c2w_dict[key][0:3,:]
        for key_2 in c2w_dict.keys():
            mg = c2w_dict[key_2][0:3,:]
            p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
            if w > 0.00001:
                totp += p*w
                totw += w
    
    if totw > 0.0:
        totp = totp/totw
    print(totp) # the cameras are looking at totp
    for key in c2w_dict.keys():
        c2w_dict[key][0:3,3] -= totp

    avglen = 0.
    for key in c2w_dict.keys():
        avglen += np.linalg.norm(c2w_dict[key][0:3,3])
    avglen /= len(c2w_dict)
    print("avg camera distance from origin", avglen)

    print("Scaling to Nerf scale...")
    for key in c2w_dict.keys():
        c2w_dict[key][0:3,3] *= 4.0 / avglen # scale to "nerf sized"
        w2c = np.linalg.inv(c2w_dict[key])
        full_poses_dict[key] = w2c

		
            
    return images, full_poses_dict, best_image

def read_poses_nerf(nerf_pose_file):

    with open(nerf_pose_file, "r") as fp:
        meta = json.load(fp)

    # https://docs.nerf.studio/quickstart/data_conventions.html
    # We need to change the y and z axes to be negative since there is a difference between 
    # the way COLMAP/OpenCV and NeRF datasets define the axes.
    # We are keeping all poses in OpenCV convention

    c2w_mats = {} 
    for i, frame in enumerate(meta["frames"]):
        c2w = np.array(frame["transform_matrix"])
        filename = frame["file_path"].split('/')[-1] +".png"
        w2c_blender = np.linalg.inv(c2w)
        w2c_opencv = w2c_blender
        # w2c_opencv[1:3] *= -1
        c2w_opencv = np.linalg.inv(w2c_opencv)
        c2w_mats[filename] = c2w_opencv
    
    # print("computing center of attention...")
    # totw = 0.0
    # totp = np.array([0.0, 0.0, 0.0])
    # #Z is the direction of the camera, thus take the 2 column index from pose
    # for key in c2w_mats.keys():
    #     mf = c2w_mats[key][0:3,:]
    #     for key_2 in c2w_mats.keys():
    #         mg = c2w_mats[key_2][0:3,:]
    #         p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
    #         if w > 0.00001:
    #             totp += p*w
    #             totw += w
    
    # if totw > 0.0:
    #     totp = totp/totw
    # print(totp) # the cameras are looking at totp

    # # Code for finding the scale of the NeRF dataset
    # avglen = 0 
    # for key in c2w_mats.keys():
    #     avglen += np.linalg.norm(c2w_mats[key][0:3,3])
    # avglen /= len(c2w_mats)
    # print(f"Scale of NerF dataset is {avglen}")
    # input('q')
    return c2w_mats

def get_3d_points_wrt_nerf_frame(path_poses_colmap,path_points, path_poses_nerf):
    """
    Estimates the average of the transformation between colmap origin and nerf origin. 
    Then transforms the colmap points into nerf frame of reference.
    Tranformation_colmap_nerf = Transformation_colmap_Cameraframe* Transformation_Cameraframe_nerf
    Transformation_x_y is transformation from x to y
    """
    
    _, comlmap_poses_dict, best_image = read_cameras_binary(path_poses_colmap)
    points_dict= read_points3d_binary(path_points)

    points3D_xyz = [point3d.xyz for point3d in points_dict.values()]

    # https://colmap.github.io/format.html This states the COLMAP poses are world to camera

    # CameraFrame wrt to colmap
    # poses_colmap_Cameraframe = [pose for pose in comlmap_poses_dict.values()]

    #CameraFrame wrt to NeRF dataset origin
    poses_Cameraframe_nerf = read_poses_nerf(path_poses_nerf)

    #Make the scale of colmap equal to nerf
    # Code adapted from https://github.dev/NVlabs/instant-ngp/blob/master/scripts/colmap2nerf.py#L179

    # Iterate over the colmap_Cameraframe dictionary since all the images might have not been registered in the COLMAP.
    # Ensuring the correct matrices are multiplied with each other by using the same filename.


    poses_colmap_nerf = []
    k=0
    for filename in comlmap_poses_dict.keys():
        # print(filename)
        temp_colmap_Cameraframe = comlmap_poses_dict[filename]
        temp_Cameraframe_nerf = poses_Cameraframe_nerf[filename]
        # print(np.linalg.inv(temp_colmap_Cameraframe))
        # print("Hello")
        # print(temp_Cameraframe_nerf)
        poses_colmap_nerf.append(np.dot(temp_colmap_Cameraframe,temp_Cameraframe_nerf))
        
        # colmap_nerf = np.dot(temp_colmap_Cameraframe,temp_Cameraframe_nerf)
        # temp_Cameraframe_colmap = np.linalg.inv(temp_colmap_Cameraframe)
        # # print(np.dot(temp_Cameraframe_colmap,colmap_nerf))
        # # print(temp_Cameraframe_nerf)
        # if k==2:
        #     break
        # k+=1


    
    
    for i in range(len(poses_colmap_nerf)):
        print(poses_colmap_nerf[i])
        if i==4:
            break
    print("Best")
    print(np.dot(comlmap_poses_dict[best_image],poses_Cameraframe_nerf[best_image]))
    input('q')

def read_poses_points_colmap(path_poses_colmap,path_points):
    _, comlmap_poses_dict, best_image = read_cameras_binary(path_poses_colmap)
    points_dict= read_points3d_binary(path_points)

    points3D_xyz = [point3d.xyz for point3d in points_dict.values()]

    return comlmap_poses_dict, points3D_xyz


    






