import numpy as np
import collections
import struct
import math
import json

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])



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


def write_points3D_text(points3D, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    if len(points3D) == 0:
        mean_track_length = 0
    else:
        mean_track_length = sum((len(pt.image_ids) for _, pt in points3D.items()))/len(points3D)
    HEADER = "# 3D point list with one line of data per point:\n"
    "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
    "# Number of points: {}, mean track length: {}\n".format(len(points3D), mean_track_length)

    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, pt in points3D.items():
            point_header = [pt.id, *pt.xyz, *pt.rgb, pt.error]
            fid.write(" ".join(map(str, point_header)) + " ")
            track_strings = []
            for image_id, point2D in zip(pt.image_ids, pt.point2D_idxs):
                track_strings.append(" ".join(map(str, [image_id, point2D])))
            fid.write(" ".join(track_strings) + "\n")

def read_cameras_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras

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

def read_images_binary_and_transform(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    images = {}

    c2w_dict={}

    up = np.zeros(3)
    # The c2w +y would be along the image
    # If we add all these vectors, it should point up in the colmap coordinate frame.
    # We can then find the rotation between this +z in colmap coordinate frame and +z in nerf coordinate frame.


    # find a central point they are all looking at
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
            # print(num_points2D)
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


    up = up / np.linalg.norm(up)
    print("up vector was", up)
    R = rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
    transfomation_rotation_only = np.pad(R,[0,1])
    transfomation_rotation_only[-1, -1] = 1

    for key in c2w_dict.keys():
        c2w_dict[key] = np.matmul(transfomation_rotation_only, c2w_dict[key]) # rotate up to be the z axis

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

            
    return images, c2w_dict, transfomation_rotation_only, totp, avglen

if __name__ == "__main__":

    path_to_images_file = './chair_all/sparse/0/images.bin'
    path_to_points_file = './chair_all/sparse/0/points3D.bin'
    path_to_cameras_file = './chair_all/sparse/0/cameras.bin'

    _, colmap_poses_dict, transfomation_rotation_only, totp, avglen = read_images_binary_and_transform(path_to_images_file)
    points_dict = read_points3d_binary(path_to_points_file)

    #Cameras dict keys are camera_id
    cameras_dict = read_cameras_binary(path_to_cameras_file)
    # print(cameras_dict)
    # input('q')

    # points3D_xyz = [point3d.xyz for point3d in points_dict.values()]
    points3D = {}
    for point3D in points_dict.values():

        temp_xyz = point3D.xyz
        
        #This is done since the during the processing of c2w, world axes are swapped between x and y
        # World z is also flipped. This must be appropraitely reflected when converting the point cloud as well.
        temp_xyz[0], temp_xyz[1] = temp_xyz[1], temp_xyz[0]
        temp_xyz[2] *= -1

        temp_xyz = np.matmul(transfomation_rotation_only, np.append(temp_xyz,1).T)
        temp_xyz = temp_xyz[:-1]/temp_xyz[-1]
        temp_xyz = temp_xyz -totp
        temp_xyz = temp_xyz*4.0/avglen

        # point3D.xyz = temp_xyz
        temp_3Dpoint = Point3D(
                id = point3D.id, xyz = temp_xyz, rgb = point3D.rgb,
                error=point3D.error, image_ids = point3D.image_ids,
                point2D_idxs = point3D.point2D_idxs)
        points3D[point3D.id] = temp_3Dpoint

    write_points3D_text(points3D, './points3D.txt')

    ## Write the new json transform file for train and val
    # camera_width = cameras_dict[1].width
    # camera_angle_x = math.atan( camera_width/ (camera["fl_x"] * 2)) * 2
    
    out_json_train = {
        "frames": []
    }
    out_json_val = {
        "frames": []
    }
    for filename, pose in colmap_poses_dict.items():

        if filename.split('/')[-1].split('_')[0] == 'train':
            pure_file_name = '_'.join(filename.split('/')[-1].split('_')[1:])
            frame = {"file_path": f"./train/{pure_file_name}", "transform_matrix": pose.tolist()}
            out_json_train["frames"].append(frame)
        
        else:
            pure_file_name = '_'.join(filename.split('/')[-1].split('_')[1:])
            frame = {"file_path": f"./val/{pure_file_name}", "transform_matrix": pose.tolist()}
            out_json_val["frames"].append(frame)
        
    
    # print(out_json_train)
    with open("transforms_train_colmap.json", "w") as outfile:
        json.dump(out_json_train, outfile, indent=2)
    with open("transforms_val_colmap.json", "w") as outfile:
        json.dump(out_json_val, outfile, indent=2)
    


        

    
    

    
    
    




    



    