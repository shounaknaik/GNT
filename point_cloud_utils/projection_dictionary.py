from read_3d_points import read_points3D_text
import sys
sys.path.append("..")
from gnt.projection import Projector
from gnt.data_loaders.create_training_dataset import create_training_dataset
from gnt.sample_ray import RaySamplerSingleImage

import config
import torch
import numpy as np
import pickle
from tqdm import tqdm

parser = config.config_parser()
args = parser.parse_args()

path_3d_points_txt = f'../data/nerf_synthetic/chair/points3D.txt'
points3D_xyz = read_points3D_text(path_3d_points_txt)

device = "cuda:{}".format(args.local_rank)

# create training dataset
train_dataset, train_sampler = create_training_dataset(args)
# currently only support batch_size=1 (i.e., one set of target and source views) for each GPU node
# please use distributed parallel on multiple GPUs to train multiple target views per batch
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=1,
    worker_init_fn=lambda _: np.random.seed(),
    num_workers=args.workers,
    pin_memory=True,
    sampler=train_sampler,
    shuffle=True if train_sampler is None else False,
)

# create projector
projector = Projector(device=device)

print("Projecting the points onto each camera and storing the projection images .....")
result_projection_dictionary = {}
for train_data in tqdm(train_loader,total = len(train_loader)):
    # load training rays
    current_camera_path = train_data["rgb_path"][0].split("/")[-1]
    # print(current_camera_path)
    ray_sampler = RaySamplerSingleImage(train_data, device)
    N_rand = int(
        1.0 * args.N_rand * args.num_source_views / train_data["src_rgbs"][0].shape[0]
    )
    ray_batch = ray_sampler.random_sample(
        N_rand,
        sample_mode=args.sample_mode,
        center_ratio=args.center_ratio,
    )
    # print(ray_batch)
    depth_image_map = projector.get_depth_in_one_image(points3D_xyz, ray_batch["camera"])
    # print(ray_batch["camera"])
    result_projection_dictionary[current_camera_path] = depth_image_map

    # print(depth_image_map)
    non_zero_indices = np.nonzero(depth_image_map)
    non_zero_values = depth_image_map[non_zero_indices]
    # print(non_zero_values)
    # print(len(non_zero_values))
    # print(depth_image_map.max())
    # print(np.min(depth_image_map[depth_image_map!=0]))
    # input('q')

print(len(result_projection_dictionary))
print(len(train_loader))



# file_path = '../data/nerf_synthetic/chair/projection_dict.pkl'

# # Pickle dump the dictionary into a file
# with open(file_path, 'wb') as f:
#     pickle.dump(result_projection_dictionary, f)