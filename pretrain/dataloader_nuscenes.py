import os
import random
import copy
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
import MinkowskiEngine as ME
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud


CUSTOM_SPLIT = [
    "scene-0008", "scene-0009", "scene-0019", "scene-0029", "scene-0032", "scene-0042",
    "scene-0045", "scene-0049", "scene-0052", "scene-0054", "scene-0056", "scene-0066",
    "scene-0067", "scene-0073", "scene-0131", "scene-0152", "scene-0166", "scene-0168",
    "scene-0183", "scene-0190", "scene-0194", "scene-0208", "scene-0210", "scene-0211",
    "scene-0241", "scene-0243", "scene-0248", "scene-0259", "scene-0260", "scene-0261",
    "scene-0287", "scene-0292", "scene-0297", "scene-0305", "scene-0306", "scene-0350",
    "scene-0352", "scene-0358", "scene-0361", "scene-0365", "scene-0368", "scene-0377",
    "scene-0388", "scene-0391", "scene-0395", "scene-0413", "scene-0427", "scene-0428",
    "scene-0438", "scene-0444", "scene-0452", "scene-0453", "scene-0459", "scene-0463",
    "scene-0464", "scene-0475", "scene-0513", "scene-0533", "scene-0544", "scene-0575",
    "scene-0587", "scene-0589", "scene-0642", "scene-0652", "scene-0658", "scene-0669",
    "scene-0678", "scene-0687", "scene-0701", "scene-0703", "scene-0706", "scene-0710",
    "scene-0715", "scene-0726", "scene-0735", "scene-0740", "scene-0758", "scene-0786",
    "scene-0790", "scene-0804", "scene-0806", "scene-0847", "scene-0856", "scene-0868",
    "scene-0882", "scene-0897", "scene-0899", "scene-0976", "scene-0996", "scene-1012",
    "scene-1015", "scene-1016", "scene-1018", "scene-1020", "scene-1024", "scene-1044",
    "scene-1058", "scene-1094", "scene-1098", "scene-1107",
]


def minkunet_collate_pair_fn(list_data):
    """
    Collate function adapted for creating batches with MinkowskiEngine.
    """
    (
        coords,
        feats,
        images,
        ori_images,
        pairing_points,
        pairing_images,
        inverse_indexes,
        superpixels,
        pc,
        pos_non_manifold, occupancies, intensities_non_manifold
    ) = list(zip(*list_data))
    batch_n_points, batch_n_pairings = [], []

    offset = 0
    for batch_id in range(len(coords)):

        # Move batchids to the beginning
        coords[batch_id][:, 0] = batch_id
        # pc[batch_id][:, 0] = batch_id
        pos_non_manifold[batch_id][:, 0] = batch_id
        pairing_points[batch_id][:] += offset
        pairing_images[batch_id][:, 0] += batch_id * images[0].shape[0]

        batch_n_points.append(coords[batch_id].shape[0])
        batch_n_pairings.append(pairing_points[batch_id].shape[0])
        offset += coords[batch_id].shape[0]

    # Concatenate all lists
    coords_batch = torch.cat(coords, 0).int()
    # pc_batch = torch.cat(pc, 0).float()
    pos_non_manifold_batch = torch.cat(pos_non_manifold, 0).float()
    intensities_non_manifold_batch = torch.cat(intensities_non_manifold).float()
    occupancies_batch = torch.cat(occupancies).int()
    pairing_points = torch.tensor(np.concatenate(pairing_points))
    pairing_images = torch.tensor(np.concatenate(pairing_images))
    feats_batch = torch.cat(feats, 0).float()
    images_batch = torch.cat(images, 0).float()
    ori_images_batch = torch.cat(ori_images, 0).float()
    superpixels_batch = torch.tensor(np.concatenate(superpixels))
    return {
        "sinput_C": coords_batch,
        "sinput_F": feats_batch,
        "input_I": images_batch,
        "target_I": ori_images_batch,
        "pairing_points": pairing_points,
        "pairing_images": pairing_images,
        "batch_n_pairings": batch_n_pairings,
        "inverse_indexes": inverse_indexes,
        "superpixels": superpixels_batch,
        "pos_non_manifold": pos_non_manifold_batch,
        "intensities_non_manifold": intensities_non_manifold_batch,
        "occupancies": occupancies_batch,
        "pc": pc,
    }


class NuScenesMatchDataset(Dataset):
    """
    Dataset matching a 3D points cloud and an image using projection.
    """

    def __init__(
        self,
        phase,
        config,
        shuffle=False,
        cloud_transforms=None,
        mixed_transforms=None,
        **kwargs,
    ):
        self.phase = phase
        self.shuffle = shuffle
        self.cloud_transforms = cloud_transforms
        self.mixed_transforms = mixed_transforms
        self.voxel_size = config["voxel_size"]
        self.cylinder = config["cylindrical_coordinates"]
        self.superpixels_type = config["superpixels_type"]
        self.bilinear_decoder = config["decoder"] == "bilinear"

        # ad hoc
        self.n_non_manifold_pts = 2048 
        self.non_manifold_dist = 0.1

        if "cached_nuscenes" in kwargs:
            self.nusc = kwargs["cached_nuscenes"]
        else:
            self.nusc = NuScenes(
                version="v1.0-trainval", dataroot="datasets/nuscenes", verbose=False
            )

        self.list_keyframes = []
        # a skip ratio can be used to reduce the dataset size and accelerate experiments
        try:
            skip_ratio = config["dataset_skip_step"]
        except KeyError:
            skip_ratio = 1
        skip_counter = 0
        if phase in ("train", "val", "test"):
            phase_scenes = create_splits_scenes()[phase]
        elif phase == "parametrizing":
            phase_scenes = list(
                set(create_splits_scenes()["train"]) - set(CUSTOM_SPLIT)
            )
        elif phase == "verifying":
            phase_scenes = CUSTOM_SPLIT
        # create a list of camera & lidar scans
        for scene_idx in range(len(self.nusc.scene)):
            scene = self.nusc.scene[scene_idx]
            if scene["name"] in phase_scenes:
                skip_counter += 1
                if skip_counter % skip_ratio == 0:
                    self.create_list_of_scans(scene)

    def create_list_of_scans(self, scene):
        # Get first and last keyframe in the scene
        current_sample_token = scene["first_sample_token"]

        # Loop to get all successive keyframes
        list_data = []
        while current_sample_token != "":
            current_sample = self.nusc.get("sample", current_sample_token)
            list_data.append(current_sample["data"])
            current_sample_token = current_sample["next"]

        # Add new scans in the list
        self.list_keyframes.extend(list_data)

    def map_pointcloud_to_image(self, data, min_dist: float = 1.0, mask_prob: float = 0.5):
        """
        Given a lidar token and camera sample_data token, load pointcloud and map it to
        the image plane. Code adapted from nuscenes-devkit
        https://github.com/nutonomy/nuscenes-devkit.
        :param min_dist: Distance from the camera below which points are discarded.
        """
        pointsensor = self.nusc.get("sample_data", data["LIDAR_TOP"])
        pcl_path = os.path.join(self.nusc.dataroot, pointsensor["filename"])
        pc_original = LidarPointCloud.from_file(pcl_path)
        
        # pc_original.subsample(16380 / pc_original.nbr_points())
        pc_ref = pc_original.points # (4, N)
        
        #### VoxelDecimation start ####
        # pos = copy.deepcopy(pc_original.points.T[:, :3])
        # pos = np.round(pos/self.voxel_size)
        # num_pts = pos.shape[0]
        # _, indices = np.unique(pos, return_index=True, axis=0)
        # pc_ref = pc_original.points[:, indices]
        # pc_original.points = pc_ref
        #### VoxelDecimation done ####

        images = []
        ori_images = []
        superpixels = []
        pairing_points = np.empty(0, dtype=np.int64)
        pairing_images = np.empty((0, 3), dtype=np.int64)
        camera_list = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_FRONT_LEFT",
        ]
        if self.shuffle:
            np.random.shuffle(camera_list)
        for i, camera_name in enumerate(camera_list):
            pc = copy.deepcopy(pc_original)
            cam = self.nusc.get("sample_data", data[camera_name])
            im = np.array(Image.open(os.path.join(self.nusc.dataroot, cam["filename"])))
            sp = Image.open(
                f"superpixels/nuscenes/"
                f"superpixels_{self.superpixels_type}/{cam['token']}.png"
            )
            superpixels.append(np.array(sp))

            patch_size = 16  # 假设使用 16x16 的 patch
            h, w = im.shape[0], im.shape[1]

            if random.random() < mask_prob:
                # 1. 在 Patch 维度生成随机 mask (True 表示被遮挡)
                num_patches_h, num_patches_w = h // patch_size, w // patch_size
                patch_mask = np.random.rand(num_patches_h, num_patches_w) < 0.5  # 50% 遮挡率
                
                # 2. 利用克罗内克积将 mask 放大回原图尺寸
                mask = np.kron(patch_mask, np.ones((patch_size, patch_size), dtype=bool))
                
                # 处理不能被 patch_size 整除的边缘像素
                mask_full = np.zeros((h, w), dtype=bool)
                mask_full[:mask.shape[0], :mask.shape[1]] = mask
                mask = mask_full
            else:
                mask = np.zeros((h, w), dtype=bool)
                
            # Points live in the point sensor frame. So they need to be transformed via
            # global to the image plane.
            # First step: transform the pointcloud to the ego vehicle frame for the
            # timestamp of the sweep.
            cs_record = self.nusc.get(
                "calibrated_sensor", pointsensor["calibrated_sensor_token"]
            )
            pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
            pc.translate(np.array(cs_record["translation"]))

            # Second step: transform from ego to the global frame.
            poserecord = self.nusc.get("ego_pose", pointsensor["ego_pose_token"])
            pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
            pc.translate(np.array(poserecord["translation"]))

            # Third step: transform from global into the ego vehicle frame for the
            # timestamp of the image.
            poserecord = self.nusc.get("ego_pose", cam["ego_pose_token"])
            pc.translate(-np.array(poserecord["translation"]))
            pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)

            # Fourth step: transform from ego into the camera.
            cs_record = self.nusc.get(
                "calibrated_sensor", cam["calibrated_sensor_token"]
            )
            pc.translate(-np.array(cs_record["translation"]))
            pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix.T)

            # Fifth step: actually take a "picture" of the point cloud.
            # Grab the depths (camera frame z axis points away from the camera).
            depths = pc.points[2, :]

            # Take the actual picture
            # (matrix multiplication with camera-matrix + renormalization).
            points = view_points(
                pc.points[:3, :],
                np.array(cs_record["camera_intrinsic"]),
                normalize=True,
            )

            # Remove points that are either outside or behind the camera
            points = points[:2].T  # Only take x, y for 2D projection
            mask_pts = np.ones(depths.shape[0], dtype=bool)
            mask_pts = np.logical_and(mask_pts, depths > min_dist)  # Depth > min_dist
            mask_pts = np.logical_and(mask_pts, points[:, 0] > 0)  # x must be within image width
            mask_pts = np.logical_and(mask_pts, points[:, 0] < im.shape[1] - 1)
            mask_pts = np.logical_and(mask_pts, points[:, 1] > 0)  # y must be within image height
            mask_pts = np.logical_and(mask_pts, points[:, 1] < im.shape[0] - 1)
            matching_points = np.where(mask_pts)[0]

            # Now match points to image pixels
            matching_pixels = np.round(np.flip(points[matching_points], axis=1)).astype(np.int64)

            # Apply the image mask using NumPy array indexing (avoiding for loop)
            valid_mask = 1 - mask[matching_pixels[:, 0], matching_pixels[:, 1]]  # Check if pixel is not in masked region
            valid_points = matching_points[valid_mask]  # Select valid points
            valid_pixels = matching_pixels[valid_mask]  # Select valid pixels

            pairing_points = np.concatenate((pairing_points, valid_points))
            pairing_images = np.concatenate(
                (
                    pairing_images,
                    np.concatenate(
                        (
                            np.ones((valid_pixels.shape[0], 1), dtype=np.int64) * i,
                            valid_pixels,
                        ),
                        axis=1,
                    ),
                )
            )

            ori_images.append(im / 255)
            masked_image = np.array(im)  # Create masked image for network input
            masked_image[mask] = 0  # Set masked region to black
            images.append(masked_image / 255)  # Normalize masked image

        return pc_ref.T, images, ori_images, pairing_points, pairing_images, np.stack(superpixels)

    def __len__(self):
        return len(self.list_keyframes)

    def __getitem__(self, idx):
        (
            pc,
            images,
            ori_images,
            pairing_points,
            pairing_images,
            superpixels,
        ) = self.map_pointcloud_to_image(self.list_keyframes[idx])
        superpixels = torch.tensor(superpixels)

        intensity = torch.tensor(pc[:, 3:] / 255.0)
        pc = torch.tensor(pc[:, :3])
        images = torch.tensor(np.array(images, dtype=np.float32).transpose(0, 3, 1, 2))
        ori_images = torch.tensor(np.array(ori_images, dtype=np.float32).transpose(0, 3, 1, 2))

        #################### select query points (non_manifold) for occupancy estimation ##################
        n_nmp = self.n_non_manifold_pts
        n_nmp_out = n_nmp // 3
        n_nmp_out_far = n_nmp // 3
        n_nmp_in = n_nmp - 2 * (n_nmp//3)
        nmp_choice_in = torch.randperm(pc.shape[0])[:n_nmp_in]
        nmp_choice_out = torch.randperm(pc.shape[0])[:n_nmp_out]
        nmp_choice_out_far = torch.randperm(pc.shape[0])[:n_nmp_out_far]
        # center
        center = torch.zeros((1,3), dtype=torch.float)

        # in points
        pos = pc[nmp_choice_in]
        dirs = F.normalize(pos, dim=1)
        pos_in = pos + self.non_manifold_dist * dirs * torch.rand((pos.shape[0],1))
        occ_in = torch.ones(pos_in.shape[0], dtype=torch.long)
        
        # out points
        pos = pc[nmp_choice_out]
        dirs = F.normalize(pos, dim=1)
        pos_out = pos - self.non_manifold_dist * dirs * torch.rand((pos.shape[0],1))
        occ_out = torch.zeros(pos_out.shape[0], dtype=torch.long)

        # out far points
        pos = pc[nmp_choice_out_far]
        dirs = F.normalize(pos, dim=1)
        pos_out_far = (pos - center) * torch.rand((pos.shape[0],1)) + center
        occ_out_far = torch.zeros(pos_out_far.shape[0], dtype=torch.long)

        intensities_in = intensity[nmp_choice_in]
        intensities_out = intensity[nmp_choice_out]
        intensities_out_far = torch.full((pos_out_far.shape[0],1), fill_value=-1)

        pos_non_manifold = torch.cat([pos_in, pos_out, pos_out_far], dim=0)
        occupancies = torch.cat([occ_in, occ_out, occ_out_far], dim=0)
        intensities_non_manifold = torch.cat([intensities_in, intensities_out, intensities_out_far])

        #################### select query points (non_manifold) for occupancy estimation done ##################
        if self.cloud_transforms:
            pc, pos_non_manifold = self.cloud_transforms(pc, pos_non_manifold)
        if self.mixed_transforms:
            (
                pc,
                intensity,
                images,
                ori_images,
                pairing_points,
                pairing_images,
                superpixels,
            ) = self.mixed_transforms(
                pc, intensity, images, ori_images, pairing_points, pairing_images, superpixels
            )

        if self.cylinder:
            # Transform to cylinder coordinate and scale for voxel size
            x, y, z = pc.T
            rho = torch.sqrt(x ** 2 + y ** 2) / self.voxel_size
            phi = torch.atan2(y, x) * 180 / np.pi  # corresponds to a split each 1°
            z = z / self.voxel_size
            coords_aug = torch.cat((rho[:, None], phi[:, None], z[:, None]), 1)
        else:
            coords_aug = pc / self.voxel_size

        # Voxelization with MinkowskiEngine
        discrete_coords, indexes, inverse_indexes = ME.utils.sparse_quantize(
            coords_aug.contiguous(), return_index=True, return_inverse=True
        )
        # indexes here are the indexes of points kept after the voxelization
        pairing_points = inverse_indexes[pairing_points]

        unique_feats = intensity[indexes]

        discrete_coords = torch.cat(
            (
                torch.zeros(discrete_coords.shape[0], 1, dtype=torch.int32),
                discrete_coords,
            ),
            1,
        )
        # first column: batch_idx
        pos_non_manifold = torch.cat([torch.zeros(pos_non_manifold.shape[0], 1, dtype=torch.int32), pos_non_manifold], 1)
        # pc = torch.cat([torch.zeros(pc.shape[0], 1, dtype=torch.int32), pc], 1)
        # voxel_continuous_coords = pc[indexes]

        return (
            discrete_coords,
            unique_feats,
            images,
            ori_images,
            pairing_points,
            pairing_images,
            inverse_indexes,
            superpixels,
            pc,
            pos_non_manifold, occupancies, intensities_non_manifold
        )
