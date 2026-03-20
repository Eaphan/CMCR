import torch
import argparse
from downstream.evaluate import evaluate
from utils.read_config import generate_config
from downstream.model_builder import make_model
from downstream.dataloader_kitti import make_data_loader as make_data_loader_kitti
from downstream.dataloader_nuscenes import make_data_loader as make_data_loader_nuscenes

import torch
from tqdm import tqdm
from copy import deepcopy
from MinkowskiEngine import SparseTensor

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
CLASSES_NUSCENES = [
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
]

CLASSES_KITTI = [
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
]

sk_class_colors = np.array([
                    [0, 0, 0],
                    [245, 150, 100],
                    [245, 230, 100],
                    [150, 60, 30],
                    [180, 30, 80],
                    [255, 0, 0],
                    [30, 30, 255],
                    [200, 40, 255],
                    [90, 30, 150],
                    [255, 0, 255],
                    [255, 150, 255],
                    [75, 0, 75],
                    [75, 0, 175],
                    [0, 200, 255],
                    [50, 120, 255],
                    [0, 175, 0],
                    [0, 60, 135],
                    [80, 240, 150],
                    [150, 240, 255],
                    [0, 0, 255],
                ], dtype=np.uint8)

ns_class_colors = np.array([
    [0, 0, 0],
    [112, 128, 144],  
    [220, 20, 60],  # Crimson
    [255, 127, 80],  # Coral
    [255, 158, 0],  # Orange
    [233, 150, 70],  # Darksalmon
    [255, 61, 99],  # Red
    [0, 0, 230],  # Blue
    [47, 79, 79],  # Darkslategrey
    [255, 140, 0],  # Darkorange
    [255, 99, 71],  # Tomato
    [0, 207, 191],  # nuTonomy green
    [175, 0, 75],
    [75, 0, 75],
    [112, 180, 60],
    [222, 184, 135],  # Burlywood
    [0, 175, 0],  # Green
    ], dtype=np.uint8)

def main():
    """
    Code for launching the downstream evaluation
    """
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file", type=str, default=None, help="specify the config for training"
    )
    parser.add_argument(
        "--resume_path", type=str, default=None, help="provide a path to resume an incomplete training"
    )
    parser.add_argument(
        "--dataset", type=str, default=None, help="Choose between nuScenes and KITTI"
    )
    args = parser.parse_args()
    if args.cfg_file is None and args.dataset is not None:
        if args.dataset.lower() == "kitti":
            args.cfg_file = "config/semseg_kitti.yaml"
        elif args.dataset.lower() == "nuscenes":
            args.cfg_file = "config/semseg_nuscenes.yaml"
        else:
            raise Exception(f"Dataset not recognized: {args.dataset}")
    elif args.cfg_file is None:
        args.cfg_file = "config/semseg_nuscenes.yaml"

    config = generate_config(args.cfg_file)
    if args.resume_path:
        config['resume_path'] = args.resume_path

    print("\n" + "\n".join(list(map(lambda x: f"{x[0]:20}: {x[1]}", config.items()))))
    print("Creating the loaders")
    if config["dataset"].lower() == "nuscenes":
        phase = "verifying" if config['training'] in ("parametrize", "parametrizing") else "val"
        val_dataloader = make_data_loader_nuscenes(
            config, phase, num_threads=config["num_threads"]
        )
    elif config["dataset"].lower() == "kitti":
        val_dataloader = make_data_loader_kitti(
            config, "val", num_threads=config["num_threads"]
        )
    else:
        raise Exception(f"Dataset not recognized: {args.dataset}")
    print("Creating the model")
    model = make_model(config, config["pretraining_path"]).to(0)
    checkpoint = torch.load(config["resume_path"], map_location=torch.device(0))
    if "config" in checkpoint:
        for cfg in ("voxel_size", "cylindrical_coordinates"):
            assert checkpoint["config"][cfg] == config[cfg], (
                f"{cfg} is not consistant.\n"
                f"Checkpoint: {checkpoint['config'][cfg]}\n"
                f"Config: {config[cfg]}."
            )
    try:
        model.load_state_dict(checkpoint["model_points"])
    except KeyError:
        weights = {
            k.replace("model.", ""): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("model.")
        }
        model.load_state_dict(weights)
    # evaluate(model, val_dataloader, config)

    # assign class colors
    if config['model_n_out'] == 17:
        class_colors = ns_class_colors
    else:
        class_colors = sk_class_colors

    model.eval()
    with torch.no_grad():
        i = 0
        full_predictions = []
        ground_truth = []
        for _idx, batch in tqdm(enumerate(val_dataloader)):
            if sum(batch["len_batch"])!=226639: continue
            # if _idx % 10 !=0: continue
            sparse_input = SparseTensor(batch["sinput_F"], batch["sinput_C"], device=0)
            output_points = model(sparse_input).F
            if config["ignore_index"]:
                output_points[:, config["ignore_index"]] = -1e6

            torch.cuda.empty_cache()
            preds = output_points.argmax(1).cpu()
            offset = 0
            for j, lb in enumerate(batch["len_batch"]):
                if j!=0: continue
                str_sign = f'{sum(batch["len_batch"])}_{lb}'
                inverse_indexes = batch["inverse_indexes"][j]
                predictions = preds[inverse_indexes + offset]

                # remove the ignored index entirely
                full_predictions.append(predictions)
                ground_truth.append(deepcopy(batch["evaluation_labels"][j]))

                gt = batch["evaluation_labels"][j] # for vis GT
                pc_coords = batch['pc'][j].detach().cpu()
                # for visualize
                if True:
                    predictions_color = class_colors[predictions,:]
                    gt_color = class_colors[gt,:]
                    # cloud = o3d.geometry.PointCloud()
                    # cloud.points = o3d.utility.Vector3dVector(pc_coords)
                    # cloud.colors = o3d.utility.Vector3dVector(gt_color/255.0) 
                    # o3d.visualization.draw_geometries([cloud])

                    cloud = o3d.geometry.PointCloud()
                    cloud.points = o3d.utility.Vector3dVector(pc_coords)
                    cloud.colors = o3d.utility.Vector3dVector(predictions_color/255.0) 
                    o3d.visualization.draw_geometries([cloud])

                # for visualize error map
                if True:
                    red_color = np.array([255, 0, 0])
                    gray_color = np.array([128, 128, 128])
                    colors = np.tile(gray_color, (pc_coords.shape[0], 1))
                    colors[predictions != gt] = red_color
                    cloud = o3d.geometry.PointCloud()
                    cloud.points = o3d.utility.Vector3dVector(pc_coords)
                    cloud.colors = o3d.utility.Vector3dVector(colors / 255.0)  # 归一化颜色值
                    o3d.visualization.draw_geometries([cloud])
                    # vis = o3d.visualization.Visualizer()
                    # vis.create_window(visible=False)
                    # vis.add_geometry(cloud)
                    # vis.poll_events()
                    # vis.update_renderer()
                    # image = vis.capture_screen_float_buffer(do_render=True)
                    # plt.imsave(f'vis.model_c.ns.skip20/{str_sign}.png', np.asarray(image))
                    # vis.destroy_window()

                print("###", str_sign)
                offset += lb
            i += j


if __name__ == "__main__":
    main()
