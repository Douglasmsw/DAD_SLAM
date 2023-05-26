import numpy as np
from tqdm import tqdm
import trimesh
from metrics import accuracy, completion, completion_ratio
import os
import json
from icecream import ic
import sys
sys.path.insert(1, "/net/projects/ranalab/justindoug/doug/robit/DAD_SLAM")
from cfg import Config
import argparse

def calc_3d_metric(mesh_rec, mesh_gt, N=200000):
    """
    3D reconstruction metric.
    """
    metrics = [[] for _ in range(4)]
    transform, extents = trimesh.bounds.oriented_bounds(mesh_gt)
    extents = extents / 0.9  # enlarge 0.9
    box = trimesh.creation.box(extents=extents, transform=np.linalg.inv(transform))
    mesh_rec = mesh_rec.slice_plane(box.facets_origin, -box.facets_normal)
    if mesh_rec.vertices.shape[0] == 0:
        print("no mesh found")
        return
    rec_pc = trimesh.sample.sample_surface(mesh_rec, N)
    rec_pc_tri = trimesh.PointCloud(vertices=rec_pc[0])

    gt_pc = trimesh.sample.sample_surface(mesh_gt, N)
    gt_pc_tri = trimesh.PointCloud(vertices=gt_pc[0])
    accuracy_rec = accuracy(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_rec = completion(gt_pc_tri.vertices, rec_pc_tri.vertices)
    completion_ratio_rec = completion_ratio(gt_pc_tri.vertices, rec_pc_tri.vertices, 0.05)
    completion_ratio_rec_1 = completion_ratio(gt_pc_tri.vertices, rec_pc_tri.vertices, 0.01)

    # accuracy_rec *= 100  # convert to cm
    # completion_rec *= 100  # convert to cm
    # completion_ratio_rec *= 100  # convert to %
    # print('accuracy: ', accuracy_rec)
    # print('completion: ', completion_rec)
    # print('completion ratio: ', completion_ratio_rec)
    # print("completion_ratio_rec_1cm ", completion_ratio_rec_1)
    metrics[0].append(accuracy_rec)
    metrics[1].append(completion_rec)
    metrics[2].append(completion_ratio_rec_1)
    metrics[3].append(completion_ratio_rec)
    return metrics

def get_gt_bg_mesh(gt_dir, background_cls_list):
    with open(os.path.join(gt_dir, "info_semantic.json")) as f:
        label_obj_list = json.load(f)["objects"]

    bg_meshes = []
    for obj in label_obj_list:
        if int(obj["class_id"]) in background_cls_list:
            obj_file = os.path.join(gt_dir, "mesh_semantic.ply_" + str(int(obj["id"])) + ".ply")
            obj_mesh = trimesh.load(obj_file)
            bg_meshes.append(obj_mesh)

    bg_mesh = trimesh.util.concatenate(bg_meshes)
    return bg_mesh

def get_obj_ids(obj_dir):
    files = os.listdir(obj_dir)
    obj_ids = []
    for f in files:
        obj_id = f.split("obj")[1][:-1]
        if obj_id == '':
            continue
        obj_ids.append(int(obj_id))
    return obj_ids


if __name__ == "__main__":
    # setting params
    parser = argparse.ArgumentParser(description='Model training for single GPU')
    parser.add_argument('--config',
                        default='./configs/Replica/config_replica_room0_vMAP.json',
                        type=str)
    parser.add_argument('--scenes',
                        default=None,
                        nargs='+')

    args = parser.parse_args()

    # log_dir = args.logdir
    config_file = args.config
    cfg = Config(config_file)
    log_dir = '/'.join(cfg.output_dir.split('/')[:-1]) # WANT RELEVANT LOG DIRECTORY FROM CONFIG WITHOUT SCENE NAME AT END, IT GETS APPENDED LATER IN SCRIPT
    data_dir = "./train_data/vmap/"

    if args.scenes == None:
        exp_name = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]
    else:
        exp_name = args.scenes

    background_cls_list = [5, 12, 30, 31, 40, 60, 92, 93, 95, 97, 98, 79]

    print("Entering environment loop")
    for exp in tqdm(exp_name):
        print("Building directories")
        gt_dir = os.path.join(data_dir, exp[:-1]+"_"+exp[-1]+"/habitat")
        exp_dir = os.path.join(log_dir, exp)
        mesh_dir = os.path.join(exp_dir, "scene_mesh")
        output_path = os.path.join(exp_dir, "eval_mesh")
        os.makedirs(output_path, exist_ok=True)
        metrics_3D = [[] for _ in range(4)]

        # get obj ids
        # obj_ids = np.loadtxt()    # todo use a pre-defined obj list or use vMAP results
        obj_ids = get_obj_ids(mesh_dir.replace("iMAP", "vMAP"))
        print("Entering object loop")
        for obj_id in tqdm(obj_ids):
            print(f"Processing mesh path for \n\t{exp=}\n\t{obj_id=}")
            if obj_id == 0: # for bg
                N = 200000
                mesh_gt = get_gt_bg_mesh(gt_dir, background_cls_list)
            else:   # for obj
                N = 10000
                obj_file = os.path.join(gt_dir, "mesh_semantic.ply_" + str(obj_id) + ".ply")
                mesh_gt = trimesh.load(obj_file)

            if "vMAP" in exp_dir:
                rec_meshfile = os.path.join(mesh_dir, "final_obj"+str(obj_id)+".obj") # BRAD & DOUG MODIFIED LINES 96 AND 98 FROM FRAME_1999 TO FINAL
            elif "iMAP" in exp_dir:
                rec_meshfile = os.path.join(mesh_dir, "final_obj0.obj")
            else:
                print("Not Implement")
                exit(-1)

            print(f"Loading in mesh for \n\t{exp=}\n\t{obj_id=}")
            mesh_rec = trimesh.load(rec_meshfile)
            # mesh_rec.invert()   # niceslam mesh face needs invert
            print(f"Calculating metrics for \n\t{exp=}\n\t{obj_id=}")
            metrics = calc_3d_metric(mesh_rec, mesh_gt, N=N)  # for objs use 10k, for scene use 200k points
            if metrics is None:
                continue
            np.save(output_path + '/metric_obj{}.npy'.format(obj_id), np.array(metrics))
            metrics_3D[0].append(metrics[0])    # acc
            metrics_3D[1].append(metrics[1])    # comp
            metrics_3D[2].append(metrics[2])    # comp ratio 1cm
            metrics_3D[3].append(metrics[3])    # comp ratio 5cm
        metrics_3D = np.array(metrics_3D)
        np.save(output_path + '/metrics_3D_obj.npy', metrics_3D)
        ic(metrics_3D)
        print("metrics 3D obj \n Acc | Comp | Comp Ratio 1cm | Comp Ratio 5cm \n", metrics_3D.mean(axis=1))
        print("-----------------------------------------")
        print("finish exp ", exp)