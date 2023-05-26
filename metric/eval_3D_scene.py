import numpy as np
from tqdm import tqdm
import trimesh
from metrics import accuracy, completion, completion_ratio
import os
import argparse
import sys
sys.path.insert(1, "/net/projects/ranalab/justindoug/doug/robit/DAD_SLAM")
from cfg import Config

def calc_3d_metric(mesh_rec, mesh_gt, N=200000):
    """
    3D reconstruction metric.
    """
    metrics = [[] for _ in range(4)]
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

    ''' METRIC DESCRIPTIONS:
    
    Accuracy - avg distance b/w points sampled on reconstruction mesh and nearest ground truth point

    Completion - avg distance b/w points sampled on ground truth mesh to nearest reconstruction mesh point

    Completion Ration - pct of points in the reconstruction mesh with completion under 5cm

    '''

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
    log_dir = '/'.join(cfg.output_dir.split('/')[:-1])

    data_dir = "./train_data/vmap/"

    if args.scenes == None:
        exp_name = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]
    else:
        exp_name = args.scenes

    # log_dir = "../logs/iMAP/"
    # log_dir = "./logs/vMAP/" # STORED IN CONFIG FILE NOW

    print("Entering loop")
    for exp in tqdm(exp_name):
        print("Building directories")
        gt_dir = os.path.join(data_dir, exp[:-1]+"_"+exp[-1]) #+"/habitat")
        #os.makedirs(gt_dir, exist_ok=True)
        exp_dir = os.path.join(log_dir, exp)
        mesh_dir = os.path.join(exp_dir, "scene_mesh")
        output_path = os.path.join(exp_dir, "eval_mesh")
        os.makedirs(output_path, exist_ok=True)
        if "vMAP" in exp_dir:
            mesh_list = os.listdir(mesh_dir)
            if "frame_1999_scene.obj" in mesh_list: 
                rec_meshfile = os.path.join(mesh_dir, "frame_1999_scene.obj")
            else: # compose obj into scene mesh
                scene_meshes = []
                for f in mesh_list:
                    _, f_type = os.path.splitext(f)
                    if f_type == ".obj" or f_type == ".ply":
                        obj_mesh = trimesh.load(os.path.join(mesh_dir, f))
                        scene_meshes.append(obj_mesh)
                scene_mesh = trimesh.util.concatenate(scene_meshes)
                scene_mesh.export(os.path.join(mesh_dir, "frame_1999_scene.obj"))
                rec_meshfile = os.path.join(mesh_dir, "frame_1999_scene.obj")
        elif "iMAP" in exp_dir: # obj0 is the scene mesh
            rec_meshfile = os.path.join(mesh_dir, "frame_1999_obj0.obj") # BRAD & DOUG, FIX THIS NAMING CONVENTION FOR NEW OBJ
                                                    # DELETE FRAME A REPLACE WITH FINAL
        else:
            print("Not Implement")
            exit(-1)
        gt_mesh_files = os.listdir(gt_dir)
        gt_mesh_file = os.path.join(gt_dir, "mesh.ply")
        print("Loading ground truth mesh")
        mesh_rec = trimesh.load(rec_meshfile)
        # mesh_rec.invert()   # niceslam mesh face needs invert
        print("Calculating metrics")
        metrics_3D = [[] for _ in range(4)]
        mesh_gt = trimesh.load(gt_mesh_file)
        metrics = calc_3d_metric(mesh_rec, mesh_gt, N=200000)  # for objs use 10k, for scene use 200k points
        metrics_3D[0].append(metrics[0])    # acc 
        metrics_3D[1].append(metrics[1])    # comp
        metrics_3D[2].append(metrics[2])    # comp ratio 1cm
        metrics_3D[3].append(metrics[3])    # comp ratio 5cm
        metrics_3D = np.array(metrics_3D)
        np.save(output_path + '/metrics_3D_scene.npy', metrics_3D)
        print("metrics 3D scene \n Acc | Comp | Comp Ratio 1cm | Comp Ratio 5cm \n ", metrics_3D.mean(axis=1))
        print("-----------------------------------------")
        print("finish exp ", exp)