import pickle as pkl
import argparse
from cfg import Config
import os
import numpy as np
import open3d
from vmap import *
import shutil
import gc

if __name__ == "__main__":
    # setting params
    parser = argparse.ArgumentParser(description='Model training for single GPU')
    parser.add_argument('--dir', default='./logs/vMAP/room0',
                        type=str)
    parser.add_argument('--config',
                        default='./configs/Replica/config_replica_room0_vMAP.json',
                        type=str)
    parser.add_argument('--item_ids',
                        default=None,
                        nargs='+')

    print("Parsing args")
    print(torch.cuda.memory_summary() )
    args = parser.parse_args()
    
    log_dir = args.dir
    config_file = args.config
    os.makedirs(log_dir, exist_ok=True)  
    shutil.copy(config_file, log_dir)
    cfg = Config(config_file)       # config params

    if args.item_ids != None:  # LINES 33 - 43 ARE MODIFICATIONS
        oberved_ids = args.item_ids
    else:
        oberved_ids = []
        for file_name in os.listdir(f"{log_dir}/vis_items"):
            id_num = file_name.split('_')[1]
            oberved_ids.append(id_num)

    if oberved_ids == None:
        with open(f"{log_dir}/oberved_ids.pickle", "rb") as f:
            oberved_ids = pkl.load(f)

    print("Setting camera")
    print(torch.cuda.memory_summary() )    
    # set camera
    cam_info = cameraInfo(cfg)
    intrinsic_open3d = open3d.camera.PinholeCameraIntrinsic(
        width=cfg.W,
        height=cfg.H,
        fx=cfg.fx,
        fy=cfg.fy,
        cx=cfg.cx,
        cy=cfg.cy)

    print("Entering loop")
    print(torch.cuda.memory_summary() )
    for i_k in oberved_ids: # BRAD & DOUG, NEED TO READ THROUGH DIRECTORY AND MAKE LIST OF ITEM NUMBERS
        print(torch.cuda.memory_summary() )
        print(f"\nLOADING IN VIS_DICT ITEM {i_k}")
        with open(f"{log_dir}/vis_items/item_{i_k}_obj.pickle", "rb") as f:
            obj_id, obj_k = pkl.load(f)
            # BRAD & DOUG: SAVE EACH ITEM AS A SEPARATE PICKLE FILE THEN LOAD SEQUENTIALLY!!!

        #print(torch.cuda.memory_summary()) 

        print("Getting object bound")
        # print(torch.cuda.memory_summary() )
        bound = obj_k.get_bound(intrinsic_open3d)
        if bound is None:
            print("get bound failed obj ", obj_id)
            continue
        print("Calculating adaptive grid")
        adaptive_grid_dim = int(np.minimum(np.max(bound.extent)//cfg.live_voxel_size+1, cfg.grid_dim)) 
        print(f"{adaptive_grid_dim=}")
        print(f"\nBuilding mesh for {obj_id=}")
        mesh = obj_k.trainer.meshing(bound, obj_k.obj_center, grid_dim=adaptive_grid_dim)
        del bound
        del obj_k
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Mesh built for {obj_id=}")
        if mesh is None:
            print("meshing failed obj ", obj_id)
            continue

        # save to dir
        obj_mesh_output = os.path.join(log_dir, "scene_mesh")
        os.makedirs(obj_mesh_output, exist_ok=True)

        print(f"\nExporting mesh at {i_k=}\n")
        mesh.export(os.path.join(obj_mesh_output, "final_obj{}.obj".format(str(obj_id))))
        print(f"\nSuccessfully exported mesh")

    print("\nPROCESS COMPLETE\n")

