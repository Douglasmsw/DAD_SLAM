import torch
import model
import embedding
import render_rays
import numpy as np
import vis
from cfg import Config
from tqdm import tqdm
import gc
'''
config_file = './configs/Replica/config_replica_room0_vMAP.json' # DOUG & BRAD FIX THIS JANK
cfg = Config(config_file)       # config params
'''
class Trainer:
    def __init__(self, cfg):
        self.obj_id = cfg.obj_id
        self.device = cfg.training_device
        self.hidden_feature_size = cfg.hidden_feature_size #32 for obj  # 256 for iMAP, 128 for seperate bg
        self.obj_scale = cfg.obj_scale # 10 for bg and iMAP
        self.n_unidir_funcs = cfg.n_unidir_funcs
        self.emb_size1 = 21*(3+1)+3
        self.emb_size2 = 21*(5+1)+3 - self.emb_size1

        self.load_network()

        if self.obj_id == 0:
            self.bound_extent = 0.995
        else:
            self.bound_extent = 0.9

    def load_network(self):
        self.fc_occ_map = model.OccupancyMap(
            self.emb_size1,
            self.emb_size2,
            hidden_size=self.hidden_feature_size
        )
        self.fc_occ_map.apply(model.init_weights).to(self.device)
        print("Pulling embedding from network")
        self.pe = embedding.UniDirsEmbed(max_deg=self.n_unidir_funcs, scale=self.obj_scale).to(self.device)

    def meshing(self, bound, obj_center, grid_dim=256):
        print("\nInitializing scales and bounds")
        occ_range = [-1., 1.]
        range_dist = occ_range[1] - occ_range[0]
        scene_scale_np = bound.extent / (range_dist * self.bound_extent)
        scene_scale = torch.from_numpy(scene_scale_np).float().to(self.device)
        transform_np = np.eye(4, dtype=np.float32)
        transform_np[:3, 3] = bound.center
        transform_np[:3, :3] = bound.R
        # transform_np = np.linalg.inv(transform_np)  #
        transform = torch.from_numpy(transform_np).to(self.device)
        print("\nMaking 3D grid")
        grid_pc = render_rays.make_3D_grid(occ_range=occ_range, dim=grid_dim, device=self.device,
                                           scale=scene_scale, transform=transform).view(-1, 3)
        print("\nShifting grid center")
        grid_pc -= obj_center.to(grid_pc.device)
        print("Evaluating points")
        ret = self.eval_points(grid_pc, object_iter="alpha") # only care about occupancy output built from alpha
        if ret is None:
            return None

        occ, _ = ret
        del _
        gc.collect()
        torch.cuda.empty_cache()
        print("\nMarching cubes")
        mesh = vis.marching_cubes(occ.view(grid_dim, grid_dim, grid_dim).cpu().numpy())
        if mesh is None:
            print("marching cube failed")
            return None

        # Transform to [-1, 1] range
        print("\nTranslating mesh")
        mesh.apply_translation([-0.5, -0.5, -0.5])
        print("\nScaling mesh")
        mesh.apply_scale(2)

        # Transform to scene coordinates
        print("\nScaling scene")
        mesh.apply_scale(scene_scale_np)
        print("\nTransforming scene")
        mesh.apply_transform(transform_np)

        vertices_pts = torch.from_numpy(np.array(mesh.vertices)).float().to(self.device)
        ret = self.eval_points(vertices_pts, object_iter="color") # only care about color from output
        if ret is None:
            return None
        _, color = ret
        del _
        gc.collect()
        torch.cuda.empty_cache()
        mesh_color = color * 255
        print("\nColoring vertices")
        vertex_colors = mesh_color.detach().squeeze(0).cpu().numpy().astype(np.uint8)
        mesh.visual.vertex_colors = vertex_colors

        return mesh

    def eval_points(self, points, chunk_size=100000, object_iter="all"): 
        # 256^3 = 16777216
        alpha, color = [], []

        print(f"\nNum points: {points.shape[0]}")
        n_chunks = int(np.ceil(points.shape[0] / chunk_size))
        print(f"\n{n_chunks=}")
        #print(torch.cuda.memory_summary() )
        with torch.no_grad():
            #for k in tqdm(range(n_chunks)): # 2s/it 1000000 pts # THIS IS THE MEMORY EATER!!!!!! BRAD & DOUG
            for k in range(n_chunks): 
                print(f"\nITERATION K = {k}\n")
                #print(torch.cuda.memory_summary() ) # THERE'S A MEMORY LEAK SOMEHWERE ON GPU???????
                print(f"\nslicing")
                chunk_idx = slice(k * chunk_size, (k + 1) * chunk_size)
                print(f"building embedding")
                embedding_k = self.pe(points[chunk_idx, ...])
                del chunk_idx
                gc.collect()
                torch.cuda.empty_cache()

                print(f"mapping color")
                alpha_k, color_k = self.fc_occ_map(embedding_k)
                del embedding_k

                if object_iter == "alpha": # if exclusively collecting alpha, don't care about color
                    del color_k

                elif object_iter == "color": # if exclusively collecting color, don't care about alpha
                    del alpha_k
                gc.collect()
                torch.cuda.empty_cache()

                if object_iter != "color": # if not exclusively tracking color, extend alpha
                    print(f"extending alpha")
                    alpha.extend(alpha_k.detach().squeeze())
                    print(f"{len(alpha)=}")
                    print(f"{alpha[-1]=}")
                
                elif object_iter != "alpha": # if not exclusively tracking alpha, extend color
                    print(f"extending color")
                    color.extend(color_k.detach().squeeze())
                    print(f"{len(color)=}")
                    print(f"{color[-1]=}")
                
                print(f"Extended")

                print(torch.cuda.memory_summary() )
                if object_iter != "color": # if not exclusively tracking color, clear memory of alpha
                    del alpha_k

                elif object_iter != "alpha": # if not exclusively tracking alpha, clear memory of color
                    del color_k
                gc.collect()
                torch.cuda.empty_cache()

        print(f"stacking")
        if object_iter != "color": # if not exclusively tracking color, stack alpha
            alpha = torch.stack(alpha)
        elif object_iter != "alpha": # if not exclusively tracking aalpha, stack color
            color = torch.stack(color)

        print(f"Occupancy activation")
        if object_iter != "color": # if not exclusively tracking color, apply sigmoid activation to alphas
            occ = render_rays.occupancy_activation(alpha).detach()
            if occ.max() == 0:
                print("no occ")
                return None
        else: # if exclusively tracking color, just return an empty list as occ
            occ = []
        
        del alpha # done with alpha after calculating occ, so delete from memory
        gc.collect()
        torch.cuda.empty_cache()
        
        return (occ, color)











