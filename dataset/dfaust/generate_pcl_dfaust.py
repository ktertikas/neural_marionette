from argparse import ArgumentParser
import logging
import numpy as np
from pathlib import Path
import trimesh

# Disable trimesh's logger
logging.getLogger("trimesh").setLevel(logging.ERROR)

SIDS = ["50002", "50004", "50007", "50009", "50020", "50021", "50022", "50025", "50026", "50027"]
MOVEMENTS = [
    "chicken_wings",
    "hips",
    "jiggle_on_toes",
    "jumping_jacks",
    "knees",
    "light_hopping_loose",
    "light_hopping_stiff",
    "one_leg_jump",
    "one_leg_loose",
    "punching",
    "running_on_spot",
    "shake_arms",
    "shake_hips",
    "shake_shoulders",
]


def sample_faces(mesh, N=20000):
    P, t = trimesh.sample.sample_surface(mesh, N)
    sampled_faces = np.hstack([P, mesh.face_normals[t, :]]).astype(np.float32)
    return sampled_faces  # (N, 3 + 3)


if __name__ == "__main__":
    parser = ArgumentParser(description="Parse DFAUST meshes and sample points on mesh")
    parser.add_argument(
        "--dfaust_path",
        type=str,
        default=None,
        required=True,
        help="DFAUST base path containing folders of the sequence meshes",
    )
    parser.add_argument(
        "--target_path",
        type=str,
        default=None,
        required=True,
        help="The target path in which to save the sampled point cloud sequences",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=20000,
        help="The number of points to be sampled. Defaults to 20000",
    )
    args = parser.parse_args()

    dfaust_path = Path(args.dfaust_path)
    target_path = Path(args.target_path)

    print(f"Creating {target_path} if it doesn't exist")
    target_path.mkdir(exist_ok=True)

    for sid in SIDS:
        target_sid_path = target_path / "surface" / sid
        target_sid_path.mkdir(exist_ok=True, parents=True)
        for seq in MOVEMENTS:
            mesh_path_str = f"{sid}_{seq}"
            full_mesh_path = dfaust_path / mesh_path_str
            if not full_mesh_path.is_dir():
                print(f"{full_mesh_path} does not exist, moving on...")
                continue
            # loading all meshes in the path and sampling points
            sampled = []
            for mesh_path in sorted(full_mesh_path.glob("*.obj")):
                mesh = trimesh.load(mesh_path, process=False)
                sampled.append(sample_faces(mesh, N=args.num_points))
            sampled = np.stack(sampled, axis=0)
            target_pcl_save_path = f"{target_sid_path}/{seq}.npy"
            print(f"Saving pointcloud in path {target_pcl_save_path}")
            np.save(target_pcl_save_path, sampled)
