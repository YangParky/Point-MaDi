import numpy as np

from plyfile import PlyData, PlyElement


def merge_ply_files(mask_path, visible_path, output_path):
    try:
        mask_ply = PlyData.read(mask_path)
        mask_vertices = np.array(mask_ply['vertex'].data)
        mask_points = np.vstack([mask_vertices['x'], mask_vertices['y'], mask_vertices['z']]).T
    except Exception as e:
        print(f"Error reading {mask_path}: {e}")
        return

    try:
        visible_ply = PlyData.read(visible_path)
        visible_vertices = np.array(visible_ply['vertex'].data)
        visible_points = np.vstack([visible_vertices['x'], visible_vertices['y'], visible_vertices['z']]).T
    except Exception as e:
        print(f"Error reading {visible_path}: {e}")
        return

    merged_points = np.concatenate([mask_points, visible_points], axis=0)
    print(
        f"Merged {len(mask_points)} masked points and {len(visible_points)} visible points into {len(merged_points)} points")

    vertex = np.array(
        [(p[0], p[1], p[2]) for p in merged_points],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    )
    el = PlyElement.describe(vertex, 'vertex')

    try:
        PlyData([el], text=True).write(output_path)
        print(f"Saved merged point cloud to {output_path}")
    except Exception as e:
        print(f"Error writing {output_path}: {e}")


def main():
    mask_path = './vis_util/rand_masked_center.ply'
    visible_path = './vis_util/rand_visible_center.ply'
    output_path = './vis_util/rand_center.ply'

    merge_ply_files(mask_path, visible_path, output_path)


if __name__ == "__main__":
    main()