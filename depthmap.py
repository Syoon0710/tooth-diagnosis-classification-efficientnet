import json
import codecs
import os
import glob
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
import meshio
import mesh_to_depth as m2d


def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def adj_nums(num):
    if num % 10 == 1:
        if num // 10 in [1, 3]:
            return (num + 10, num + 1)
        else:
            return (num - 10, num + 1)
    else:
        return (num - 1, num + 1)

def is_in_ccw(n1, n2):
    if n1//10 < n2//10:
        return True
    elif n1//10 == n2//10:
        return n1 > n2 if n1//10 in [1, 3] else n1 < n2
    else:
        return False



def generate_depthmap(verts, tris, cam_pos, cam_lookat, cam_up):
    
    param = {
            'cam_pos': cam_pos, 'cam_lookat': cam_lookat, 'cam_up': cam_up,
            'x_fov': 1,  # End-to-end field of view in radians
            'near': 0.1, 'far': 30,
            'height': 224, 'width': 224,
            'is_depth': True,
        }
    
    depth_map = m2d.mesh2depth(verts.astype(np.float32), tris.astype(np.uint32), [param], empty_pixel_value=np.nan)[0]
    depth_map = (np.nanmax(depth_map) - depth_map)
    depth_map = 255 * depth_map / np.nanmax(depth_map)
    
    return depth_map


def create_jaw_depthmap(filename: str, json_data, is_upper_jaw: bool):

    # read mesh
    mesh = meshio.read(filename)
    mesh_centroid = mesh.points[:,:3].mean(axis=0)

    # get gids
    with open(filename) as f:
        group_ids = [int(line[9:]) for line in f.readlines() if line[0] == 'g']

    # compute 'up' vector (root -> crown)
    up = normalize(mesh.point_data['obj:vn'].mean(axis=0))

    # compute 'forward' vector (throat -> mouth)
    pca = PCA(n_components=3).fit(mesh.points[:,:3])
    forward = pca.components_[1]

    # create depth map for each tooth
    for tooth in json_data['tooth']:

        if tooth['number'] % 10 == 8:
            continue

        # skip teeth on the opposite jaw
        if is_upper_jaw and tooth['number'] > 30 or not is_upper_jaw and tooth['number'] < 30:
            continue
        
        # skip teeth with treatments or diseases other than abrasion
        diagnosis = 'Normal' if len(tooth['diag']) <= 0 else tooth['diag'][0]
        if len(tooth['treat']) > 0 or diagnosis not in ['Normal', 'Abrasion']:
            continue

        # extract tooth info
        try:
            gid = group_ids.index(tooth['label'])
        except ValueError:
            continue
        tooth_tris = mesh.cells[gid].data
        tooth_verts = mesh.points[np.unique(tooth_tris), :3]
        tooth_centroid = tooth_verts.mean(axis=0)
        
        # set camera base position according to its tooth number
        base_pos = mesh_centroid
        if (tooth['number'] % 10) > 5:  
            base_pos += np.dot(forward, tooth_centroid - mesh_centroid) * forward

        # compute camera positions
        # ori = normalize(tooth_centroid - base_pos)
        # ori = ori - np.dot(ori, up) * up

        adj_teeth = [t for t in json_data['tooth'] if t['number'] in adj_nums(tooth['number'])]

        if len(adj_teeth) == 0:
            continue
        
        if len(adj_teeth) == 1:
            adj_teeth.append(tooth)
        
        if not is_in_ccw(adj_teeth[0]['number'], adj_teeth[1]['number']):
            adj_teeth.reverse()
        

        keypoints = np.array([t['keypoint'] for t in adj_teeth]).astype(float)
        
        if np.isnan(keypoints).any():
            continue

        ori = normalize(np.cross(keypoints[0] - keypoints[1], up))

        facial_cam = tooth_centroid + ori * 15
        up_cam = tooth_centroid + up * 15
        lingual_cam = tooth_centroid - ori * 15

        filter_tris = False
        
        if filter_tris:
            # filter triangles to display
            tooth_tri_norms =  mesh.point_data['obj:vn'][tooth_tris[:, 0], :]
            facial_tris = tooth_tris[np.dot(tooth_tri_norms, ori) > 0, :]
            up_tris = tooth_tris
            lingual_tris = tooth_tris[np.dot(tooth_tri_norms, -ori) > 0, :]
            root_dir = 'out'
        else:
            facial_tris = up_tris = lingual_tris = mesh.cells_dict['triangle']
            root_dir = 'out_full'

        # generate depth maps
        facial_depth_map = generate_depthmap(mesh.points[:, :3], facial_tris, facial_cam, tooth_centroid, up)
        up_depth_map = generate_depthmap(mesh.points[:, :3], up_tris, up_cam, tooth_centroid, ori)
        lingual_depth_map = generate_depthmap(mesh.points[:, :3], lingual_tris, lingual_cam, tooth_centroid, up)
        depth_maps = np.concatenate([facial_depth_map, up_depth_map, lingual_depth_map], axis=1)
        
        # save image
        Path(f"{root_dir}/{diagnosis}/").mkdir(parents=True, exist_ok=True)
        # Image.fromarray(depth_maps).convert("L").save(f"{root_dir}/{diagnosis}/{filename[10:-8]}_{tooth['number']}.png")
        Image.fromarray(depth_maps).convert("L").save(f"{root_dir}/{diagnosis}/{filename[6:-10]}_{tooth['number']}.png")

for filename in os.listdir('json'):

    with open(f'json/{filename}', 'r') as f:
        json_data = json.load(f)

    # print(f"{filename}: {len(json_data['tooth'])}")

    # skip scans without abrasion
    # if not any([len(t['diag']) > 0 and t['diag'][0] == 'Abrasion' for t in json_data['tooth']]):
    #     continue
    
    # scan data filenames
    up, low = json_data['scandata']['up'], json_data['scandata']['low']
    upper_jaw, lower_jaw = f"scans/{up}", f"scans/{low}"
    # upper_jaw, lower_jaw = f"scans/{up[:-4]}-C{up[-4:]}", f"scans/{low[:-4]}-C{low[-4:]}"
    
    
    if os.path.exists(upper_jaw):
        create_jaw_depthmap(upper_jaw, json_data, is_upper_jaw=True)
    if os.path.exists(lower_jaw):
        create_jaw_depthmap(lower_jaw, json_data, is_upper_jaw=False)
