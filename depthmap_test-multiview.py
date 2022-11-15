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
    mesh_centroid = mesh.points[:,:3].mean(axis=0) # 중심 좌표
    
    # get gids
    with open(filename) as f:
        group_ids = [int(line[9:]) for line in f.readlines() if line[0] == 'g']

    # if OBJ has no "vn" data, calculate vn for each vertex --------   
    if not mesh.point_data:
        vn=np.zeros((len(mesh.points),3), dtype=float)
        # a face has 3 vertices : f fv1 fv2 fv3
        #calc face normal vector
        f=mesh.cells_dict['triangle']
        for j in range(len(f)):
            av = mesh.points[f[j,1], 0:3] - mesh.points[f[j,0], 0:3]
            bv = mesh.points[f[j,2], 0:3] - mesh.points[f[j,0], 0:3]
            cv = np.cross(av, bv)
            nv = cv / np.linalg.norm(cv)
            #add face noraml to vertex normal vector
            vn[f[j,0]]=vn[f[j,0]] + nv
            vn[f[j,1]]=vn[f[j,1]] + nv
            vn[f[j,2]]=vn[f[j,2]] + nv
        #calc vn by normalizing vn
        for i in range(len(vn)):
            vn[i]=vn[i]/np.linalg.norm(vn[i])
        mesh.point_data['obj:vn']=vn
    # -------------------------------------------------------------
 
    # compute 'up' vector (root -> crown)
    up = normalize(mesh.point_data['obj:vn'].mean(axis=0))
    

    # compute 'forward' vector (throat -> mouth)
    pca = PCA(n_components=3).fit(mesh.points[:,:3])
    forward = pca.components_[1]
    i=0
    # create depth map for each tooth
    for tooth in json_data['tooth']:

        if tooth['number'] == 0:
            continue

        if tooth['number'] % 10 == 8: # 8번치아는 제외
            continue

        # skip teeth on the opposite jaw
        if is_upper_jaw and tooth['number'] > 30 or not is_upper_jaw and tooth['number'] < 30: # 치아번호와 상악 하악이 매칭이 안되는 경우 제외
            continue

        # skip teeth with treatments or diseases other than abrasion
        diagnosis = 'Normal' if len(tooth['diag']) <= 0 else tooth['diag'][0] # 치료된 치아 또는 치아의 질병이 이미 normal 또는 abrasion 이라고 표시되어있는 치아는 제외
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
        wdist = 15  # distance from CAM to TOOTH
        dangle = 5  # Cam pos change step (angle)
        dpos = wdist * dangle*(3.14/180)  # cam pos change step (mm)
        dtr = normalize(np.cross(up, ori))  # transverse pos change step for all cam
        dr1 = up    # rotational pos cahnge step for facial_ and ligual_ cam
        dr2 = ori   # rotational pos cahnge pos step for up_cam
        
        cam_pos_list = [[0,0],[1,0],[-1,0],[0,1],[0,-1]]   # cam pos list
        for cp in cam_pos_list:
            
            facial_cam = tooth['keypoint'] + ori * wdist + dtr*cp[0] + dr1*cp[1]
            up_cam = tooth['keypoint'] + up * wdist + dtr*cp[0] + dr2*cp[1]
            lingual_cam = tooth['keypoint'] - ori * wdist + dtr*cp[0] + dr1*cp[1]
    
            # filter_tris = False 
            # # True = 잇몸이 안나옴
            # # False = 잇몸까지 나옴
            # if filter_tris:
            #     # filter triangles to display
            #     tooth_tri_norms =  mesh.point_data['obj:vn'][tooth_tris[:, 0], :]
            #     facial_tris = tooth_tris[np.dot(tooth_tri_norms, ori) > 0, :]
            #     up_tris = tooth_tris
            #     lingual_tris = tooth_tris[np.dot(tooth_tri_norms, -ori) > 0, :]
            # else:
            facial_tris = up_tris = lingual_tris = mesh.cells_dict['triangle']
            root_dir = 'out_full'
    
            # generate depth maps
            facial_depth_map = generate_depthmap(mesh.points[:, :3], facial_tris, facial_cam, tooth['keypoint'], up)
            up_depth_map = generate_depthmap(mesh.points[:, :3], up_tris, up_cam, tooth['keypoint'], ori)
            lingual_depth_map = generate_depthmap(mesh.points[:, :3], lingual_tris, lingual_cam, tooth['keypoint'], up)
            depth_maps = np.concatenate([facial_depth_map, up_depth_map, lingual_depth_map], axis=1)
            
            # save image
            Path(f"{root_dir}/{diagnosis}/").mkdir(parents=True, exist_ok=True)
            # Image.fromarray(depth_maps).convert("L").save(f"{root_dir}/{diagnosis}/{filename[10:-8]}_{tooth['number']}.png")
            file_name = filename[6:].split(".")[0]
            Image.fromarray(depth_maps).convert("L").save(f"{root_dir}/{diagnosis}/{file_name}_{tooth['number']}_{cp[0]}{cp[1]}.png")
            # print("finish")
        
for filename in os.listdir('json'):

    with open(f'json/{filename}', 'r', encoding='utf-8-sig') as f:
        json_data = json.load(f)

    # print(f"{filename}: {len(json_data['tooth'])}")

    # skip scans without abrasion
    if not any([len(t['diag']) > 0 and t['diag'][0] == 'Abrasion' for t in json_data['tooth']]):
        continue
    
    # scan data filenames
    up, low = json_data['scandata']['up'], json_data['scandata']['low']
    # upper_jaw, lower_jaw = f"scans1/{up}", f"scans1/{low}"
    upper_jaw, lower_jaw = f"scans/{up}", f"scans/{low}"
    
    if os.path.exists(upper_jaw):
        create_jaw_depthmap(upper_jaw, json_data, is_upper_jaw=True)
    if os.path.exists(lower_jaw):
        create_jaw_depthmap(lower_jaw, json_data, is_upper_jaw=False)
