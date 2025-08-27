import os
import json
from utils import read_pcd_file, write_bin_file, filter_points_in_box, read_lidar2camera, \
                  read_camera_intrinsic, project_to_image, generate_dynamic_mask, \
                  get_image_size, imu2ego, load_extrinsic_yaml, load_lidar2camera_yaml, \
                  load_intrinsic_yaml, load_camera_intrinsic_yaml, project_points_to_image, \
                  draw_and_fill_box, find_track_id_frame, find_track_id_obj2world, \
                  find_track_id_boxsize, convert_ndarray_to_list
from PIL import Image
import cv2
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from scipy.spatial.transform import Rotation
from tqdm import tqdm

source_dir = "/data/gls/code/drivestudio/data/chery_data"
target_dir = "/data/gls/code/drivestudio/data/waymo_chery/processed/training"

### finish 
def generate_dynamic_masks():
    ### 需要雷达的box投影到2d上
    ### all human vehicle三个文件夹
    ### 从data/chery_data/clip_1746752396800/dynamic_obj/autolabel_10hz/clip_1746752396800.json读取
    ### "category" 是person的对应human，其余的都是vehicle
    ### 找到映射后处理
    
    for index, clip_name in enumerate(os.listdir(source_dir)):
        ex_dir_all = os.path.join(target_dir, f"{index:03}", "dynamic_masks", "all")
        ex_dir_human = os.path.join(target_dir, f"{index:03}", "dynamic_masks", "human")
        ex_dir_vehicle = os.path.join(target_dir, f"{index:03}", "dynamic_masks", "vehicle")
        
        if not os.path.exists(ex_dir_all): os.makedirs(ex_dir_all)
        if not os.path.exists(ex_dir_human): os.makedirs(ex_dir_human)
        if not os.path.exists(ex_dir_vehicle): os.makedirs(ex_dir_vehicle)
                
        source_data_path = os.path.join(source_dir, f"{clip_name}/dynamic_obj/autolabel_10hz/{clip_name}.json")
        # 读取JSON文件
        with open(source_data_path, 'r') as file:
            data = json.load(file)
        
        ### 从对应的sample里读取雷达数据
            
        for frame_index, params in tqdm(enumerate(data['frames'])):
            ### 从对应的sample里读取对应帧的雷达数据
            frame_name = params['frame_name']
            
            object_detection_anns_info = params['annotated_info']['3d_city_object_detection_annotated_info']['annotated_info']['3d_object_detection_info']['3d_object_detection_anns_info']
            ### 把lidar2camera的外参按顺序读取到
            lidar2camera = read_lidar2camera(os.path.join(source_dir, clip_name, "extrinsics", "lidar2camera"))
            
            camera_intrinsic = read_camera_intrinsic(os.path.join(target_dir, f"{index:03}", "intrinsics"))  # cam2img
            images_size = []
            for i in range(len(lidar2camera)):
                image_size = get_image_size(os.path.join(target_dir, f"{index:03}", "images", f"{frame_index:03}_{i}.jpg"))
                images_size.append(image_size)
            images_vehicle = []
            for i in range(len(lidar2camera)):
                image = np.zeros((images_size[i][0], images_size[i][1], 3), dtype=np.uint8)
                images_vehicle.append(image)
            images_human = []
            for i in range(len(lidar2camera)):
                image = np.zeros((images_size[i][0], images_size[i][1], 3), dtype=np.uint8)
                images_human.append(image)                          
            tag_person = 0
            tag_car = 0
            ### params是一帧
            for each_object in object_detection_anns_info:  ### 对一帧里面的每个物体
                if each_object['category'] == "person":
                    tag_person = 1
                    l, w, h = each_object['size']  # ann_box is one of dynamic objs
                    box2lidar = np.eye(4, dtype=np.float64)   ### box2lidar是每个实例的外参
                    box2lidar[:3, :3] = Rotation.from_quat(np.array(each_object['obj_rotation'])).as_matrix() # xyzw
                    box2lidar[:3, 3] = np.array(each_object['obj_center_pos'])
                    ### 1.对任何一个物体，先对应到激光雷达点云
                    size = each_object['size']
                    obj_center_pos = each_object['obj_center_pos']
                    # mask_pointcloud = filter_points_in_box(pointcloud, obj_center_pos, size)
                    ### 2.把激光点云向7个视角均投影得到uv
                    ### 3.七个视角最终全部投影得到结果
                    ### 4.把七个视角结果按照000_0的格式写到ex_dir_vehicle中
                    for i in range(len(lidar2camera)):
                        box2cam = lidar2camera[i] @ box2lidar  ## 变换矩阵的乘法是从右向左应用
                        qx, qy, qz, qw = Rotation.from_matrix(box2cam[:3, :3]).as_quat()
                        cam_box = Box( ## from nuscenes.utils.data_classes import Box
                            center=box2cam[:3, 3], size=[w, l, h], 
                            orientation=Quaternion([qw, qx, qy, qz]), ## from pyquaternion import Quaternion
                            name=each_object['category'])
                        corners = cam_box.corners().T.astype(np.float32) # 8x3
                        points2d = project_points_to_image(corners, camera_intrinsic[i])
                        
                        # 转换点云到图像坐标系
                        # points_image = project_to_image(pointcloud, lidar2camera[i], camera_intrinsic[i])
                        # 生成dynamic_mask标签
                        image_with_box = generate_dynamic_mask(points2d, image_size, images_human[i])
                        # 保存标签图像
                        filename = f"{frame_index:03}_{i}.png"
                        cv2.imwrite(os.path.join(ex_dir_human, filename), image_with_box)
                        cv2.imwrite(os.path.join(ex_dir_all, filename), image_with_box)
                        
                else:
                    tag_car = 1
                    l, w, h = each_object['size']  # ann_box is one of dynamic objs
                    box2lidar = np.eye(4, dtype=np.float64)
                    box2lidar[:3, :3] = Rotation.from_quat(np.array(each_object['obj_rotation'])).as_matrix() # xyzw
                    box2lidar[:3, 3] = np.array(each_object['obj_center_pos'])
                    ### 1.对任何一个物体，先对应到激光雷达点云
                    size = each_object['size']
                    obj_center_pos = each_object['obj_center_pos']
                    # mask_pointcloud = filter_points_in_box(pointcloud, obj_center_pos, size)
                    ### 2.把激光点云向7个视角均投影得到uv
                    ### 3.七个视角最终全部投影得到结果
                    ### 4.把七个视角结果按照000_0的格式写到ex_dir_vehicle中
                    for i in range(len(lidar2camera)):
                        box2cam = lidar2camera[i] @ box2lidar
                        qx, qy, qz, qw = Rotation.from_matrix(box2cam[:3, :3]).as_quat()
                        cam_box = Box( ## from nuscenes.utils.data_classes import Box
                            center=box2cam[:3, 3], size=[w, l, h], 
                            orientation=Quaternion([qw, qx, qy, qz]), ## from pyquaternion import Quaternion
                            name=each_object['category'])
                        corners = cam_box.corners().T.astype(np.float32) # 8x3
                        points2d = project_points_to_image(corners, camera_intrinsic[i])
                        
                        # 转换点云到图像坐标系
                        # points_image = project_to_image(pointcloud, lidar2camera[i], camera_intrinsic[i])
                        # 生成dynamic_mask标签
                        image_with_box = generate_dynamic_mask(points2d, image_size, images_vehicle[i])
                        # 保存标签图像
                        filename = f"{frame_index:03}_{i}.png"
                        cv2.imwrite(os.path.join(ex_dir_vehicle, filename), image_with_box)
                        cv2.imwrite(os.path.join(ex_dir_all, filename), image_with_box)

                if tag_person == 0:
                    # 没有人，全部人填空
                    for i in range(len(lidar2camera)):
                        image = np.zeros((images_size[i][0], images_size[i][1], 3), dtype=np.uint8)
                        # 保存标签图像
                        filename = f"{frame_index:03}_{i}.png"
                        cv2.imwrite(os.path.join(ex_dir_human, filename), image)
                if tag_car == 0:
                    # 没有车，全部车填空
                    for i in range(len(lidar2camera)):
                        image = np.zeros((images_size[i][0], images_size[i][1], 3), dtype=np.uint8)
                        # 保存标签图像
                        filename = f"{frame_index:03}_{i}.png"
                        cv2.imwrite(os.path.join(ex_dir_vehicle, filename), image)
### finish
### ego_pose==ego2world
'''
    方案：lidar2imu + imu2ego→lidar2ego→ego2lidar + lidar2world → ego2world
'''
def generate_ego_pose():  ## ego2world
    ### ego_pose指的是自车相对于世界坐标系的位姿
    for index, clip_name in enumerate(os.listdir(source_dir)):
        ex_dir = os.path.join(target_dir, f"{index:03}", "ego_pose")
        if not os.path.exists(ex_dir): os.makedirs(ex_dir)
        clip_dir = os.path.join(source_dir, clip_name)
        # Lidar2imu
        lidar2imu = load_extrinsic_yaml(
            f"{clip_dir}/extrinsics/lidar2imu/lidar2imu.yaml"
        )

        # Lidar2cam
        lidar2cam = load_lidar2camera_yaml(
            f"{clip_dir}/extrinsics/lidar2camera/lidar2frontwide.yaml"
        )
        lidar2ego = imu2ego() @ lidar2imu
        ego2lidar = np.linalg.inv(lidar2ego)
        
        source_data_path = os.path.join(source_dir, f"{clip_name}/dynamic_obj/autolabel_10hz/{clip_name}.json")
        # 读取JSON文件
        with open(source_data_path, 'r') as file:
            data = json.load(file)
            
        for frame_index, params in enumerate(data['frames']):
            ### 从对应的sample里读取对应帧的雷达数据
            lidar_pose = params['lidar_pose'] # lidar_pose指的是什么？
            lidar2world = lidar_pose
            ego2world = lidar2world @ ego2lidar  ### fream里面的lidar_pose代表lidar2world
            # 格式化矩阵为指定的字符串格式
            formatted_matrix = "\n".join(
                " ".join(f"{num:.15e}" for num in row)
                for row in ego2world
            )
            text_path = os.path.join(ex_dir, f"{frame_index:03d}.txt")
            # 写入到以索引命名的文本文件中
            with open(text_path, 'w') as outfile:
                outfile.write(formatted_matrix)
### finish
'''
    方案：extrinsics直接是相机外参，从dynamic_obj.json中直接读取
        dynamic_obj的外参应该表示cam2ego，和waymo的extrinsics一致
'''
def generate_extrinsics():
    for index, clip_name in enumerate(os.listdir(source_dir)):
        ex_dir = os.path.join(target_dir, f"{index:03}", "extrinsics")
        if not os.path.exists(ex_dir): os.makedirs(ex_dir)
        source_data_path = os.path.join(source_dir, f"{clip_name}/dynamic_obj/autolabel_10hz/{clip_name}.json")
        # 读取JSON文件
        with open(source_data_path, 'r') as file:
            data = json.load(file)

        # 遍历calibration中的每个相机
        for camera, params in data['calibration'].items():
            if not camera.startswith("camera"):
                continue
            camera_index = ''.join(filter(str.isdigit, camera))
            # 获取extrinsic矩阵
            extrinsic = params['extrinsic']
            
            # 格式化矩阵为指定的字符串格式
            formatted_matrix = "\n".join(
                " ".join(f"{num:.15e}" for num in row)
                for row in extrinsic
            )
            text_path = os.path.join(ex_dir, f"{int(camera_index)}.txt")
            # 写入到以索引命名的文本文件中
            with open(text_path, 'w') as outfile:
                outfile.write(formatted_matrix)
### None                
def generate_fine_dynamic_masks():
    pass
### None
def generate_humanpose():
    pass
### finish
def generate_images():
    for index, clip_name in enumerate(os.listdir(source_dir)):
        ex_dir = os.path.join(target_dir, f"{index:03}", "images")
        if not os.path.exists(ex_dir): os.makedirs(ex_dir)
        directory = os.path.join(source_dir, clip_name)
        all_items = os.listdir(directory)
        sample_folders = [item for item in all_items if os.path.isdir(os.path.join(directory, item)) and item.startswith('sample')]
        # Sort the folders by the trailing numbers
        sample_folders.sort(key=lambda x: int(x.split('_')[-1]))
        for time_index, sample in enumerate(sample_folders):
            path = os.path.join(directory, sample)
            camera_name = [name for name in os.listdir(path) if name.startswith('camera') and name.split(".")[0].split("_")[-1]=="undist"]
            for i in camera_name:
                camera_path = os.path.join(path, i) ### camera0_undist
                camera_number = i.split('_')[0][6:]  # Camera number after 'camera'
                ### to do 打开这个imges然后根据time_index保存到000_0, 000_1
                save_jpg_name = f"{time_index:03}" + "_" + f"{camera_number}" + ".jpg"
                save_jpg_path = os.path.join(ex_dir, save_jpg_name)
                image = Image.open(camera_path)
                image.save(save_jpg_path)          
import time
## finish
def generate_instances():
    for index, clip_name in enumerate(os.listdir(source_dir)):

        ex_dir = os.path.join(target_dir, f"{index:03}", "instances")
        ex_dir_extrinsics = os.path.join(target_dir, f"{index:03}", "extrinsics")
        
        if not os.path.exists(ex_dir): os.makedirs(ex_dir)
        target_frame_instances_json_path = os.path.join(ex_dir, "frame_instances.json")
        target_instances_info_json_path = os.path.join(ex_dir, "instances_info.json")
        '''frame_instances.json'''        
        source_data_path = os.path.join(source_dir, f"{clip_name}/dynamic_obj/autolabel_10hz/{clip_name}.json")
        # source_data_path = os.path.join(source_dir, f"{clip_name}/dynamic_obj/autolabel_10hz/clip_1_frame.json")
        # 读取JSON文件
        with open(source_data_path, 'r') as file:
            data = json.load(file)
        result = {}
        track_id_category = {}
        # 遍历calibration中的每个相机
        for frame_index, frame_data in enumerate(data['frames']):
            track_id_list = []
            object_detection_anns_info = frame_data.get('annotated_info', {}).get(
                '3d_city_object_detection_annotated_info', {}
            ).get('annotated_info', {}).get('3d_object_detection_info', {}).get(
                '3d_object_detection_anns_info', [])
            for obj in object_detection_anns_info:
                track_id = obj.get('track_id')
                # print("track_id:", track_id)
                # time.sleep(1000)
                category = obj.get('category')
                if track_id is not None and track_id not in track_id_list:
                    track_id_list.append(track_id)
                ### 建立一个track_id和category的映射
                if track_id not in track_id_category:
                    track_id_category[str(track_id)] = category
                    
            result[str(frame_index)] = track_id_list
        ### 把result写到target_frame_instances_json_path
        with open(target_frame_instances_json_path, 'w') as outfile:
            json.dump(result, outfile, indent=4)
        print("start: instances_info")
        '''instances_info.json'''
        
        result_instance_info = {}
        ### result["实例编号"]["frame_annotations"]
        #                                           ["frame_idx"] ["obj_to_world 4x4"] ["box_size 三维"]
        # frame_idx可以反投影上面的result
        # box_size是clip的size属性
        # obj_to_world 每一个实例的旋转？？？
        for track_id, category in track_id_category.items():
            print("type:", type(track_id))
            frame_list = find_track_id_frame(track_id, result)
            print("frame_list:", frame_list)
            result_instance_info[track_id] = {}
            if 'frame_annotations' not in result_instance_info[track_id]:
                result_instance_info[track_id]['frame_annotations'] = {}
            if 'frame_idx' not in result_instance_info[track_id]['frame_annotations']:
                result_instance_info[track_id]['frame_annotations']['frame_idx'] = {}
            result_instance_info[track_id]['frame_annotations']['frame_idx'] = frame_list
            result_instance_info[track_id]['id'] = "track_" + track_id
            if track_id_category[track_id] == "person":
                result_instance_info[track_id]['class_name'] = "Pedestrian"
            else:
                result_instance_info[track_id]['class_name'] = "Vehicle"
            obj2world_list = find_track_id_obj2world(track_id, data, frame_list, source_dir, clip_name, ex_dir_extrinsics)
            result_instance_info[track_id]['frame_annotations']['obj_to_world'] = obj2world_list
            box_size_list = find_track_id_boxsize(track_id, data, frame_list)
            result_instance_info[track_id]['frame_annotations']['box_size'] = box_size_list
        result_instance_info = convert_ndarray_to_list(result_instance_info)
        with open(target_instances_info_json_path, 'w') as outfile:
            json.dump(result_instance_info, outfile, indent=4)
            
### finish
def generate_intrinsics():
    for index, clip_name in enumerate(os.listdir(source_dir)):
        ex_dir = os.path.join(target_dir, f"{index:03}", "intrinsics")
        if not os.path.exists(ex_dir): os.makedirs(ex_dir)
        source_data_path = os.path.join(source_dir, f"{clip_name}/dynamic_obj/autolabel_10hz/{clip_name}.json")
        # 读取JSON文件
        with open(source_data_path, 'r') as file:
            data = json.load(file)

        # 遍历calibration中的每个相机
        for camera, params in data['calibration'].items():
            if not camera.startswith("camera") or int(camera[6:]) >= 7:
                continue
            camera_index = ''.join(filter(str.isdigit, camera))
            # 获取intrinsic矩阵
            intrinsic = params['intrinsic']
            fx = intrinsic[0][0]
            cx = intrinsic[0][2]
            fy = intrinsic[1][1]
            cy = intrinsic[1][2]
            distcoeff_8 = params['distcoeff'][0]
            k1,k2,k3,k4,k5,k6,p1,p2 = distcoeff_8
            distcoeff_5 = [k1, k2, p1, p2, k3]
            
            # 格式化矩阵为单列字符串格式
            formatted_values = [
                f"{fx:.15e}",
                f"{fy:.15e}",
                f"{cx:.15e}",
                f"{cy:.15e}",
                f"{distcoeff_5[0]:.15e}",
                f"{distcoeff_5[1]:.15e}",
                f"{distcoeff_5[2]:.15e}",
                f"{distcoeff_5[3]:.15e}",
                f"{distcoeff_5[4]:.15e}"
            ]
            text_path = os.path.join(ex_dir, f"{int(camera_index)}.txt")
            # 写入到以索引命名的文本文件中
            with open(text_path, 'w') as outfile:
                for value in formatted_values:
                    outfile.write(f"{value}\n")

### finish
def generate_lidar():
    for index, clip_name in enumerate(os.listdir(source_dir)):
        ex_dir = os.path.join(target_dir, f"{index:03}", "lidar")
        if not os.path.exists(ex_dir): os.makedirs(ex_dir)
        directory = os.path.join(source_dir, clip_name)
        all_items = os.listdir(directory)
        sample_folders = [item for item in all_items if os.path.isdir(os.path.join(directory, item)) and item.startswith('sample')]
        # Sort the folders by the trailing numbers
        sample_folders.sort(key=lambda x: int(x.split('_')[-1]))
        for bin_index, sample in enumerate(sample_folders):
            path = os.path.join(directory, sample)
            lidar0_name = [name for name in os.listdir(path) if name.startswith('lidar0')][0]
            lidar0_path = os.path.join(path, lidar0_name)
            pointcloud = read_pcd_file(lidar0_path)
            bin_path = os.path.join(ex_dir, f"{bin_index:03}.bin")
            write_bin_file(bin_path, pointcloud)
### finish
def generate_sky_masks():
    command = "cd /data/gls/code/drivestudio"
    os.system(command)
    command = "python datasets/tools/extract_masks.py --data_root data/waymo_chery/processed/training --segformer_path=threeparty/SegFormer-master --checkpoint=threeparty/SegFormer-master/pretrained/segformer.b5.1024x1024.city.160k.pth --split_file data/waymo_example_scenes_chery.txt"
    os.system(command)
    
def generate_frame_info():
    pass

def main():
    # generate_intrinsics()
    generate_dynamic_masks()
    # generate_ego_pose()
    # generate_extrinsics()
    # generate_fine_dynamic_masks()
    # generate_humanpose()
    # generate_images()
    # generate_lidar()
    # generate_sky_masks()
    # generate_frame_info()
    # generate_instances()
    
if __name__ == "__main__":
    main()