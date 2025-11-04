import os
import numpy as np
import torch
from utils.lidar_utils import lidar_to_pano_with_intensities
import open3d as o3d

class Chery_Dataloader:
    """
    Load Chery dataset, output as Waymo format
    """
    FOV_HORIZONTAL = 120 * torch.pi / 180.0
    FOV_UP = 7.0 * torch.pi / 180.0     # 向上7度
    FOV_DOWN = 13.0 * torch.pi / 180.0       # 总共20度垂直视场
    NUM_BEAMS = 100                 # 128线
    W_LIDAR = 1200                  # 水平分辨率
    H_LIDAR = 100                   # 垂直分辨率
    BEAM_INCLINATIONS = [
        -13.03, -11.82, -10.84, -10.03, -9.47, -9.07, -8.66, -8.25, -7.88, -7.47,
        -7.07, -6.66, -6.26, -5.86, -5.45, -5.05, -4.64, -4.55, -4.45, -4.34, 
        -4.23, -4.14, -4.04, -3.94, -3.83, -3.73, -3.64, -3.53, -3.42, -3.33, 
        -3.23, -3.13, -3.02, -2.92, -2.83, -2.72, -2.62, -2.52, -2.42, -2.32, 
        -2.21, -2.12, -2.02, -1.91, -1.81, -1.71, -1.61, -1.51, -1.41, -1.31, 
        -1.21, -1.11, -1.01, -0.91, -0.81, -0.71, -0.61, -0.51, -0.41, -0.30, 
        -0.20, -0.10, 0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 
        0.81, 0.91, 1.00, 1.11, 1.21, 1.31, 1.41, 1.51, 1.61, 1.71, 
        1.81, 1.91, 2.01, 2.11, 2.21, 2.31, 2.41, 2.52, 2.62, 3.03, 
        3.43, 3.84, 4.24, 4.65, 5.05, 5.46, 5.87, 6.28, 6.69, 7.09
    ]

    def __init__(self, args, train=True, train_frame_times=None, dtype=np.float32):
        self.train = train
        self.root_path = args.source_path
        self.case = args.caseid
        
        self.frames_data = None
        lidar_filefolder = os.path.join(args.source_path, "lidar")
        lidarpose_filefolder = os.path.join(args.source_path, "lidar_pose")
        bin_files = [f for f in os.listdir(lidar_filefolder) if f.endswith('.bin')]
        bin_files = sorted(bin_files, key=lambda x: int(x.split('.')[0]))
        print("[ Info ] find {} bin files in {}".format(len(bin_files), lidar_filefolder))
        self.start_frame = train_frame_times[0]
        frame_cnt = train_frame_times[-1] - train_frame_times[0] + 1
        print("[ Info ] start frame is {}".format(self.start_frame))
        
        frames_data = []
        lidar_to_world_start = np.loadtxt(
            os.path.join(lidarpose_filefolder, str(self.start_frame).zfill(3)+'.txt')
        )
        for i in range(self.start_frame, self.start_frame + frame_cnt):
            """
            1. log_time_stamp
            2. lidar2world
            3. lidar_points
            4. time_stamp
            5. ring
            6. lidar_id
            7. path {pcd}
            """
            if i < self.start_frame: continue
            single_frame_data = {}
            single_frame_data['log_time_stamp'] = i

            lidar_pose_i_path = os.path.join(lidarpose_filefolder, str(i).zfill(3)+'.txt')
            lidar_to_world_current = np.loadtxt(lidar_pose_i_path)
            lidar_to_world = np.linalg.inv(lidar_to_world_start) @ lidar_to_world_current
            single_frame_data['lidar2world'] = lidar_to_world

            lidar_info = np.fromfile(
                os.path.join(args.source_path, "lidar", str(i).zfill(3)+'.bin'), dtype=np.float32
            ).reshape(-1, 5)
            # 增加两列 ring 和 lidar_id，补全为7列，ring全为0，lidar_id全为0
            lidar_info[:, 3] = lidar_info[:, 3] * 255 # 强度归一化到0-255之间
            ring = np.zeros((lidar_info.shape[0], 1), dtype=np.float32)

            lidar_id = np.zeros((lidar_info.shape[0], 1), dtype=np.float32)
            lidar_info = np.concatenate([lidar_info, ring, lidar_id], axis=1) # shape (N, 7)
            # 过滤掉lidar_id不为0的点
            lidar_info = lidar_info[lidar_info[:, 6] == 0]

            lidar_points = lidar_info[:, :3]  # shape (N, 3)
            lidar_intensity = lidar_info[:, 3:4] / 255.0  # shape (N, 1)
            lidar_data = np.concatenate([lidar_points, lidar_intensity], axis=1)  # (N, 4)
            single_frame_data['lidar_points'] = lidar_data  # 存 NumPy array
            single_frame_data['time_stamp'] = lidar_info[:,4]
            single_frame_data['ring'] = lidar_info[:,5]
            single_frame_data['lidar_id'] = lidar_info[:,6]
            pcd_i_path = os.path.join(lidar_filefolder, str(i).zfill(3)+'.bin')
            single_frame_data['path'] = {'pcd': os.path.relpath(pcd_i_path, self.root_path)}
            frames_data.append(single_frame_data)
        self.frames_data = frames_data
        print("[ Info ] this case have {} frames totally".format(len(self.frames_data)))

        # 如果没有提供精确的 beam_inclinations，使用常量参数
        fov_up = self.FOV_UP
        fov_down = self.FOV_DOWN
        num_beams = self.NUM_BEAMS
        # 注意这里是顺序是从小到大，即从 -fov_down 到 +fov_up, 而且是弧度制
        self.beam_inclinations = np.linspace(-fov_down, fov_up, num_beams, dtype=np.float32)
        # self.beam_inclinations = [angle * torch.pi / 180.0 for angle in self.BEAM_INCLINATIONS]

        # 假设雷达位置就是自车位置
        R = np.eye(3, dtype=np.float32)
        T = np.zeros((3,1), dtype=np.float32)
        self.extrinsic = np.block([[R, T], [0,0,0,1]])# laser_to_vehicle

        self.W_lidar = int(self.W_LIDAR)
        self.H_lidar = int(self.H_LIDAR)
        self.train_frame_times = train_frame_times
        self.max_frame_num = len(train_frame_times)
        self.max_depth = args.max_depth # helios 5515 的最大深度是150 看情况调整，大于100的点也很少基本，误差更大
        print("max_depth", self.max_depth)
        self.aabb_min = np.ones(3, dtype=np.float32)*(10000000)
        self.aabb_max = np.ones(3, dtype=np.float32)*(-10000000)

        self.sensor2baselidar = dict() # 记录每个lidar到toplidar的变换矩阵
        self.pcds = [] # 原始每帧点云
        self.pcds_label = [] # onemodle的语义结果 ==10为地面 ==0为背景 
        self.l2ws = [] # 每帧的l2w
        self.timestep_2_frameid = dict() # 通过timsestep查询对应训练的帧的id _ 0 to 50
        self.frameid_2_timestep = [] # 通过frame id 反查询对应训练帧的timestep
        self.baselidar2world = []
        self.pcds, self.l2ws = self.load_pcds(frames=self.frames_data, train_frame_times=self.train_frame_times, frame_num=self.max_frame_num)
        self.obj_id_list = None

        self.obj_frames_id = dict()  # 通过obj_id 查询实例出现在哪几帧（列表）(frame id : 0-50)
        self.obj_pcd = dict() # 通过obj_id 查询实例的拼接后的完整的pcd
        self.obj_o2l = dict() # 通过obj_id 和对应那一帧的frame id查询实例的o2l，字典嵌套了一个字典


        # # 拼接训练帧点云作为静态场景，后续的高斯初始化需要
        # pcd_xyzs = []
        # for i in range(0, len(self.pcds)):
        #     lidar_to_world = self.l2ws[i]
        #     R = lidar_to_world[:3, :3]
        #     T = lidar_to_world[:3, 3]
        #     pcd = self.pcds[i][:, :3]
        #     pcd_transformed = (R @ pcd.T).T + T  # shape (N, 3)
        #     pcd_xyz = pcd_transformed[:, :3]    # shape: (N_i, 3)
        #     pcd_xyzs.append(pcd_xyz)
        # self.static_pcd = np.concatenate(pcd_xyzs, axis=0)  # shape: (total_points, 3)

        # # 保存整个静态pcd到txt文件，用于可视化，文件名args.block_id + static_scene.txt
        # np.savetxt(os.path.join(self.root_path, "static_scene_all_frames.txt"), self.static_pcd)
        # print("[ Info ] static scene have {} points".format(self.static_pcd.shape[0]))

        self.static_pcd = np.loadtxt("/home/not0513/data/orinY/processed/training/20250702_133223_Q2517/static_scene_all_frames.txt")

        self.range_views, self.masks = self.load_rangeview(self.H_lidar,self.W_lidar)

    def load_rangeview(self, H_lidar, W_lidar):
        '''
        return [H,W,3] rangeiew
        '''
        range_views = []
        masks = []
        for frame_idx in range(self.max_frame_num):
            pano, intensities, mask = lidar_to_pano_with_intensities(  
                    local_points_with_intensities=self.pcds[frame_idx],
                    lidar_H=H_lidar,
                    lidar_W=W_lidar,
                    beam_inclinations=self.beam_inclinations,
                    max_depth=self.max_depth,
                )
            range_view = np.zeros((H_lidar, W_lidar, 3))
            range_view[:, :, 1] = intensities
            range_view[:, :, 2] = pano
            ray_drop = np.where(range_view.reshape(-1, 3)[:, 2] <= 0.0, 0.0,
                                1.0).reshape(H_lidar, W_lidar, 1)
            image_lidar = np.concatenate(
                [
                    ray_drop,
                    np.clip(range_view[:, :, 1, None], 0, 1),
                    range_view[:, :, 2, None]
                ],
                axis=-1,
            )
            range_views.append(image_lidar)
            masks.append(mask)
        return range_views,masks

    def load_pcds(self, frames, train_frame_times, frame_num):
        '''
        加载每帧点云 作为gt 以及计算边界 其中selected_sensor = 0 / 1 / 3 / 4 分别代表 ROTOTOP, ROBO_BACK, ROBO_LEFT_FRONT, ROBO_RIGHT_FRONT
        return : 原始每帧点云_baselidar系 
        '''

        pcds = []
        pcds_label = []
        l2ws = []
        count_ind = 0
        for f_id, frame in enumerate(frames):
            if count_ind == frame_num: break
            if frame['log_time_stamp'] != int(train_frame_times[count_ind]): continue
            # 这里的 lidar2world 是baselidar系
            l2w = np.array(frame['lidar2world'])
            self.baselidar2world.append(l2w)
            sl2w = l2w @ self.extrinsic # vehicle2wordl @ lidar2vehicle
            l2ws.append(sl2w)

            pcd = frame['lidar_points']
            ring_data = frame['ring']
            ring_min = np.min(ring_data)
            ring_max = np.max(ring_data)

            pcd_world = (np.pad(pcd[...,:3], ((0,0),(0, 1)), constant_values=1) @ l2w.T)[:,:3]
            aabb_min = np.min(pcd_world, axis=0)
            aabb_max = np.max(pcd_world, axis=0)
            self.aabb_min = np.minimum(self.aabb_min, aabb_min)
            self.aabb_max = np.maximum(self.aabb_max, aabb_max)

            vehicle_to_laser = np.linalg.inv(self.extrinsic)
            sensor_pcd = (np.pad(pcd[...,:3], ((0,0),(0, 1)), constant_values=1) @ vehicle_to_laser.T)[:,:3]      
            pcd[:,:3] = sensor_pcd[:,:3]
            pcds.append(pcd)

            timestep = str(frame["log_time_stamp"])
            self.timestep_2_frameid.update({timestep : count_ind})
            self.frameid_2_timestep.append(timestep)
            count_ind += 1

        if count_ind < frame_num: raise ValueError("Missing Frame or Abnormal Loading.")
        return pcds, l2ws#, pcds_label
    
    def get_frames_nums(self):
        return self.max_frame_num

    def get_static_pcd(self):
        return self.static_pcd
    
    def get_obj_pcd(self, object_id):
        return self.obj_pcd[str(object_id)]

    def get_rangeview(self,frame_idx):
        '''
        return [H,W,3] numpy 
        '''
        return self.range_views[frame_idx]
    def get_mask(self,frame_idx):
        '''
        return [H,W,3] numpy 
        '''
        return self.masks[frame_idx]

    def getlidar2world(self):
        '''
        return list 所有帧的sensorlidar2world
        '''
        return self.l2ws

    def get_obj2lidar(self,occurred_frame_idx,object_id, newcar_render=None):        
        '''
        return [4,4] numpy 返回obj2lidar的矩阵
        baselidar2sensor @ obj2baselidar
        '''
        vehicle_to_laser = np.linalg.inv(self.extrinsic)
        return vehicle_to_laser @ self.obj_o2l[str(object_id)][str(occurred_frame_idx)] # TODO 
        # return self.obj_o2l[str(object_id)][str(occurred_frame_idx)]

    def get_sensor2baselidar(self,sensorid):
        '''
        字典返回每个雷达系到baselidar系的矩阵
        '''
        return self.sensor2baselidar[self.lidar_map[sensorid]]
        
    # def get_sensor2baselidar_newcar(self,sensorid):
    #     return self.new_sensor2baselidar[self.lidar_map[sensorid]]

    def get_obj_frames(self,object_id):
        '''
        列表返回obj出现的帧
        '''
        return self.obj_frames_id[str(object_id)]

    def get_dynamic_obj_id_list(self):
        '''
        列表返回移动的obj的id
        '''
        return self.obj_id_list

    def get_beam_inclination(self):
        return self.beam_inclinations

    def get_lidar_res(self):
        return self.W_lidar, self.H_lidar

    def get_fov_horizontal(self):
        return self.FOV_HORIZONTAL

    def get_fov_up(self):
        return self.FOV_UP

    def get_fov_down(self):
        return self.FOV_DOWN