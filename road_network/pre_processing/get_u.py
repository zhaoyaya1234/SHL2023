import random
from tqdm import tqdm
import os
from chinese_calendar import is_holiday

import numpy as np
import torch

from pre_processing.spatial_func import distance
from pre_processing.trajectory import get_tid, Trajectory
from utils.parse_traj import ParseMMTraj
from utils.save_traj import SaveTraj2MM
from utils.utils import create_dir

from models.model_utils import toseq, get_constraint_mask
from map_matching.candidate_point import get_U_candidates
'''
这个文件用来生成U，来完成图神经网络的消息更新,search_dis 比计算约束矩阵大
需要注释掉split_data
'''

class Get_U_can(torch.utils.data.Dataset):
    """
    customize a dataset for PyTorch
    """

    def __init__(self, trajs_dir,save_path,rn,\
                        parameters,seed_value):
        self.parameters = parameters
        self.save_path= save_path
        self.seed_value = seed_value
        self.rn = rn 
        self.trajs_dir = trajs_dir
        self.grid_size = parameters.grid_size
        self.time_span = parameters.time_span  # time interval between two consecutive points.
    def get_data(self):

        random.seed(self.seed_value)
        if 'train' in self.trajs_dir:
            type_name = 'train_rid'
        elif 'valid' in self.trajs_dir:
            type_name = 'valid_rid'
        elif 'test' in self.trajs_dir:
            type_name = 'test_rid'
        
        trajs =np.load(self.trajs_dir,allow_pickle=True)
        print('traj {} data load well'.format(type_name))

        rid = []
        for traj in tqdm(trajs):
            single_rid= self.parse_traj(traj,self.parameters.win_size, self.parameters.ds_type, self.parameters.keep_ratio)
            rid.extend(single_rid)
        
        print('{} save well'.format(type_name))
        return rid 
        

    def parse_traj(self,traj, win_size, ds_type, keep_ratio):
       
        new_trajs = self.get_win_trajs(traj, win_size)
        single_rid = []
        for tr in new_trajs:
            tmp_pt_list = tr.pt_list
            # get source sequence
            '''
            拿剩下的这些去预测
            '''
            ds_pt_list = self.downsample_traj(tmp_pt_list, ds_type, keep_ratio)
            s_rid = self.get_src_seq(ds_pt_list)
            single_rid.extend(s_rid)
        return single_rid
    def get_distance(self,pt_list):
        """
        Get geographical distance of a trajectory (pt_list)
        sum of two adjacent points
        meters
        """
        dist = []
        pre_pt = pt_list[0]
        for pt in pt_list[1:]:
            tmp_dist = distance(pre_pt, pt)
            dist.append(tmp_dist)
            pre_pt = pt
        search_distance = np.zeros(len(pt_list))
        for i in range(len(pt_list)):
            if i == 0:
                search_distance[i]=dist[i]/2
            elif i == len(pt_list)-1:
                search_distance[i]=dist[-1]/2
            else:
                search_distance[i]=max(dist[i-1],dist[i])/2
        return list(search_distance)
    def get_src_seq(self, ds_pt_list):
        search_distance = self.get_distance(ds_pt_list)
        # print('distance',search_distance) # 删掉
        s_rid = [candidate for k,ds_pt in enumerate(ds_pt_list) for  candidate in get_U_candidates(ds_pt,self.rn, search_distance[k])]
        return s_rid
      

    def get_win_trajs(self, traj, win_size):
        pt_list = traj.pt_list
        len_pt_list = len(pt_list)
        if len_pt_list < win_size:
            return [traj]

        num_win = len_pt_list // win_size
        last_traj_len = len_pt_list % win_size + 1
        new_trajs = []
        ss = 0 
        for w in range(num_win):
            # if last window is large enough then split to a single trajectory
            if w == num_win and last_traj_len > 15:
                tmp_pt_list = pt_list[win_size * w - 1:]
            # elif last window is not large enough then merge to the last trajectory
            elif w == num_win - 1 and last_traj_len <= 15:
                # fix bug, when num_win = 1
                ind = 0
                if win_size * w - 1 > 0:
                    ind = win_size * w - 1
                tmp_pt_list = pt_list[ind:]
            # else split trajectories based on the window size
            else:
                tmp_pt_list = pt_list[max(0, (win_size * w - 1)):win_size * (w + 1)]
                # -1 to make sure the overlap between two trajs

            new_traj = Trajectory(traj.oid, traj.tid+"_"+str(ss), tmp_pt_list)
            new_trajs.append(new_traj)
            ss+=1
        return new_trajs


    @staticmethod
    def downsample_traj(pt_list, ds_type, keep_ratio):
        """
        Down sample trajectory
        Args:
        -----
        pt_list:
            list of Point()
        ds_type:
            ['uniform', 'random']
            uniform: sample GPS point every down_stepth element.
                     the down_step is calculated by 1/remove_ratio
            random: randomly sample (1-down_ratio)*len(old_traj) points by ascending.
        keep_ratio:
            float. range in (0,1). The ratio that keep GPS points to total points.
        Returns:
        -------
        traj:
            new Trajectory()
        """
        assert ds_type in ['uniform', 'random'], 'only `uniform` or `random` is supported'

        old_pt_list = pt_list.copy()
        start_pt = old_pt_list[0]
        end_pt = old_pt_list[len(pt_list)-1]

        if ds_type == 'uniform':
            if (len(old_pt_list) - 1) % int(1 / keep_ratio) == 0:
                new_pt_list = old_pt_list[::int(1 / keep_ratio)]
            else:
                new_pt_list = old_pt_list[::int(1 / keep_ratio)] + [end_pt]
        elif ds_type == 'random':
            sampled_inds = sorted(\
                random.sample(range(1, len(old_pt_list) - 1), int((len(old_pt_list) - 2) * keep_ratio)))
            new_pt_list = [start_pt] + list(np.array(old_pt_list)[sampled_inds]) + [end_pt]

        return new_pt_list