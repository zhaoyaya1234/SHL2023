from map_matching.candidate_point import get_point_candidates
import re
from datetime import datetime
import pandas as pd
import numpy as np
import time
from pre_processing.trajectory import Trajectory, STPoint
from pre_processing.spatial_func import SPoint, LAT_PER_METER, LNG_PER_METER, project_pt_to_segment, distance
from pre_processing.mbr import MBR
from tqdm import tqdm
import math

def get_candi_shl(df_point,rn,roadid2code,save_path,search_dis=50):


    candidates =  df_point.apply(lambda x:get_point_candidates(STPoint(x.latitude,x.longitude),rn,search_dis), axis=1)
    
    eid = [candidate.eid for candidate in candidates]
    code = [roadid2code[candidate.eid] for candidate in candidates]
    error = [candidate.error for candidate in candidates]

    df_point[["eid","error",'code']] = pd.DataFrame({"eid":eid,"error":error,'code':code})
    df_point.to_csv('{}/get_road_feature.csv'.format(save_path), index=False)
    return df_point





