from pre_processing.spatial_func import SPoint, LAT_PER_METER, LNG_PER_METER, project_pt_to_segment, distance
from pre_processing.mbr import MBR
import math

'''
这段代码实现了一个函数 get_candidates，其目的是根据给定的点 pt，从路网中找到离它最近的路段，并将该路段上距离该点一定距离范围内的投影点作为候选点返回。

其中用到了另一个函数 cal_candidate_point，它的作用是计算给定路段上距离给定点最近的投影点，并返回该点的相关属性，包括投影点的坐标、误差、在路段上的偏移量和路段的比率。

具体的实现细节可以看函数注释和代码中的变量名，例如 pt 表示点，rn 表示路网，search_dist 表示搜索半径，eid 表示路段的 ID，error 表示距离误差等等。
'''

class CandidatePoint(SPoint):
    def __init__(self, lat, lng, eid, error):
        super(CandidatePoint, self).__init__(lat, lng)
        self.eid = eid
        self.error = error
        self.lat = lat
        self.lng = lng

    def __str__(self):
        return '{},{},{},{}'.format(self.eid, self.lat, self.lng, self.error)

    def __repr__(self):
        return '{},{},{},{}'.format(self.eid, self.lat, self.lng, self.error)

    def __hash__(self):
        return hash(self.__str__())




def get_point_candidates(pt, rn, search_dist):
    """
    Args:
    -----
    pt: point STPoint()
    rn: road network
    search_dist: in meter. a parameter for HMM_mm. range of pt's potential road
    Returns:
    --------
    candidates: list of potential projected points.
    """
    candidates = None
    
    while True:
        mbr = MBR(pt.lat - search_dist * LAT_PER_METER,
              pt.lng - search_dist * LNG_PER_METER,
              pt.lat + search_dist * LAT_PER_METER,
              pt.lng + search_dist * LNG_PER_METER)
        all_candidate_nodes = rn.range_query(mbr) 
        candidate_nodes = all_candidate_nodes

        if len(candidate_nodes) > 0:
            break
        else:
            search_dist = search_dist+50

    candi_pt_list = [cal_candidate_point(pt, rn, candidate_node) for candidate_node in candidate_nodes]
    # refinement
    error_list = [candi_pt.error for candi_pt in candi_pt_list]
    index = error_list.index(min(error_list))
    candidates = candi_pt_list[index]
    return candidates

def cal_candidate_point(raw_pt, rn, node):
    """
    Get attributes of candidate point
    """

    coords = rn.node_index[node]['coords']  # GPS points in road segment, may be larger than 2
    candidates = [project_pt_to_segment(coords[i], coords[i + 1], raw_pt) for i in range(len(coords) - 1)]
    idx, (projection, coor_rate, dist) = min(enumerate(candidates), key=lambda x: x[1][2])
    # enumerate return idx and (), x[1] --> () x[1][2] --> dist. get smallest error project edge
   
    return CandidatePoint(projection.lat, projection.lng, rn.node_index[node]['eid'], dist)

