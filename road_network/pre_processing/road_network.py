import networkx as nx
from rtree import Rtree
from osgeo import ogr
from .spatial_func import SPoint, distance
from .mbr import MBR
import copy
import torch
from tqdm import tqdm

class RoadNetwork(nx.DiGraph):
    def __init__(self, g, edge_spatial_idx, node_index):
        super(RoadNetwork, self).__init__(g)
        # entry: eid
        self.edge_spatial_idx = edge_spatial_idx
        # eid -> edge key (start_coord, end_coord)
        self.node_index = node_index
        self.g = g

    def range_query(self, mbr):
        """
        spatial range query
        :param mbr: query mbr
        :return: qualified edge keys
        """
        eids = self.edge_spatial_idx.intersection((mbr.min_lng, mbr.min_lat, mbr.max_lng, mbr.max_lat))
        return list(eids)

def load_rn_shp(g, is_directed=False):
    edge_spatial_idx = Rtree()
    node_idx = {}
  

    # edge attrs: eid, length, coords, ...
    for node, data in g.nodes(data=True): 
        coords = [SPoint(value[0],value[1]) for value in data['coords']]
        data['coords'] = coords
        data['length'] = sum([distance(coords[i], coords[i + 1]) for i in range(len(coords) - 1)])
        edge_spatial_idx.insert(data['eid'], data['zone_area'])
        node_idx[data['eid']] = data
        eid_value = int(data['eid'])

    print('# of nodes:{}'.format(g.number_of_nodes()))
    return RoadNetwork(g, edge_spatial_idx, node_idx)