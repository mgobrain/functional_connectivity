import networkx as nx
from scipy import stats
import pandas as pd
import numpy as np

class Graph:
    def __init__(self, table, thresh):
        self.data = pd.DataFrame(
                data=stats.zscore(table),
                columns=table.columns,
                index=table.index
                )
        self.graph = nx.Graph()
        for roi1 in self.data.columns:
            for roi2 in self.data.columns:
                if roi1 != roi2 and self.data[roi1][roi2] > thresh:
                    self.graph.add_edge(
                            roi1,roi2,
                            weight=self.data[roi1][roi2]
                            )


def subgraph_score(subgraph_set,score_func):
    dicts = [score_func(s_) for s_ in subgraph_set]
    single_dict = {}
    while len(dicts) > 0:
        single_dict.update(dicts.pop())
    return single_dict
    
def subgraph_node_score(subgraph,score_func):
    data = {}
    for node in subgraph:
        data[node] = score_func(subgraph, node)
    return data
    

# load initial results
import pickle
results = pickle.load(open('results.pkl','rb'))

# z-score cutoff
thresh = 0.5
features = {}
for sub in results:
    G = Graph(results[sub],thresh)
    data = {}
    data['connected_components'] = len(
        list(nx.connected_components(G.graph))
        )
    cluster_score = subgraph_score(
        nx.connected_component_subgraphs(G.graph),
        nx.clustering
        )
    for roi in cluster_score.keys():
        data['cluster_'+roi] = cluster_score[roi]
    features[sub] = data

features = pd.DataFrame.from_dict(features)
