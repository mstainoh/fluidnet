from typing import Hashable


class Node:
    def __init__(self, name: Hashable, z: float=0):
        self.name = name
        self.z = z
    
    def __hash__(self):
        return hash(self.name)

class Edge:
    def __init__(self, node_from: Node, node_to: Node, **kwargs):
        self.node_from = node_from
        self.node_to = node_to
        kwargs['dz'] = self.node_to.z - self.node_from.z
        self.parameters = kwargs
    
    def __hash__(self):
        return hash((self.node_from, self.node_to))
    
    def get_dz(self):
        return self.parameters['dz']
    
    def as_tuple(self):
        return (self.node_to, self.node_from, self.parameters)