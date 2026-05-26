from typing import Hashable
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
import numpy as np

class Node:
    def __init__(self, name: Hashable, z: float=0, **kwargs):
        self.name = name
        self.z = z
        self.parameters = kwargs
    
    def __hash__(self):
        return hash(self.name)

class Pipe():
    def __init__(self, node_from: Node, node_to: Node, **kwargs):
        self.node_from = node_from
        self.node_to = node_to
        self.parameters = kwargs
    
    def __hash__(self):
        return hash((self.node_from, self.node_to))
    
    def get_dz(self):
        return self.node_to.z - self.node_from.z
    
    def as_tuple(self):
        return (self.node_to, self.node_from)

Edge = Pipe # alias

class PipeWithTrayectory(Pipe):
    def __init__(self, node_from: Node, node_to: Node, xs, ys, **kwargs):
        self.xs, self.ys = map(np.asarray, (xs, ys))
        self.spline = IUS(xs, ys)
        super().__init__(node_from, node_to, **kwargs)
    
    def get_inclination(self, x):
        return self.spline.derivative()(x)