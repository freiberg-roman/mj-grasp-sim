from abc import ABC
from typing import List

from flax import nnx


class KinematicsModel(ABC):
    num_dofs: int
    kinematic_graph: List[List[int]]
    base_to_contact: nnx.Variable
    kinematics_transforms: nnx.Variable
    joint_transforms: nnx.Variable
    joint_ranges: nnx.Variable

    def parent_joints(self):
        parent_joints = []
        for chain in self.kinematic_graph:
            current_parent_chain = [-1]
            for i in range(1, len(chain)):
                current_parent_chain.append(chain[i - 1])
            parent_joints.append(current_parent_chain)
        return parent_joints

    @staticmethod
    def kinematic_graph_to_linear(graph) -> List[int]:
        chains = []
        for chain in graph:
            chains.extend(chain)
        return chains

    @staticmethod
    def to_emb_idx(graph) -> List[int]:
        lin_graph = KinematicsModel.kinematic_graph_to_linear(graph)
        idx_graph = list(map(lambda x: x + 1, lin_graph))  # +1 for base node
        return idx_graph
