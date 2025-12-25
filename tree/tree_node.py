from typing import List, Dict, Optional, Any
import json

class TreeNode:
    def __init__(self, node_id: str, parent_id: str, depth: int, index: int):
        self.node_id = node_id
        self.parent_id = parent_id
        self.depth = depth
        self.index = index  # j in node_d_j
        self.children_ids: List[str] = []
        self.generated_sql: str = ""
        self.execution_feedback: str = ""
        self.is_correct: bool = False
        self.input_seq: str = ""  # Prompt used to generate children (if any)
        self.revision_distance: int = 999  # Default to inf

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parent_id": self.parent_id,
            "d": self.depth,
            "j": self.index,
            "children_num": len(self.children_ids),
            "children_ids": self.children_ids,
            "generated_sql": self.generated_sql,
            "execution_feedback": self.execution_feedback,
            "is_correct": self.is_correct,
            "input_seq": self.input_seq,
            "revision_distance": self.revision_distance
        }

    @classmethod
    def from_dict(cls, node_id: str, data: Dict[str, Any]) -> 'TreeNode':
        node = cls(node_id, data.get("parent_id"), data.get("d"), data.get("j"))
        node.children_ids = data.get("children_ids", [])
        node.generated_sql = data.get("generated_sql", "")
        node.execution_feedback = data.get("execution_feedback", "")
        node.is_correct = data.get("is_correct", False)
        node.input_seq = data.get("input_seq", "")
        node.revision_distance = data.get("revision_distance", 999)
        return node

class RevisionTree:
    def __init__(self, question_id: str, db_id: str, question: str, evidence: str, ground_truth_sql: str):
        self.question_id = question_id
        self.db_id = db_id
        self.question = question
        self.evidence = evidence
        self.ground_truth_sql = ground_truth_sql
        self.nodes: Dict[str, TreeNode] = {}
        self.node_initial: Dict[str, str] = {}
        self.max_depth = 0
        self.tree_status = "bad"

    def add_node(self, node: TreeNode):
        self.nodes[node.node_id] = node

    def calculate_revision_distance(self):
        """
        Calculate revision distance for all nodes using backpropagation.
        Logic:
        - If is_correct is True: distance = 0
        - If leaf and not correct: distance = 999
        - If internal and not correct: distance = min(children's distance) + 1
        """
        # We need to process from bottom up (max depth to 1)
        max_d = 0
        for node in self.nodes.values():
            max_d = max(max_d, node.depth)
        
        # Initialize all to 999 or 0 if correct
        for node in self.nodes.values():
            if node.is_correct:
                node.revision_distance = 0
            else:
                node.revision_distance = 999

        # Backpropagate
        for d in range(max_d, 0, -1):
            # Get nodes at depth d
            nodes_at_d = [n for n in self.nodes.values() if n.depth == d]
            for node in nodes_at_d:
                # If node is correct, it's already 0, no need to update
                if node.is_correct:
                    continue
                
                # If node is not correct, look at children
                if not node.children_ids:
                    # Leaf and incorrect -> 999
                    node.revision_distance = 999
                else:
                    # Internal and incorrect
                    min_child_dist = 999
                    for child_id in node.children_ids:
                        if child_id in self.nodes:
                            min_child_dist = min(min_child_dist, self.nodes[child_id].revision_distance)
                    
                    if min_child_dist == 999:
                        node.revision_distance = 999
                    else:
                        node.revision_distance = min_child_dist + 1

        # Check tree status (root distance)
        root = self.nodes.get("node_1_1")
        if root and root.revision_distance < 999:
            self.tree_status = "good"
        else:
            self.tree_status = "bad"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "db_id": self.db_id,
            "evidence": self.evidence,
            "question": self.question,
            "SQL": self.ground_truth_sql,
            "node_initial": self.node_initial,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "max_depth": self.max_depth,
            "tree_status": self.tree_status
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RevisionTree':
        tree = cls(
            data.get("question_id"),
            data.get("db_id"),
            data.get("question"),
            data.get("evidence"),
            data.get("SQL")
        )
        tree.node_initial = data.get("node_initial", {})
        tree.max_depth = data.get("max_depth", 0)
        tree.tree_status = data.get("tree_status", "bad")
        
        nodes_data = data.get("nodes", {})
        for nid, ndata in nodes_data.items():
            tree.add_node(TreeNode.from_dict(nid, ndata))
            
        return tree
