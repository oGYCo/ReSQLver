from typing import List, Dict, Set, Tuple, Any
from itertools import combinations
from .tree_node import RevisionTree, TreeNode
class DPOExtractor:
    @staticmethod
    def extract_pairs(tree: RevisionTree) -> List[Dict[str, Any]]:
        """
        Extracts chosen/rejected pairs from the tree.
        Candidate set for each node: {Parent, Siblings, Self, Children}
        """
        pairs = []
        seen_pairs = set()  # Store tuple of (id1, id2) sorted to avoid duplicates
        
        nodes = tree.nodes
        
        # Helper to get siblings
        # Group by parent
        siblings_map = {}
        for nid, node in nodes.items():
            pid = node.parent_id
            if pid not in siblings_map:
                siblings_map[pid] = []
            siblings_map[pid].append(nid)

        for center_id, center_node in nodes.items():
            candidates = set()
            candidates.add(center_id)
            
            # Parent (if exists and is in nodes map - node_initial is not in nodes map usually)
            if center_node.parent_id in nodes:
                candidates.add(center_node.parent_id)
                
            # Children
            for child_id in center_node.children_ids:
                if child_id in nodes:
                    candidates.add(child_id)
                    
            # Siblings
            pid = center_node.parent_id
            if pid in siblings_map:
                for sib_id in siblings_map[pid]:
                    candidates.add(sib_id)
            
            # Convert to list
            cand_list = list(candidates)
            
            # Generate pairs
            for id1, id2 in combinations(cand_list, 2):
                # Sort to ensure uniqueness check
                pair_key = tuple(sorted((id1, id2)))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                
                node1 = nodes[id1]
                node2 = nodes[id2]
                
                dist1 = node1.revision_distance
                dist2 = node2.revision_distance
                
                chosen = None
                rejected = None
                
                if dist1 < dist2:
                    chosen = node1
                    rejected = node2
                elif dist2 < dist1:
                    chosen = node2
                    rejected = node1
                
                if chosen and rejected:
                    pairs.append({
                        "question_id": tree.question_id,
                        "db_id": tree.db_id,
                        "question": tree.question,
                        "evidence": tree.evidence,
                        "chosen": chosen.generated_sql,
                        "rejected": rejected.generated_sql,
                        "chosen_dist": chosen.revision_distance,
                        "rejected_dist": rejected.revision_distance
                    })
                    
        return pairs
