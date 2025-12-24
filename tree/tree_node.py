"""
树节点和修订树的数据结构定义

每棵树针对一个question，根节点的node_id是node_1_1
节点ID格式: node_d_j，其中d表示深度，j表示该深度的位置索引
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
import json


@dataclass
class TreeNode:
    """
    修订树的节点
    
    Attributes:
        node_id: 节点ID，格式为 node_d_j
        parent_id: 父节点ID
        d: 节点深度
        j: 该深度的位置索引
        generated_sql: 模型生成的SQL代码
        execution_feedback: SQL执行反馈信息
        is_correct: SQL执行结果是否正确
        input_seq: 用于生成子节点的输入提示词
        children_num: 子节点个数（采样次数）
        children_ids: 子节点ID列表
        revision_distance: 修订距离（反向传播计算）
    """
    node_id: str
    parent_id: str
    d: int  # 深度
    j: int  # 该深度的位置索引
    generated_sql: str = ""
    execution_feedback: str = ""
    is_correct: bool = False
    input_seq: str = ""
    children_num: int = 0
    children_ids: List[str] = field(default_factory=list)
    revision_distance: int = 999  # 999 表示无穷大
    
    # 额外的元数据
    execution_result: Any = None  # 执行结果
    execution_status: str = ""  # success, error, timeout
    response_text: str = ""  # 模型的完整响应
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "d": self.d,
            "j": self.j,
            "generated_sql": self.generated_sql,
            "execution_feedback": self.execution_feedback,
            "is_correct": self.is_correct,
            "input_seq": self.input_seq,
            "children_num": self.children_num,
            "children_ids": self.children_ids,
            "revision_distance": self.revision_distance,
            "execution_status": self.execution_status
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TreeNode':
        """从字典创建节点"""
        return cls(
            node_id=data.get("node_id", ""),
            parent_id=data.get("parent_id", ""),
            d=data.get("d", 0),
            j=data.get("j", 0),
            generated_sql=data.get("generated_sql", ""),
            execution_feedback=data.get("execution_feedback", ""),
            is_correct=data.get("is_correct", False),
            input_seq=data.get("input_seq", ""),
            children_num=data.get("children_num", 0),
            children_ids=data.get("children_ids", []),
            revision_distance=data.get("revision_distance", 999),
            execution_status=data.get("execution_status", "")
        )


@dataclass 
class InitialNode:
    """初始节点，用于生成根节点的SQL代码"""
    input_seq: str = ""
    
    def to_dict(self) -> Dict:
        return {"input_seq": self.input_seq}


class RevisionTree:
    """
    修订树数据结构
    
    每棵树对应BIRD数据集中的一个question
    """
    
    INF_DISTANCE = 999  # 表示无穷大的修订距离
    
    def __init__(
        self,
        question_id: str,
        db_id: str,
        question: str,
        evidence: str = "",
        difficulty: str = "simple",
        gold_sql: str = "",
        max_depth: int = 5,
        sample_num: int = 3
    ):
        """
        初始化修订树
        
        Args:
            question_id: 问题ID
            db_id: 数据库ID
            question: 自然语言问题
            evidence: 辅助信息
            difficulty: 问题难度
            gold_sql: 标准答案SQL
            max_depth: 树的最大深度
            sample_num: 每个节点的采样次数
        """
        self.question_id = question_id
        self.db_id = db_id
        self.question = question
        self.evidence = evidence
        self.difficulty = difficulty
        self.gold_sql = gold_sql
        self.max_depth = max_depth
        self.sample_num = sample_num
        
        # 初始节点
        self.node_initial = InitialNode()
        
        # 节点字典，key为node_id
        self.nodes: Dict[str, TreeNode] = {}
        
        # 树的状态
        self.tree_status = "bad"  # good/bad
        
        # 每层节点计数器（用于生成node_id）
        self._depth_counters: Dict[int, int] = {}
        
    def _get_next_node_id(self, depth: int) -> str:
        """获取下一个节点ID"""
        if depth not in self._depth_counters:
            self._depth_counters[depth] = 0
        self._depth_counters[depth] += 1
        return f"node_{depth}_{self._depth_counters[depth]}"
    
    def create_node(
        self,
        parent_id: str,
        depth: int,
        generated_sql: str = "",
        response_text: str = ""
    ) -> TreeNode:
        """
        创建新节点
        
        Args:
            parent_id: 父节点ID
            depth: 节点深度
            generated_sql: 生成的SQL
            response_text: 模型响应文本
            
        Returns:
            新创建的节点
        """
        node_id = self._get_next_node_id(depth)
        j = self._depth_counters[depth]
        
        node = TreeNode(
            node_id=node_id,
            parent_id=parent_id,
            d=depth,
            j=j,
            generated_sql=generated_sql,
            response_text=response_text
        )
        
        self.nodes[node_id] = node
        
        # 更新父节点的children信息
        if parent_id != "node_initial" and parent_id in self.nodes:
            self.nodes[parent_id].children_ids.append(node_id)
            self.nodes[parent_id].children_num = len(self.nodes[parent_id].children_ids)
            
        return node
    
    def get_node(self, node_id: str) -> Optional[TreeNode]:
        """获取指定节点"""
        return self.nodes.get(node_id)
    
    def get_nodes_at_depth(self, depth: int) -> List[TreeNode]:
        """获取指定深度的所有节点"""
        return [node for node in self.nodes.values() if node.d == depth]
    
    def get_leaf_nodes(self) -> List[TreeNode]:
        """获取所有叶子节点"""
        return [node for node in self.nodes.values() if node.children_num == 0]
    
    def get_incorrect_expandable_nodes(self) -> List[TreeNode]:
        """获取所有错误且可继续扩展的节点（未达到最大深度）"""
        return [
            node for node in self.nodes.values()
            if not node.is_correct and node.d < self.max_depth
        ]
    
    def compute_revision_distances(self) -> None:
        """
        反向传播计算修订距离
        
        规则：
        1. is_correct == True 的节点，revision_distance = 0
        2. 叶子节点且 is_correct == False，revision_distance = INF (999)
        3. 非叶子节点且 is_correct == False，revision_distance = min(children) + 1
        """
        # 首先标记所有正确节点
        for node in self.nodes.values():
            if node.is_correct:
                node.revision_distance = 0
        
        # 从最大深度开始，逐层向上计算
        for depth in range(self.max_depth, 0, -1):
            nodes_at_depth = self.get_nodes_at_depth(depth)
            
            for node in nodes_at_depth:
                if node.is_correct:
                    continue  # 已经是0，跳过
                    
                if node.children_num == 0:
                    # 叶子节点且错误
                    node.revision_distance = self.INF_DISTANCE
                else:
                    # 非叶子节点，计算子节点的最小修订距离
                    child_distances = [
                        self.nodes[child_id].revision_distance 
                        for child_id in node.children_ids
                        if child_id in self.nodes
                    ]
                    if child_distances:
                        min_child_dist = min(child_distances)
                        if min_child_dist < self.INF_DISTANCE:
                            node.revision_distance = min_child_dist + 1
                        else:
                            node.revision_distance = self.INF_DISTANCE
        
        # 更新树状态
        self._update_tree_status()
    
    def _update_tree_status(self) -> None:
        """根据根节点的修订距离更新树状态"""
        # 查找根节点（深度为1的节点）
        root_nodes = self.get_nodes_at_depth(1)
        
        if not root_nodes:
            self.tree_status = "bad"
            return
            
        # 如果任何一个根节点是正确的或者修订距离不是INF，则为good
        for node in root_nodes:
            if node.revision_distance < self.INF_DISTANCE:
                self.tree_status = "good"
                return
                
        self.tree_status = "bad"
    
    def has_correct_node(self) -> bool:
        """检查树中是否有正确的节点"""
        return any(node.is_correct for node in self.nodes.values())
    
    def get_first_correct_node(self) -> Optional[TreeNode]:
        """获取第一个正确的节点（BFS顺序）"""
        for depth in range(1, self.max_depth + 1):
            for node in self.get_nodes_at_depth(depth):
                if node.is_correct:
                    return node
        return None
    
    def get_statistics(self) -> Dict:
        """获取树的统计信息"""
        total_nodes = len(self.nodes)
        correct_nodes = sum(1 for n in self.nodes.values() if n.is_correct)
        
        depth_stats = {}
        for depth in range(1, self.max_depth + 1):
            nodes_at_d = self.get_nodes_at_depth(depth)
            depth_stats[f"depth_{depth}"] = {
                "total": len(nodes_at_d),
                "correct": sum(1 for n in nodes_at_d if n.is_correct)
            }
        
        return {
            "total_nodes": total_nodes,
            "correct_nodes": correct_nodes,
            "tree_status": self.tree_status,
            "depth_statistics": depth_stats
        }
    
    def to_dict(self) -> Dict:
        """将树转换为字典格式（用于JSON序列化）"""
        return {
            # 元数据
            "question_id": self.question_id,
            "db_id": self.db_id,
            "question": self.question,
            "evidence": self.evidence,
            "difficulty": self.difficulty,
            "SQL": self.gold_sql,
            
            # 初始节点
            "node_initial": self.node_initial.to_dict(),
            
            # 所有节点
            "nodes": {
                node_id: node.to_dict() 
                for node_id, node in self.nodes.items()
            },
            
            # 树的全局属性
            "max_depth": self.max_depth,
            "sample_num": self.sample_num,
            "tree_status": self.tree_status,
            
            # 统计信息
            "statistics": self.get_statistics()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RevisionTree':
        """从字典创建树"""
        tree = cls(
            question_id=data.get("question_id", ""),
            db_id=data.get("db_id", ""),
            question=data.get("question", ""),
            evidence=data.get("evidence", ""),
            difficulty=data.get("difficulty", "simple"),
            gold_sql=data.get("SQL", ""),
            max_depth=data.get("max_depth", 5),
            sample_num=data.get("sample_num", 3)
        )
        
        # 恢复初始节点
        if "node_initial" in data:
            tree.node_initial.input_seq = data["node_initial"].get("input_seq", "")
        
        # 恢复所有节点
        if "nodes" in data:
            for node_id, node_data in data["nodes"].items():
                node = TreeNode.from_dict(node_data)
                tree.nodes[node_id] = node
                
                # 更新深度计数器
                depth = node.d
                if depth not in tree._depth_counters:
                    tree._depth_counters[depth] = 0
                tree._depth_counters[depth] = max(
                    tree._depth_counters[depth], 
                    node.j
                )
        
        tree.tree_status = data.get("tree_status", "bad")
        
        return tree
    
    def save_to_file(self, filepath: str) -> None:
        """保存树到JSON文件"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'RevisionTree':
        """从JSON文件加载树"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def __repr__(self) -> str:
        return (
            f"RevisionTree(question_id={self.question_id}, "
            f"nodes={len(self.nodes)}, status={self.tree_status})"
        )
