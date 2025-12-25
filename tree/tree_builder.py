import re
from typing import List, Callable, Dict, Any
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import os
import multiprocessing

from .tree_node import RevisionTree, TreeNode
from .sql_utils import SQLExecutor
from .prompts import Prompts

def extract_sql_from_response(response: str) -> str:
    """
    Extract SQL from model response.
    """
    def clean_sql(sql: str) -> str:
        lines = sql.strip().split('\n')
        # Filter out lines that are purely comments or empty
        valid_lines = [line for line in lines if line.strip() and not line.strip().startswith('--')]
        return '\n'.join(valid_lines).strip()

    if "<answer>" in response and "</answer>" in response:
        answer = response.split("<answer>")[-1].split("</answer>")[0]
        if "```sql" in answer:
            pattern = r"```sql\s*(.*?)\s*```"
            matches = re.findall(pattern, answer, re.DOTALL)
            if matches:
                return clean_sql(matches[-1])
        return clean_sql(answer)

    pattern = r"```sql\s*(.*?)\s*```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return clean_sql(matches[-1])

    pattern = r"```\s*(.*?)\s*```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        for match in reversed(matches):
            sql = match.strip()
            if sql.upper().startswith("SELECT") or sql.upper().startswith("WITH"):
                return clean_sql(sql)
    
    # Fallback
    upper_response = response.upper()
    if "SELECT" in upper_response or "WITH" in upper_response:
        lines = response.split("\n")
        sql_lines = []
        in_sql = False
        for line in lines:
            stripped = line.strip().upper()
            if (stripped.startswith("SELECT") or stripped.startswith("WITH")) and not in_sql:
                in_sql = True
                sql_lines = [line]
            elif in_sql:
                sql_lines.append(line)
                if ";" in line:
                    break
        if sql_lines:
            return clean_sql("\n".join(sql_lines).strip())

    return clean_sql(response.strip())

def evaluate_single_response(response, ground_truth_sql, db_path):
    generated_sql = extract_sql_from_response(response)
    if not generated_sql:
        return "", False, "Error: No valid SQL found in response"
        
    is_correct, feedback = SQLExecutor.execute_sql(generated_sql, ground_truth_sql, db_path)
    return generated_sql, is_correct, feedback

class TreeBuilder:
    def __init__(self, llm_generator: Callable[[List[str]], List[str]], db_root_path: str, max_depth: int = 5, sample_num: int = 3):
        self.llm_generator = llm_generator
        self.db_root_path = db_root_path
        self.max_depth = max_depth
        self.sample_num = sample_num
        # Use a process pool for parallel SQL execution
        # Use 'spawn' context to avoid issues with CUDA initialized in parent process
        self.executor = ProcessPoolExecutor(
            max_workers=min(32, os.cpu_count() or 1),
            mp_context=multiprocessing.get_context("spawn")
        )

    def _parse_input_seq(self, input_seq: str):
        schema = ""
        question = ""
        evidence = ""

        # Extract Schema
        if "Database Schema:" in input_seq:
            parts = input_seq.split("Database Schema:")
            if len(parts) > 1:
                # Look for the next section which is usually "Question:"
                schema_part = parts[1]
                if "Question:" in schema_part:
                    schema = schema_part.split("Question:")[0].strip()
                else:
                    schema = schema_part.strip()
        
        # Extract Question (and potentially evidence if embedded)
        if "Question:" in input_seq:
            parts = input_seq.split("Question:")
            if len(parts) > 1:
                # Look for the next section which is usually "Instructions:"
                question_part = parts[1]
                if "Instructions:" in question_part:
                    question = question_part.split("Instructions:")[0].strip()
                else:
                    question = question_part.strip()

        # If schema is still empty, try the old line-based method as fallback or if format is different
        if not schema and not question:
             # ... (keep old logic or just return empty)
             pass
             
        return schema, evidence, question

    def _prune_tree(self, tree: RevisionTree):
        """
        Retain only nodes that are part of a path to a correct node.
        """
        useful_nodes = set()
        
        # 1. Find all correct nodes
        for node in tree.nodes.values():
            if node.is_correct:
                curr = node
                # 2. Backtrack to root
                while True:
                    useful_nodes.add(curr.node_id)
                    if curr.parent_id == "node_initial" or curr.parent_id not in tree.nodes:
                        break
                    curr = tree.nodes[curr.parent_id]
        
        # 3. Remove useless nodes
        for nid in list(tree.nodes.keys()):
            if nid not in useful_nodes:
                del tree.nodes[nid]
        
        # 4. Clean up children references
        for node in tree.nodes.values():
            node.children_ids = [cid for cid in node.children_ids if cid in useful_nodes]

    def build_tree(self, data: Dict[str, Any]) -> RevisionTree:
        question_id = str(data.get("question_id", ""))
        db_id = data.get("db_id", "")
        # question = data.get("question", "")
        # evidence = data.get("evidence", "")
        ground_truth_sql = data.get("SQL", "")
        initial_input_seq = data.get("input_seq", "") # From train_bird.json

        # Parse input_seq to get schema, evidence, and question
        schema, evidence, question = self._parse_input_seq(initial_input_seq)
        
        # Fallback if parsing failed to extract question/evidence (though user implies input_seq has them)
        if not question:
            question = data.get("question", "")
        if not evidence and "evidence" in data:
            # data['evidence'] might be empty string too
            evidence = data.get("evidence", "")

        tree = RevisionTree(question_id, db_id, question, evidence, ground_truth_sql)
        tree.node_initial = {"input_seq": initial_input_seq}
        tree.max_depth = self.max_depth

        db_path = SQLExecutor.get_db_path(self.db_root_path, db_id)
        # schema = SQLExecutor.get_schema(db_path) # Removed as per instruction

        # Queue for BFS: (parent_node_id, input_prompt, depth)
        # Initial step: parent is "node_initial"
        queue = deque([("node_initial", initial_input_seq, 1)])
        
        # Keep track of nodes to process at each depth to batch generation
        # But BFS usually processes one by one or level by level.
        # To optimize, we should process level by level.
        
        current_level_nodes = [("node_initial", initial_input_seq)]
        
        for depth in range(1, self.max_depth + 1):
            if not current_level_nodes:
                break
                
            # Prepare prompts for all nodes in this level
            # Each parent generates 'sample_num' children
            # Total requests = len(current_level_nodes) * sample_num
            
            prompts = []
            metadata = [] # (parent_id, child_index)
            
            for parent_id, prompt in current_level_nodes:
                for i in range(1, self.sample_num + 1):
                    prompts.append(prompt)
                    metadata.append((parent_id, i))
            
            if not prompts:
                break
                
            # Batch generate
            responses = self.llm_generator(prompts)
            
            next_level_nodes = []
            
            # Parallel execution of SQL validation
            results = list(self.executor.map(evaluate_single_response, responses, [ground_truth_sql]*len(responses), [db_path]*len(responses)))

            for ((parent_id, child_idx), response), (generated_sql, is_correct, feedback) in zip(zip(metadata, responses), results):
                # Skip incorrect nodes at max depth
                if depth == self.max_depth and not is_correct:
                    continue

                # Create child node
                # Node ID format: node_{depth}_{global_index_at_depth}
                
                # Let's count how many nodes we have at this depth so far
                current_depth_count = len([n for n in tree.nodes.values() if n.depth == depth])
                child_node_id = f"node_{depth}_{current_depth_count + 1}"
                
                child_node = TreeNode(child_node_id, parent_id, depth, current_depth_count + 1)
                child_node.generated_sql = generated_sql
                child_node.is_correct = is_correct
                child_node.execution_feedback = feedback
                
                # Add to tree
                tree.add_node(child_node)
                
                # Update parent's children list
                if parent_id != "node_initial":
                    if parent_id in tree.nodes:
                        tree.nodes[parent_id].children_ids.append(child_node_id)
                
                # Prepare for next level if needed
                if not is_correct and depth < self.max_depth:
                    # Construct revision prompt
                    revision_prompt = Prompts.REVISION_TEMPLATE.format(
                        schema=schema,
                        evidence=f"Evidence: {evidence}\n" if evidence else "",
                        question=question,
                        previous_sql=child_node.generated_sql,
                        execution_feedback=feedback
                    )
                    child_node.input_seq = revision_prompt
                    next_level_nodes.append((child_node_id, revision_prompt))
                else:
                    # Correct or max depth reached
                    child_node.input_seq = "" 
            
            current_level_nodes = next_level_nodes

        # Prune tree to keep only paths to correct nodes
        self._prune_tree(tree)

        # Calculate revision distances
        tree.calculate_revision_distance()
        
        return tree
