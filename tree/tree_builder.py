"""
树构建器模块

功能：
1. 使用BFS算法构建修订树
2. 调用LLM进行SQL生成和修订
3. 管理树的扩展和剪枝
"""

import os
import re
import sys
from collections import deque
from typing import List, Dict, Optional, Callable, Any
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tree.tree_node import RevisionTree, TreeNode
from tree.sql_executor import SQLExecutor, get_db_path
from tree.prompt_templates import PromptTemplates, SchemaLoader


def extract_sql_from_response(response: str) -> str:
    """
    从模型响应中提取SQL
    
    Args:
        response: 模型的完整响应
        
    Returns:
        提取的SQL语句
    """
    def clean_sql(sql: str) -> str:
        """清理SQL，去掉开头的注释行"""
        lines = sql.strip().split('\n')
        # 跳过开头的空行和注释行
        start_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith('--'):
                start_idx = i
                break
        return '\n'.join(lines[start_idx:]).strip()
    
    # 尝试从 <answer> 标签中提取
    if "<answer>" in response and "</answer>" in response:
        answer = response.split("<answer>")[-1].split("</answer>")[0]
        # 如果answer中有sql代码块
        if "```sql" in answer:
            pattern = r"```sql\s*(.*?)\s*```"
            matches = re.findall(pattern, answer, re.DOTALL)
            if matches:
                return clean_sql(matches[-1])
        return clean_sql(answer)
    
    # 尝试从 ```sql 代码块中提取
    pattern = r"```sql\s*(.*?)\s*```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return clean_sql(matches[-1])
    
    # 尝试从 ``` 代码块中提取
    pattern = r"```\s*(.*?)\s*```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        sql = matches[-1].strip()
        if sql.upper().startswith("SELECT"):
            return sql
    
    # 尝试直接查找SELECT语句
    if "SELECT" in response.upper():
        # 找到最后一个SELECT语句
        lines = response.split("\n")
        sql_lines = []
        in_sql = False
        
        for line in lines:
            if "SELECT" in line.upper() and not in_sql:
                in_sql = True
                sql_lines = [line]
            elif in_sql:
                sql_lines.append(line)
                if ";" in line:
                    break
        
        if sql_lines:
            return "\n".join(sql_lines).strip().rstrip(";")
    
    return response.strip()


class TreeBuilder:
    """
    修订树构建器
    
    使用BFS算法逐层构建修订树
    """
    
    def __init__(
        self,
        model_generator: Callable,  # 模型生成函数
        db_root: str,  # 数据库根目录
        max_depth: int = 5,
        sample_num: int = 3,
        temperature: float = 0.8,
        timeout: float = 30.0,
        early_stop: bool = True,  # 找到正确答案后是否停止
        verbose: bool = True
    ):
        """
        初始化树构建器
        
        Args:
            model_generator: 模型生成函数，接收prompt列表，返回响应列表
            db_root: 数据库根目录
            max_depth: 最大树深度
            sample_num: 每个节点的采样次数
            temperature: 采样温度
            timeout: SQL执行超时时间
            early_stop: 是否在找到正确答案后停止扩展
            verbose: 是否打印详细信息
        """
        self.model_generator = model_generator
        self.db_root = db_root
        self.max_depth = max_depth
        self.sample_num = sample_num
        self.temperature = temperature
        self.early_stop = early_stop
        self.verbose = verbose
        
        # SQL执行器
        self.sql_executor = SQLExecutor(timeout=timeout)
    
    def build_tree(
        self,
        question_data: Dict,
        input_seq: str = None
    ) -> RevisionTree:
        """
        为单个问题构建修订树
        
        Args:
            question_data: 问题数据，包含 question_id, db_id, question, evidence, SQL等
            input_seq: 初始生成的input_seq（可选，如果已有则直接使用）
            
        Returns:
            构建完成的修订树
        """
        # 创建树
        tree = RevisionTree(
            question_id=str(question_data.get("question_id", "")),
            db_id=question_data.get("db_id", ""),
            question=question_data.get("question", ""),
            evidence=question_data.get("evidence", ""),
            difficulty=question_data.get("difficulty", "simple"),
            gold_sql=question_data.get("SQL", ""),
            max_depth=self.max_depth,
            sample_num=self.sample_num
        )
        
        # 设置初始节点的input_seq
        if input_seq:
            tree.node_initial.input_seq = input_seq
        else:
            tree.node_initial.input_seq = question_data.get("input_seq", "")
        
        # 获取数据库路径
        db_path = get_db_path(self.db_root, tree.db_id)
        
        if self.verbose:
            print(f"\n[Building Tree] Question ID: {tree.question_id}, DB: {tree.db_id}")
        
        # 第一步：生成根节点（深度1）
        self._generate_root_nodes(tree, db_path)
        
        # 检查是否有正确节点
        if self.early_stop and tree.has_correct_node():
            if self.verbose:
                print(f"  Found correct node at depth 1, stopping early.")
            tree.compute_revision_distances()
            return tree
        
        # BFS扩展
        self._bfs_expand(tree, db_path)
        
        # 计算修订距离
        tree.compute_revision_distances()
        
        if self.verbose:
            stats = tree.get_statistics()
            print(f"  Tree built: {stats['total_nodes']} nodes, "
                  f"{stats['correct_nodes']} correct, status={tree.tree_status}")
        
        return tree
    
    def _generate_root_nodes(self, tree: RevisionTree, db_path: str) -> None:
        """生成根节点（深度1的节点）"""
        initial_prompt = tree.node_initial.input_seq
        
        if not initial_prompt:
            if self.verbose:
                print("  Warning: No initial prompt found!")
            return
        
        # 生成多个采样
        prompts = [initial_prompt] * self.sample_num
        responses = self.model_generator(prompts, temperature=self.temperature)
        
        for response in responses:
            # 提取SQL
            generated_sql = extract_sql_from_response(response)
            
            # 创建节点
            node = tree.create_node(
                parent_id="node_initial",
                depth=1,
                generated_sql=generated_sql,
                response_text=response
            )
            
            # 执行SQL并获取反馈
            self._evaluate_node(node, tree, db_path)
    
    def _bfs_expand(self, tree: RevisionTree, db_path: str) -> None:
        """使用BFS算法扩展树"""
        # 获取schema信息（用于构建修订提示词）
        schema = SchemaLoader.get_schema_from_input_seq(tree.node_initial.input_seq)
        
        for depth in range(2, self.max_depth + 1):
            if self.verbose:
                print(f"  Expanding depth {depth}...")
            
            # 获取上一层需要扩展的节点
            parent_nodes = [
                node for node in tree.get_nodes_at_depth(depth - 1)
                if not node.is_correct
            ]
            
            if not parent_nodes:
                if self.verbose:
                    print(f"    No nodes to expand at depth {depth}")
                break
            
            # 为每个父节点生成子节点
            for parent_node in parent_nodes:
                # 构建修订提示词
                revision_prompt = PromptTemplates.build_revision_prompt(
                    schema=schema,
                    question=tree.question,
                    previous_sql=parent_node.generated_sql,
                    execution_feedback=parent_node.execution_feedback,
                    evidence=tree.evidence
                )
                
                # 保存到父节点
                parent_node.input_seq = revision_prompt
                
                # 生成多个采样
                prompts = [revision_prompt] * self.sample_num
                responses = self.model_generator(prompts, temperature=self.temperature)
                
                for response in responses:
                    # 提取SQL
                    generated_sql = extract_sql_from_response(response)
                    
                    # 创建节点
                    node = tree.create_node(
                        parent_id=parent_node.node_id,
                        depth=depth,
                        generated_sql=generated_sql,
                        response_text=response
                    )
                    
                    # 执行SQL并获取反馈
                    self._evaluate_node(node, tree, db_path)
                
                # 检查是否可以提前停止
                if self.early_stop and tree.has_correct_node():
                    if self.verbose:
                        print(f"    Found correct node at depth {depth}, stopping early.")
                    return
    
    def _evaluate_node(
        self, 
        node: TreeNode, 
        tree: RevisionTree, 
        db_path: str
    ) -> None:
        """
        评估节点的SQL执行结果
        
        Args:
            node: 要评估的节点
            tree: 所属的树
            db_path: 数据库路径
        """
        # 执行SQL并比对结果
        comparison = self.sql_executor.generate_feedback(
            pred_sql=node.generated_sql,
            gold_sql=tree.gold_sql,
            db_path=db_path
        )
        
        # 更新节点信息
        node.is_correct = comparison.is_correct
        node.execution_feedback = comparison.feedback
        node.execution_status = comparison.pred_status
        node.execution_result = comparison.pred_result


class BatchTreeBuilder:
    """
    批量树构建器
    
    用于处理整个数据集
    """
    
    def __init__(
        self,
        model_generator: Callable,
        db_root: str,
        output_dir: str,
        max_depth: int = 5,
        sample_num: int = 3,
        temperature: float = 0.8,
        timeout: float = 30.0,
        early_stop: bool = True,
        save_interval: int = 10,  # 每处理多少个保存一次
        resume: bool = True  # 是否支持断点续传
    ):
        """
        初始化批量构建器
        
        Args:
            model_generator: 模型生成函数
            db_root: 数据库根目录
            output_dir: 输出目录
            max_depth: 最大树深度
            sample_num: 每个节点的采样次数
            temperature: 采样温度
            timeout: SQL执行超时时间
            early_stop: 是否提前停止
            save_interval: 保存间隔
            resume: 是否支持断点续传
        """
        self.tree_builder = TreeBuilder(
            model_generator=model_generator,
            db_root=db_root,
            max_depth=max_depth,
            sample_num=sample_num,
            temperature=temperature,
            timeout=timeout,
            early_stop=early_stop,
            verbose=False
        )
        
        self.output_dir = output_dir
        self.save_interval = save_interval
        self.resume = resume
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
    
    def _get_processed_ids(self) -> set:
        """获取已处理的question_id集合"""
        processed = set()
        
        if not self.resume:
            return processed
        
        # 检查输出目录中的文件
        if os.path.exists(self.output_dir):
            for filename in os.listdir(self.output_dir):
                if filename.startswith("tree_") and filename.endswith(".json"):
                    qid = filename[5:-5]  # 提取question_id
                    processed.add(qid)
        
        return processed
    
    def build_trees(
        self,
        dataset: List[Dict],
        input_seqs: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        为整个数据集构建修订树
        
        Args:
            dataset: 数据集列表
            input_seqs: 对应的input_seq列表（可选）
            
        Returns:
            统计信息
        """
        processed_ids = self._get_processed_ids()
        
        trees = []
        stats = {
            "total": len(dataset),
            "processed": 0,
            "good_trees": 0,
            "bad_trees": 0,
            "skipped": 0,
            "good_tree_ids": [],  # 记录good tree的question_id
            "bad_tree_ids": [],   # 记录bad tree的question_id
            "error_ids": []       # 记录出错的question_id
        }
        
        total = len(dataset)
        
        print(f"\n{'='*70}")
        print(f"Starting to build {total} revision trees...")
        print(f"{'='*70}\n")
        
        # 使用tqdm创建进度条，显示预估时间
        pbar = tqdm(
            enumerate(dataset),
            total=total,
            desc="Building Trees",
            unit="tree",
            ncols=120,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        )
        
        for idx, data in pbar:
            question_id = str(data.get("question_id", idx))
            db_id = data.get("db_id", "unknown")
            
            # 更新进度条的后缀信息
            pbar.set_postfix({
                'qid': question_id,
                'good': stats['good_trees'],
                'bad': stats['bad_trees']
            })
            
            # 检查是否已处理
            if question_id in processed_ids:
                stats["skipped"] += 1
                continue
            
            # 获取input_seq
            input_seq = None
            if input_seqs and idx < len(input_seqs):
                input_seq = input_seqs[idx]
            
            try:
                # 构建树
                tree = self.tree_builder.build_tree(data, input_seq)
                
                # 保存单棵树
                tree_file = os.path.join(
                    self.output_dir, 
                    f"tree_{question_id}.json"
                )
                tree.save_to_file(tree_file)
                
                trees.append(tree)
                stats["processed"] += 1
                
                if tree.tree_status == "good":
                    stats["good_trees"] += 1
                    stats["good_tree_ids"].append(question_id)
                else:
                    stats["bad_trees"] += 1
                    stats["bad_tree_ids"].append(question_id)
                
                # 更新进度条后缀
                pbar.set_postfix({
                    'qid': question_id,
                    'good': stats['good_trees'],
                    'bad': stats['bad_trees']
                })
                
                # 定期保存统计信息
                if stats["processed"] % self.save_interval == 0:
                    self._save_stats(stats)
                    
            except Exception as e:
                stats["error_ids"].append({"question_id": question_id, "error": str(e)})
                tqdm.write(f"[ERROR] question_id={question_id}: {e}")
                continue
        
        pbar.close()
        
        # 打印最终统计
        print(f"\n{'='*70}")
        print(f"Build Complete!")
        print(f"  Total:      {stats['total']}")
        print(f"  Processed:  {stats['processed']}")
        print(f"  Good Trees: {stats['good_trees']} ✓")
        print(f"  Bad Trees:  {stats['bad_trees']} ✗")
        print(f"  Skipped:    {stats['skipped']}")
        print(f"  Errors:     {len(stats['error_ids'])}")
        if stats['processed'] > 0:
            success_rate = stats['good_trees'] / stats['processed'] * 100
            print(f"  Success Rate: {success_rate:.2f}%")
        print(f"{'='*70}\n")
        
        # 打印bad tree ids摘要
        if stats['bad_tree_ids']:
            print(f"Bad Tree IDs ({len(stats['bad_tree_ids'])}): {stats['bad_tree_ids'][:20]}{'...' if len(stats['bad_tree_ids']) > 20 else ''}")
            
        # 最终保存
        self._save_stats(stats)
        self._save_all_trees(trees)
        
        return stats
    
    def _save_stats(self, stats: Dict) -> None:
        """保存统计信息"""
        import json
        from datetime import datetime
        
        # 添加时间戳
        stats_with_timestamp = stats.copy()
        stats_with_timestamp["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 计算成功率
        if stats["processed"] > 0:
            stats_with_timestamp["success_rate"] = f"{stats['good_trees'] / stats['processed'] * 100:.2f}%"
            
        stats_file = os.path.join(self.output_dir, "build_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_with_timestamp, f, indent=2, ensure_ascii=False)
    
    def _save_all_trees(self, trees: List[RevisionTree]) -> None:
        """保存所有树到一个文件"""
        import json
        all_trees_file = os.path.join(self.output_dir, "all_trees.json")
        all_trees_data = [tree.to_dict() for tree in trees]
        with open(all_trees_file, 'w', encoding='utf-8') as f:
            json.dump(all_trees_data, f, ensure_ascii=False, indent=2)

class ParallelBatchTreeBuilder(BatchTreeBuilder):
    """
    ????????????????????????
    
    ???????????????????????????????????????????????????GPU????????????
    ??????????????????????????????????????????Batch?????????LLM???
    """
    
    def __init__(
        self,
        model_generator: Callable,
        db_root: str,
        output_dir: str,
        max_depth: int = 5,
        sample_num: int = 3,
        temperature: float = 0.8,
        timeout: float = 30.0,
        early_stop: bool = True,
        save_interval: int = 10,
        resume: bool = True,
        batch_size: int = 16  # ???????????????????????????
    ):
        super().__init__(
            model_generator, db_root, output_dir, max_depth, 
            sample_num, temperature, timeout, early_stop, 
            save_interval, resume
        )
        self.batch_size = batch_size
        self.sql_executor = SQLExecutor(timeout=timeout)

    def build_trees(
        self,
        dataset: List[Dict],
        input_seqs: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        ???????????????
        """
        processed_ids = self._get_processed_ids()
        
        # ???????????????????????????
        pending_data = []
        for idx, data in enumerate(dataset):
            question_id = str(data.get("question_id", idx))
            if question_id in processed_ids:
                continue
            
            input_seq = None
            if input_seqs and idx < len(input_seqs):
                input_seq = input_seqs[idx]
            else:
                input_seq = data.get("input_seq", "")
                
            pending_data.append({
                "data": data,
                "input_seq": input_seq,
                "question_id": question_id
            })
            
        stats = {
            "total": len(dataset),
            "processed": 0,
            "good_trees": 0,
            "bad_trees": 0,
            "skipped": len(dataset) - len(pending_data)
        }
        
        # ???????????????????????????
        active_contexts = []
        # ????????????????????????
        data_queue = deque(pending_data)
        # ??????????????????
        completed_trees = []
        
        pbar = tqdm(total=len(pending_data), desc="Building trees (Parallel)")
        
        try:
            while active_contexts or data_queue:
                # 1. ?????????????????????
                while len(active_contexts) < self.batch_size and data_queue:
                    item = data_queue.popleft()
                    context = self._init_tree_context(item)
                    active_contexts.append(context)
                
                if not active_contexts:
                    break
                    
                # 2. ?????????????????????Prompts
                generation_requests = []
                for ctx in active_contexts:
                    prompts = self._get_next_prompts(ctx)
                    if prompts:
                        generation_requests.append({
                            "ctx": ctx,
                            "prompts": prompts
                        })
                
                if not generation_requests:
                    # ????????????????????????????????????????????????????????????????????????
                    finished_indices = []
                    for i, ctx in enumerate(active_contexts):
                        if ctx["status"] == "finished":
                            finished_indices.append(i)
                    
                    # ??????????????????????????????????????????
                    for i in sorted(finished_indices, reverse=True):
                        self._finalize_tree(active_contexts[i], stats, completed_trees)
                        active_contexts.pop(i)
                        pbar.update(1)
                    continue

                # 3. ????????????
                all_prompts = []
                for req in generation_requests:
                    all_prompts.extend(req["prompts"])
                
                # ??????????????????
                all_responses = []
                if all_prompts:
                    try:
                        all_responses = self.tree_builder.model_generator(
                            all_prompts, 
                            temperature=self.tree_builder.temperature
                        )
                    except Exception as e:
                        print(f"Error during batch generation: {e}")
                        # Fallback: try one by one to isolate the error
                        all_responses = []
                        for req in generation_requests:
                            req_prompts = req["prompts"]
                            try:
                                req_responses = self.tree_builder.model_generator(
                                    req_prompts,
                                    temperature=self.tree_builder.temperature
                                )
                                all_responses.extend(req_responses)
                            except Exception as inner_e:
                                print(f"Error generating for specific request (Question ID: {req['ctx']['tree'].question_id}): {inner_e}")
                                # Append error messages to keep alignment
                                error_msg = f"Error: Generation failed - {str(inner_e)}"
                                all_responses.extend([error_msg] * len(req_prompts))
                else:
                    all_responses = []
                
                # 4. ?????????????????????
                resp_idx = 0
                finished_indices = []
                
                for i, ctx in enumerate(active_contexts):
                    # ?????????????????????
                    req = next((r for r in generation_requests if r["ctx"] is ctx), None)
                    
                    if req:
                        num_prompts = len(req["prompts"])
                        responses = all_responses[resp_idx : resp_idx + num_prompts]
                        resp_idx += num_prompts
                        
                        # ?????????????????????????????????SQL??????
                        self._process_responses(ctx, responses)
                    
                    # ??????????????????
                    if ctx["status"] == "finished":
                        finished_indices.append(i)
                
                # 5. ????????????????????????
                for i in sorted(finished_indices, reverse=True):
                    self._finalize_tree(active_contexts[i], stats, completed_trees)
                    active_contexts.pop(i)
                    pbar.update(1)
                    
                    # ????????????
                    if stats["processed"] % self.save_interval == 0:
                        self._save_stats(stats)
        finally:
            pbar.close()
            
            # ????????????
            self._save_stats(stats)
            self._save_all_trees(completed_trees)
        
        return stats

    def _init_tree_context(self, item: Dict) -> Dict:
        """????????????????????????"""
        data = item["data"]
        tree = RevisionTree(
            question_id=str(data.get("question_id", "")),
            db_id=data.get("db_id", ""),
            question=data.get("question", ""),
            evidence=data.get("evidence", ""),
            difficulty=data.get("difficulty", "simple"),
            gold_sql=data.get("SQL", ""),
            max_depth=self.tree_builder.max_depth,
            sample_num=self.tree_builder.sample_num
        )
        tree.node_initial.input_seq = item["input_seq"]
        
        return {
            "tree": tree,
            "db_path": get_db_path(self.tree_builder.db_root, tree.db_id),
            "status": "init",  # init, expanding, finished
            "current_depth": 1,
            "nodes_to_expand": [], # ??????????????????????????????
            "schema": None # ??????schema
        }

    def _get_next_prompts(self, ctx: Dict) -> List[str]:
        """??????????????????????????????Prompts"""
        tree = ctx["tree"]
        status = ctx["status"]
        
        if status == "finished":
            return []
            
        prompts = []
        
        if status == "init":
            # ???????????????
            initial_prompt = tree.node_initial.input_seq
            if not initial_prompt:
                ctx["status"] = "finished"
                return []
            prompts = [initial_prompt] * self.tree_builder.sample_num
            ctx["pending_action"] = "generate_root"
            
        elif status == "expanding":
            # ????????????
            depth = ctx["current_depth"]
            if depth > self.tree_builder.max_depth:
                ctx["status"] = "finished"
                return []
                
            # ???????????????????????????
            if not ctx["nodes_to_expand"]:
                # ????????????????????????????????????
                parent_nodes = [
                    node for node in tree.get_nodes_at_depth(depth - 1)
                    if not node.is_correct
                ]
                if not parent_nodes:
                    ctx["status"] = "finished"
                    return []
                ctx["nodes_to_expand"] = parent_nodes
                
                # ??????Schema????????????????????????
                if ctx["schema"] is None:
                    ctx["schema"] = SchemaLoader.get_schema_from_input_seq(tree.node_initial.input_seq)
            
            # ??????????????????????????????Prompt
            # ????????????????????????????????????????????????Prompt
            for parent_node in ctx["nodes_to_expand"]:
                revision_prompt = PromptTemplates.build_revision_prompt(
                    schema=ctx["schema"],
                    question=tree.question,
                    previous_sql=parent_node.generated_sql,
                    execution_feedback=parent_node.execution_feedback,
                    evidence=tree.evidence
                )
                # ??????Prompt??????????????????????????????
                parent_node.input_seq = revision_prompt
                
                prompts.extend([revision_prompt] * self.tree_builder.sample_num)
            
            ctx["pending_action"] = "expand_layer"
            
        return prompts

    def _process_responses(self, ctx: Dict, responses: List[str]) -> None:
        """??????????????????"""
        tree = ctx["tree"]
        action = ctx.get("pending_action")
        db_path = ctx["db_path"]
        
        if action == "generate_root":
            # ???????????????????????????
            for response in responses:
                generated_sql = extract_sql_from_response(response)
                node = tree.create_node(
                    parent_id="node_initial",
                    depth=1,
                    generated_sql=generated_sql,
                    response_text=response
                )
                self.tree_builder._evaluate_node(node, tree, db_path)
            
            # ????????????????????????
            if self.tree_builder.early_stop and tree.has_correct_node():
                ctx["status"] = "finished"
            else:
                ctx["status"] = "expanding"
                ctx["current_depth"] = 2
                ctx["nodes_to_expand"] = [] # ???????????????????????? _get_next_prompts ???????????????
                
        elif action == "expand_layer":
            # ?????????????????????
            # responses ????????????????????????[node1_sample1, node1_sample2, ..., node2_sample1, ...]
            sample_num = self.tree_builder.sample_num
            nodes_to_expand = ctx["nodes_to_expand"]
            
            resp_idx = 0
            for parent_node in nodes_to_expand:
                node_responses = responses[resp_idx : resp_idx + sample_num]
                resp_idx += sample_num
                
                for response in node_responses:
                    generated_sql = extract_sql_from_response(response)
                    node = tree.create_node(
                        parent_id=parent_node.node_id,
                        depth=ctx["current_depth"],
                        generated_sql=generated_sql,
                        response_text=response
                    )
                    self.tree_builder._evaluate_node(node, tree, db_path)
            
            # ????????????????????????
            if self.tree_builder.early_stop and tree.has_correct_node():
                ctx["status"] = "finished"
            else:
                # ???????????????
                ctx["current_depth"] += 1
                ctx["nodes_to_expand"] = []
                if ctx["current_depth"] > self.tree_builder.max_depth:
                    ctx["status"] = "finished"

    def _finalize_tree(self, ctx: Dict, stats: Dict, completed_trees: List[RevisionTree]) -> None:
        """??????????????????????????????????????????"""
        tree = ctx["tree"]
        tree.compute_revision_distances()
        
        # ??????
        tree_file = os.path.join(
            self.output_dir, 
            f"tree_{tree.question_id}.json"
        )
        tree.save_to_file(tree_file)
        
        completed_trees.append(tree)
        stats["processed"] += 1
        
        if tree.tree_status == "good":
            stats["good_trees"] += 1
        else:
            stats["bad_trees"] += 1
