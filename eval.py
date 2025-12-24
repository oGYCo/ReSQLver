"""
验证树数据集中correct节点的SQL正确性

功能：
1. 读取生成的树JSON文件
2. 检查所有标记为is_correct=True的节点
3. 验证这些节点的SQL执行结果是否与gold SQL一致
"""

import os
import sys
import json
import argparse
import sqlite3
from typing import Dict, List, Any, Tuple
from func_timeout import func_timeout, FunctionTimedOut


def execute_sql(predicted_sql: str, ground_truth: str, db_path: str) -> int:
    """
    执行SQL并比对结果
    
    Args:
        predicted_sql: 预测的SQL
        ground_truth: 标准答案SQL
        db_path: 数据库路径
        
    Returns:
        1 如果结果一致，0 如果不一致
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()
    cursor.execute(ground_truth)
    ground_truth_res = cursor.fetchall()
    conn.close()
    
    res = 0
    if set(predicted_res) == set(ground_truth_res):
        res = 1
    return res


def safe_execute_sql(predicted_sql: str, ground_truth: str, db_path: str, timeout: float = 30.0) -> Tuple[int, str]:
    """
    安全执行SQL（带超时和异常处理）
    
    Returns:
        (result, message): result为1表示正确，0表示错误；message为详细信息
    """
    try:
        res = func_timeout(timeout, execute_sql, args=(predicted_sql, ground_truth, db_path))
        if res == 1:
            return 1, "MATCH"
        else:
            return 0, "MISMATCH"
    except FunctionTimedOut:
        return 0, "TIMEOUT"
    except sqlite3.Error as e:
        return 0, f"SQL_ERROR: {str(e)}"
    except Exception as e:
        return 0, f"ERROR: {str(e)}"


def load_tree_data(file_path: str) -> List[Dict]:
    """加载树数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 如果是单个树，转换为列表
    if isinstance(data, dict):
        return [data]
    return data


def get_correct_nodes(tree_data: Dict) -> List[Dict]:
    """获取树中所有标记为correct的节点"""
    correct_nodes = []
    nodes = tree_data.get("nodes", {})
    
    for node_id, node_info in nodes.items():
        if node_info.get("is_correct", False):
            correct_nodes.append({
                "node_id": node_id,
                "depth": node_info.get("depth", -1),
                "generated_sql": node_info.get("generated_sql", ""),
                "execution_status": node_info.get("execution_status", "")
            })
    
    return correct_nodes


def verify_tree(
    tree_data: Dict, 
    db_root: str, 
    timeout: float = 30.0,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    验证单棵树中所有correct节点
    
    Returns:
        验证结果
    """
    question_id = tree_data.get("question_id", "unknown")
    db_id = tree_data.get("db_id", "")
    gold_sql = tree_data.get("gold_sql", "")
    if not gold_sql:
        gold_sql = tree_data.get("SQL", "")
    tree_status = tree_data.get("tree_status", "unknown")
    
    db_path = os.path.join(db_root, db_id, f"{db_id}.sqlite")
    
    # 获取correct节点
    correct_nodes = get_correct_nodes(tree_data)
    
    result = {
        "question_id": question_id,
        "db_id": db_id,
        "tree_status": tree_status,
        "total_correct_nodes": len(correct_nodes),
        "verified_correct": 0,
        "verified_wrong": 0,
        "details": []
    }
    
    if not correct_nodes:
        if verbose:
            print(f"  [Q{question_id}] No correct nodes found (tree_status={tree_status})")
        return result
    
    # 验证每个correct节点
    for node in correct_nodes:
        node_id = node["node_id"]
        generated_sql = node["generated_sql"]
        
        if not generated_sql:
            result["verified_wrong"] += 1
            result["details"].append({
                "node_id": node_id,
                "status": "NO_SQL",
                "message": "Empty SQL"
            })
            continue
        
        # 执行验证
        res, message = safe_execute_sql(generated_sql, gold_sql, db_path, timeout)
        
        if res == 1:
            result["verified_correct"] += 1
            result["details"].append({
                "node_id": node_id,
                "depth": node["depth"],
                "status": "VERIFIED_CORRECT",
                "message": message
            })
        else:
            result["verified_wrong"] += 1
            result["details"].append({
                "node_id": node_id,
                "depth": node["depth"],
                "status": "VERIFIED_WRONG",
                "message": message,
                "generated_sql": generated_sql[:200] + "..." if len(generated_sql) > 200 else generated_sql
            })
    
    if verbose:
        status_icon = "✓" if result["verified_wrong"] == 0 else "✗"
        print(f"  [Q{question_id}] {status_icon} correct_nodes={result['total_correct_nodes']}, "
              f"verified_correct={result['verified_correct']}, verified_wrong={result['verified_wrong']}")
    
    return result


def verify_all_trees(
    tree_file: str,
    db_root: str,
    output_file: str = None,
    timeout: float = 30.0,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    验证所有树
    
    Args:
        tree_file: 树JSON文件路径（单个树或all_trees.json）
        db_root: 数据库根目录
        output_file: 输出报告文件路径
        timeout: SQL执行超时时间
        verbose: 是否打印详细信息
        
    Returns:
        汇总统计
    """
    print("=" * 70)
    print("Tree Correct Node Verification")
    print("=" * 70)
    print(f"Tree file: {tree_file}")
    print(f"DB root: {db_root}")
    print("=" * 70)
    
    # 加载数据
    trees = load_tree_data(tree_file)
    print(f"\nLoaded {len(trees)} trees")
    
    # 统计
    total_stats = {
        "total_trees": len(trees),
        "trees_with_correct_nodes": 0,
        "total_correct_nodes": 0,
        "verified_correct": 0,
        "verified_wrong": 0,
        "problematic_trees": [],
        "tree_results": []
    }
    
    print("\nVerifying trees...")
    print("-" * 70)
    
    for tree_data in trees:
        result = verify_tree(tree_data, db_root, timeout, verbose)
        total_stats["tree_results"].append(result)
        
        if result["total_correct_nodes"] > 0:
            total_stats["trees_with_correct_nodes"] += 1
            total_stats["total_correct_nodes"] += result["total_correct_nodes"]
            total_stats["verified_correct"] += result["verified_correct"]
            total_stats["verified_wrong"] += result["verified_wrong"]
            
            # 记录有问题的树
            if result["verified_wrong"] > 0:
                total_stats["problematic_trees"].append({
                    "question_id": result["question_id"],
                    "db_id": result["db_id"],
                    "verified_wrong": result["verified_wrong"],
                    "details": [d for d in result["details"] if d["status"] == "VERIFIED_WRONG"]
                })
    
    # 打印汇总
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"Total trees:                 {total_stats['total_trees']}")
    print(f"Trees with correct nodes:    {total_stats['trees_with_correct_nodes']}")
    print(f"Total correct nodes:         {total_stats['total_correct_nodes']}")
    print(f"Verified correct:            {total_stats['verified_correct']} ✓")
    print(f"Verified WRONG:              {total_stats['verified_wrong']} ✗")
    
    if total_stats['total_correct_nodes'] > 0:
        accuracy = total_stats['verified_correct'] / total_stats['total_correct_nodes'] * 100
        print(f"Verification accuracy:       {accuracy:.2f}%")
    
    # 打印有问题的树
    if total_stats["problematic_trees"]:
        print("\n" + "-" * 70)
        print(f"PROBLEMATIC TREES ({len(total_stats['problematic_trees'])} trees with wrong 'correct' nodes):")
        print("-" * 70)
        for pt in total_stats["problematic_trees"][:10]:  # 只显示前10个
            print(f"\n  Question ID: {pt['question_id']}, DB: {pt['db_id']}")
            for detail in pt["details"][:3]:  # 每棵树只显示前3个问题节点
                print(f"    - Node: {detail['node_id']}, Error: {detail['message']}")
                if "generated_sql" in detail:
                    print(f"      SQL: {detail['generated_sql'][:100]}...")
        
        if len(total_stats["problematic_trees"]) > 10:
            print(f"\n  ... and {len(total_stats['problematic_trees']) - 10} more problematic trees")
    else:
        print("\n✓ All correct nodes verified successfully!")
    
    print("=" * 70)
    
    # 保存报告
    if output_file:
        # 简化输出（不包含所有详情）
        summary = {
            "total_trees": total_stats["total_trees"],
            "trees_with_correct_nodes": total_stats["trees_with_correct_nodes"],
            "total_correct_nodes": total_stats["total_correct_nodes"],
            "verified_correct": total_stats["verified_correct"],
            "verified_wrong": total_stats["verified_wrong"],
            "accuracy": total_stats['verified_correct'] / total_stats['total_correct_nodes'] * 100 if total_stats['total_correct_nodes'] > 0 else 0,
            "problematic_trees": total_stats["problematic_trees"]
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nReport saved to: {output_file}")
    
    return total_stats


def main():
    parser = argparse.ArgumentParser(description="Verify correct nodes in tree dataset")
    parser.add_argument("--tree_file", type=str, required=True,
                        help="Path to tree JSON file (single tree or all_trees.json)")
    parser.add_argument("--db_root", type=str, required=True,
                        help="Path to database root directory")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to output report file")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="SQL execution timeout in seconds")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress detailed output")
    
    args = parser.parse_args()
    
    verify_all_trees(
        tree_file=args.tree_file,
        db_root=args.db_root,
        output_file=args.output,
        timeout=args.timeout,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
