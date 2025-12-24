"""
SQL执行器模块

功能：
1. 在SQLite数据库中执行SQL
2. 比对预测结果与标准答案
3. 生成详细的执行反馈信息（包含错误分析和修复建议）
"""

import sqlite3
import sys
import os
import re
from typing import Tuple, Optional, List, Any, Dict
from dataclasses import dataclass
from dataclasses import dataclass, field
from func_timeout import func_timeout, FunctionTimedOut


@dataclass
class ExecutionResult:
    """SQL执行结果"""
    success: bool  # 是否执行成功
    result: Any  # 执行结果（查询返回的数据）
    error_message: str  # 错误信息
    status: str  # success, error, timeout
    column_names: List[str] = field(default_factory=list)  # 列名
    row_count: int = 0  # 行数
    

@dataclass
class ComparisonResult:
    """SQL比对结果"""
    is_correct: bool  # 是否正确
    feedback: str  # 反馈信息
    pred_result: Any  # 预测结果
    gold_result: Any  # 标准答案结果
    pred_status: str  # 预测SQL的执行状态
    gold_status: str  # 标准SQL的执行状态
    error_type: str = ""  # 错误类型
    diff_analysis: str = ""  # 差异分析


class SQLExecutor:
    """
    SQL执行器
    
    支持：
    - 超时控制
    - 只读执行
    - 结果比对
    - 反馈生成
    """
    
    # 禁止执行的SQL关键词
    FORBIDDEN_KEYWORDS = [
        'DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 
        'CREATE', 'TRUNCATE', 'REPLACE', 'MERGE'
    ]
    
    def __init__(self, timeout: float = 30.0, max_result_display: int = 10):
        """
        初始化执行器
        
        Args:
            timeout: 执行超时时间（秒）
            max_result_display: 反馈中显示的最大结果条数
        """
        self.timeout = timeout
        self.max_result_display = max_result_display
    
    def _is_safe_sql(self, sql: str) -> Tuple[bool, str]:
        """
        检查SQL是否安全（不包含危险操作）
        
        Args:
            sql: 待检查的SQL
            
        Returns:
            (是否安全, 错误信息)
        """
        sql_upper = sql.upper()
        for keyword in self.FORBIDDEN_KEYWORDS:
            if keyword in sql_upper:
                return False, f"SQL contains forbidden keyword: {keyword}"
        return True, ""
    
    def _execute_sql_internal(
        self, 
        db_path: str, 
        sql: str
    ) -> ExecutionResult:
        """
        内部SQL执行方法（无超时控制）
        
        Args:
            db_path: 数据库路径
            sql: SQL语句
            
        Returns:
            执行结果
        """
        # 安全检查
        is_safe, error_msg = self._is_safe_sql(sql)
        if not is_safe:
            return ExecutionResult(
                success=False,
                result=None,
                error_message=error_msg,
                status="error"
            )
        
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute(sql)
            result = cursor.fetchall()
            
            # 获取列名
            column_names = [desc[0] for desc in cursor.description] if cursor.description else []
            
            return ExecutionResult(
                success=True,
                result=result,
                error_message="",
                status="success",
                column_names=column_names,
                row_count=len(result) if result else 0
            )
            
        except sqlite3.Error as e:
            return ExecutionResult(
                success=False,
                result=None,
                error_message=str(e),
                status="error"
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                result=None,
                error_message=str(e),
                status="error"
            )
        finally:
            if conn:
                conn.close()
    def execute_sql(
        self, 
        db_path: str, 
        sql: str,
        timeout: Optional[float] = None
    ) -> ExecutionResult:
        """
        执行SQL（带超时控制）
        
        Args:
            db_path: 数据库路径
            sql: SQL语句
            timeout: 超时时间，None则使用默认值
            
        Returns:
            执行结果
        """
        timeout = timeout or self.timeout
        
        try:
            result = func_timeout(
                timeout, 
                self._execute_sql_internal, 
                args=(db_path, sql)
            )
            return result
            
        except FunctionTimedOut:
            return ExecutionResult(
                success=False,
                result=None,
                error_message=f"SQL execution timed out after {timeout} seconds",
                status="timeout"
            )
        except KeyboardInterrupt:
            sys.exit(0)
        except Exception as e:
            return ExecutionResult(
                success=False,
                result=None,
                error_message=str(e),
                status="error"
            )
    
    def compare_results(
        self,
        pred_result: Any,
        gold_result: Any,
        ignore_order: bool = True
    ) -> bool:
        """
        比对两个SQL的执行结果
        与evaluation.py中的execute_sql函数完全一致：
        
        res = 0
        if set(predicted_res) == set(ground_truth_res):
            res = 1
        return res
        Args:
            pred_result: 预测SQL的执行结果
            gold_result: 标准SQL的执行结果
            ignore_order: 是否忽略顺序（保留参数但不使用，保持与evaluation.py一致）
            
        Returns:
            是否相等
        """
        # 与 evaluation.py 完全一致的判断逻辑
        if pred_result is None or gold_result is None:
            return False
            
        # evaluation.py: set(predicted_res) == set(ground_truth_res)
        return set(pred_result) == set(gold_result)
    
    def _format_result_for_display(self, result: Any, column_names: List[str] = None) -> str:
        """格式化结果用于显示"""
        if result is None:
            return "None"
        
        if not result:
            return "Empty result set (0 rows)"
        
        # 添加列名信息
        header = ""
        if column_names:
            header = f"Columns: {column_names}\n"
        
        # 截断过长的字符串值
        def truncate_val(val):
            s = str(val)
            if len(s) > 100:
                return s[:100] + "..."
            return val
            
        truncated_result = []
        for row in result[:self.max_result_display]:
            truncated_row = tuple(truncate_val(v) for v in row)
            truncated_result.append(truncated_row)
        
        row_count = len(result)
        if row_count <= self.max_result_display:
            return f"{header}Rows ({row_count}): {truncated_result}"
        else:
            return f"{header}Rows ({row_count}, showing first {self.max_result_display}): {truncated_result}"
    def _analyze_result_difference(
        self,
        pred_result: Any,
        gold_result: Any,
        pred_columns: List[str] = None,
        gold_columns: List[str] = None
    ) -> str:
        """
        详细分析预测结果与标准答案的差异
        
        Returns:
            差异分析字符串
        """
        analysis_parts = []
        
        # 1. 行数差异
        pred_rows = len(pred_result) if pred_result else 0
        gold_rows = len(gold_result) if gold_result else 0
        
        if pred_rows != gold_rows:
            analysis_parts.append(
                f"[Row Count Mismatch] Expected {gold_rows} rows, but got {pred_rows} rows."
            )
            if pred_rows > gold_rows:
                analysis_parts.append(
                    f"  -> Your query returns {pred_rows - gold_rows} extra rows. "
                    "Check if you need stricter WHERE conditions or are missing a DISTINCT."
                )
            else:
                analysis_parts.append(
                    f"  -> Your query is missing {gold_rows - pred_rows} rows. "
                    "Check if your WHERE conditions are too restrictive or JOINs are dropping rows."
                )
        
        # 2. 列数差异
        if pred_result and gold_result:
            pred_cols = len(pred_result[0]) if pred_result[0] else 0
            gold_cols = len(gold_result[0]) if gold_result[0] else 0
            
            if pred_cols != gold_cols:
                analysis_parts.append(
                    f"[Column Count Mismatch] Expected {gold_cols} columns, but got {pred_cols} columns."
                )
                if pred_cols > gold_cols:
                    analysis_parts.append(
                        "  -> Your query returns extra columns. Remove unnecessary columns from SELECT."
                    )
                else:
                    analysis_parts.append(
                        "  -> Your query is missing columns. Add the required columns to SELECT."
                    )
        
        # 3. 值差异分析
        if pred_result and gold_result and pred_rows == gold_rows:
            try:
                pred_set = set(map(tuple, pred_result))
                gold_set = set(map(tuple, gold_result))
                
                missing_rows = gold_set - pred_set
                extra_rows = pred_set - gold_set
                
                if missing_rows:
                    sample_missing = list(missing_rows)[:3]
                    analysis_parts.append(
                        f"[Missing Values] {len(missing_rows)} expected rows not found in your result."
                    )
                    analysis_parts.append(f"  -> Sample missing rows: {sample_missing}")
                
                if extra_rows:
                    sample_extra = list(extra_rows)[:3]
                    analysis_parts.append(
                        f"[Extra Values] {len(extra_rows)} unexpected rows in your result."
                    )
                    analysis_parts.append(f"  -> Sample extra rows: {sample_extra}")
                
                # 检查是否只是顺序不同
                if not missing_rows and not extra_rows and pred_result != gold_result:
                    analysis_parts.append(
                        "[Order Difference] The values are correct but in different order. "
                        "If ORDER BY is required, check your sorting logic."
                    )
                    
            except (TypeError, ValueError):
                pass
        
        # 4. 空结果分析
        if pred_rows == 0 and gold_rows > 0:
            analysis_parts.append(
                "[Empty Result] Your query returned no rows but should return data.\n"
                "  -> Common causes:\n"
                "     * WHERE condition is too restrictive or has wrong values\n"
                "     * JOIN condition doesn't match any rows\n"
                "     * Column name or table name is misspelled"
            )
        elif pred_rows > 0 and gold_rows == 0:
            analysis_parts.append(
                "[Should Be Empty] Your query returned rows but should return no data.\n"
                "  -> Check if your WHERE conditions correctly filter out all rows."
            )
        
        return "\n".join(analysis_parts) if analysis_parts else "Results differ in values."
    
    def _generate_error_fix_suggestions(self, error_message: str, sql: str) -> str:
        """
        根据错误信息生成修复建议
        
        Args:
            error_message: 错误信息
            sql: 出错的SQL
            
        Returns:
            修复建议
        """
        suggestions = []
        error_lower = error_message.lower()
        
        if "no such table" in error_lower:
            # 提取错误的表名
            match = re.search(r"no such table:\s*(\w+)", error_lower)
            table_name = match.group(1) if match else "unknown"
            suggestions.append(
                f"[Table Not Found] The table '{table_name}' does not exist.\n"
                "  Fix suggestions:\n"
                "  1. Check the exact table name in the schema (case-sensitive)\n"
                "  2. Verify you're using the correct database\n"
                "  3. Table names in schema are the definitive source"
            )
            
        elif "no such column" in error_lower:
            match = re.search(r"no such column:\s*(\S+)", error_lower)
            col_name = match.group(1) if match else "unknown"
            suggestions.append(
                f"[Column Not Found] The column '{col_name}' does not exist.\n"
                "  Fix suggestions:\n"
                "  1. Check column names in the table schema exactly\n"
                "  2. If using alias, ensure the alias is defined\n"
                "  3. For JOINs, prefix column with table name (e.g., table.column)\n"
                "  4. Column names are case-sensitive in SQLite"
            )
            
        elif "ambiguous column" in error_lower:
            match = re.search(r"ambiguous column[^:]*:\s*(\S+)", error_lower)
            col_name = match.group(1) if match else "unknown"
            suggestions.append(
                f"[Ambiguous Column] The column '{col_name}' exists in multiple tables.\n"
                "  Fix: Prefix with table name or alias, e.g., T1.{col_name} or table_name.{col_name}"
            )
            
        elif "syntax error" in error_lower:
            suggestions.append(
                "[Syntax Error] The SQL has invalid syntax.\n"
                "  Common fixes:\n"
                "  1. Check for missing commas between columns\n"
                "  2. Verify parentheses are balanced\n"
                "  3. Ensure keywords are spelled correctly\n"
                "  4. String literals should use single quotes 'value'\n"
                "  5. Check for missing or extra keywords (SELECT, FROM, WHERE, etc.)"
            )
            # 尝试定位语法错误位置
            if "near" in error_lower:
                match = re.search(r'near "([^"]+)"', error_lower)
                if match:
                    suggestions.append(f"  -> Error is near: '{match.group(1)}'")
                    
        elif "misuse of aggregate" in error_lower:
            suggestions.append(
                "[Aggregate Misuse] Aggregate function used incorrectly.\n"
                "  Fix suggestions:\n"
                "  1. Add GROUP BY clause for non-aggregated columns in SELECT\n"
                "  2. Or remove the aggregate function if grouping is not needed\n"
                "  3. Use HAVING instead of WHERE for aggregate conditions"
            )
            
        elif "no such function" in error_lower:
            match = re.search(r"no such function:\s*(\w+)", error_lower)
            func_name = match.group(1) if match else "unknown"
            suggestions.append(
                f"[Function Not Found] '{func_name}' is not a valid SQLite function.\n"
                "  Note: SQLite has limited functions compared to other databases.\n"
                "  Common alternatives:\n"
                "  - ISNULL() -> Use IFNULL() or COALESCE()\n"
                "  - NVL() -> Use IFNULL() or COALESCE()\n"
                "  - DATEADD() -> Use date() with modifiers\n"
                "  - TOP N -> Use LIMIT N"
            )
            
        elif "timeout" in error_lower:
            suggestions.append(
                "[Query Timeout] The query took too long to execute.\n"
                "  Fix suggestions:\n"
                "  1. Add appropriate WHERE conditions to limit data\n"
                "  2. Avoid cartesian products (ensure proper JOIN conditions)\n"
                "  3. Simplify complex subqueries\n"
                "  4. Check if DISTINCT or GROUP BY is causing performance issues"
            )
            
        elif "datatype mismatch" in error_lower:
            suggestions.append(
                "[Type Mismatch] Comparing or operating on incompatible data types.\n"
                "  Fix suggestions:\n"
                "  1. Use CAST() to convert types: CAST(column AS INTEGER)\n"
                "  2. Ensure string comparisons use quotes\n"
                "  3. Check if numeric comparison is accidentally comparing strings"
            )
        
        if not suggestions:
            suggestions.append(
                f"[Error] {error_message}\n"
                "  Please check:\n"
                "  1. All table and column names match the schema exactly\n"
                "  2. SQL syntax is valid for SQLite\n"
                "  3. Data types are compatible in comparisons"
            )
        
        return "\n".join(suggestions)
    
    def generate_feedback(
        self,
        pred_sql: str,
        gold_sql: str,
        db_path: str
    ) -> ComparisonResult:
        """
        执行SQL并生成详细的反馈信息
        
        Args:
            pred_sql: 预测的SQL
            gold_sql: 标准答案SQL
            db_path: 数据库路径
            
        Returns:
            比对结果（包含详细反馈信息）
        """
        # 执行预测SQL
        pred_exec = self.execute_sql(db_path, pred_sql)
        
        # 执行标准SQL
        gold_exec = self.execute_sql(db_path, gold_sql)
        
        # 如果预测SQL执行失败
        if not pred_exec.success:
            error_type = self._categorize_error(pred_exec.error_message)
            fix_suggestions = self._generate_error_fix_suggestions(pred_exec.error_message, pred_sql)
            
            feedback = (
                f"=== EXECUTION ERROR ===\n"
                f"Status: {pred_exec.status}\n"
                f"Error Type: {error_type}\n"
                f"Error Message: {pred_exec.error_message}\n\n"
                f"=== FIX SUGGESTIONS ===\n"
                f"{fix_suggestions}"
            )
            
            return ComparisonResult(
                is_correct=False,
                feedback=feedback,
                pred_result=None,
                gold_result=gold_exec.result,
                pred_status=pred_exec.status,
                gold_status=gold_exec.status,
                error_type=error_type,
                diff_analysis=""
            )
        
        # 如果标准SQL执行失败（数据问题）
        if not gold_exec.success:
            feedback = f"Warning: Gold SQL execution failed: {gold_exec.error_message}"
            return ComparisonResult(
                is_correct=False,
                feedback=feedback,
                pred_result=pred_exec.result,
                gold_result=None,
                pred_status=pred_exec.status,
                gold_status=gold_exec.status,
                error_type="gold_error",
                diff_analysis=""
            )
        
        # 比对结果
        is_correct = self.compare_results(pred_exec.result, gold_exec.result)
        
        if is_correct:
            feedback = (
                f"Your query returned the expected result.\n"
                f"Rows: {pred_exec.row_count}"
            )
            return ComparisonResult(
                is_correct=True,
                feedback=feedback,
                pred_result=pred_exec.result,
                gold_result=gold_exec.result,
                pred_status=pred_exec.status,
                gold_status=gold_exec.status,
                error_type="",
                diff_analysis=""
            )
        
        # 结果不正确，生成详细差异分析
        diff_analysis = self._analyze_result_difference(
            pred_exec.result, 
            gold_exec.result,
            pred_exec.column_names,
            gold_exec.column_names
        )
        
        pred_display = self._format_result_for_display(pred_exec.result, pred_exec.column_names)
        gold_display = self._format_result_for_display(gold_exec.result, gold_exec.column_names)
        
        feedback = (
            f"=== INCORRECT RESULT ===\n"
            f"Your query executed successfully but returned wrong results.\n\n"
            f"=== EXPECTED OUTPUT ===\n{gold_display}\n\n"
            f"=== YOUR OUTPUT ===\n{pred_display}\n\n"
            f"=== DIFFERENCE ANALYSIS ===\n{diff_analysis}\n\n"
            f"=== HINTS ===\n"
            f"1. Compare your SELECT columns with the expected columns\n"
            f"2. Check your WHERE/JOIN conditions\n"
            f"3. Verify GROUP BY and aggregate functions if used\n"
            f"4. Consider if DISTINCT is needed"
        )
        
        # 截断过长的反馈信息
        if len(feedback) > 4000:
            feedback = feedback[:4000] + "\n... (truncated)"
            
        return ComparisonResult(
            is_correct=False,
            feedback=feedback,
            pred_result=pred_exec.result,
            gold_result=gold_exec.result,
            pred_status=pred_exec.status,
            gold_status=gold_exec.status,
            error_type="wrong_result",
            diff_analysis=diff_analysis
        )
    
    def _categorize_error(self, error_message: str) -> str:
        """
        分类错误信息
        
        Args:
            error_message: 错误信息
            
        Returns:
            错误类型
        """
        error_lower = error_message.lower()
        
        if "syntax error" in error_lower:
            return "Syntax Error"
        elif "no such table" in error_lower:
            return "Table Not Found"
        elif "no such column" in error_lower:
            return "Column Not Found"
        elif "ambiguous column" in error_lower:
            return "Ambiguous Column Name"
        elif "no such function" in error_lower:
            return "Function Not Found"
        elif "timeout" in error_lower:
            return "Timeout"
        elif "forbidden" in error_lower:
            return "Forbidden Operation"
        else:
            return "Unknown Error"
    
    def batch_execute(
        self,
        db_path: str,
        sqls: List[str],
        gold_sql: str
    ) -> List[ComparisonResult]:
        """
        批量执行SQL并生成反馈
        
        Args:
            db_path: 数据库路径
            sqls: SQL列表
            gold_sql: 标准答案SQL
            
        Returns:
            比对结果列表
        """
        results = []
        for sql in sqls:
            result = self.generate_feedback(sql, gold_sql, db_path)
            results.append(result)
        return results


def get_db_path(db_root: str, db_id: str) -> str:
    """
    获取数据库文件路径
    
    Args:
        db_root: 数据库根目录
        db_id: 数据库ID
        
    Returns:
        数据库文件完整路径
    """
    return os.path.join(db_root, db_id, f"{db_id}.sqlite")
