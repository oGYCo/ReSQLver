"""
提示词模板模块

功能：
1. 初始SQL生成提示词模板
2. 渐进修订提示词模板（用于错误SQL的修复）
"""

from typing import Dict, Optional


class PromptTemplates:
    """
    提示词模板管理器
    
    提供两种类型的提示词：
    1. 初始生成模板：用于从问题生成初始SQL
    2. 修订模板：用于根据错误反馈修改SQL
    """
    
    # 系统提示词
    SYSTEM_PROMPT = """You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>
...
</think>
<answer>
...
</answer>"""

    # 渐进修订模板（用于修复错误SQL）- 更详细版本
    REVISION_TEMPLATE = """You are a SQL debugging expert. Fix the incorrect SQL query based on the execution feedback.

## DATABASE SCHEMA
```sql
{schema}
```

## QUESTION
{evidence}{question}

## INCORRECT SQL
```sql
{previous_sql}
```

## EXECUTION FEEDBACK
{execution_feedback}

## DEBUGGING GUIDE

### For Execution Errors:
| Error Type | Common Cause | Fix |
|------------|--------------|-----|
| no such table | Typo in table name | Check schema for exact name |
| no such column | Typo or wrong table | Verify column exists in the table |
| ambiguous column | Column in multiple tables | Add table prefix: T1.column |
| syntax error | SQL syntax issue | Check keywords, quotes, parentheses |
| near "X" | Error at token X | Fix the SQL near that position |

### For Wrong Results:
| Issue | Likely Cause | Fix |
|-------|--------------|-----|
| Too many rows | Missing WHERE or JOIN condition | Add filters |
| Too few rows | WHERE too restrictive | Relax conditions |
| Extra columns | SELECT has unnecessary columns | Remove them |
| Missing columns | SELECT missing columns | Add required columns |
| Wrong values | Wrong JOIN or calculation | Check logic |

### SQLite-Specific Rules:
1. String literals: Use 'single quotes' (not double quotes)
2. No FULL/RIGHT JOIN: Use LEFT JOIN or restructure query
3. No TOP N: Use LIMIT N at the end of query
4. DISTINCT: Use when duplicates appear unexpectedly
5. GROUP BY: Required when mixing aggregate and non-aggregate columns
6. HAVING: Use for conditions on aggregate results (not WHERE)
7. Column aliases: Can't use alias in WHERE, use subquery instead

## OUTPUT
Provide the corrected SQL query:
```sql
SELECT ...
```
"""

    # 简化版修订模板（更短的提示词）
    REVISION_TEMPLATE_SIMPLE = """Fix this SQL query for SQLite.

Schema:
```sql
{schema}
```

Question: {question}
{evidence}

Incorrect SQL:
```sql
{previous_sql}
```

Feedback: {execution_feedback}

Key SQLite rules:
- Use exact table/column names from schema (case-sensitive)
- String values use single quotes: 'value'
- No FULL/RIGHT JOIN - use LEFT JOIN
- Add table prefix for ambiguous columns: table.column
- GROUP BY required for non-aggregated columns with aggregates

Corrected SQL:
```sql
SELECT ...
```
"""

    @classmethod
    def build_revision_prompt(
        cls,
        schema: str,
        question: str,
        previous_sql: str,
        execution_feedback: str,
        evidence: str = "",
        use_simple: bool = False
    ) -> str:
        """
        构建修订提示词
        
        Args:
            schema: 数据库schema信息
            question: 自然语言问题
            previous_sql: 之前错误的SQL
            execution_feedback: 执行反馈信息
            evidence: 辅助信息
            use_simple: 是否使用简化版模板
            
        Returns:
            完整的修订提示词
        """
        evidence_text = f"Context: {evidence}\n" if evidence else ""
        
        # 截断输入以防止Token溢出
        if len(schema) > 15000:
            schema = schema[:15000] + "\n... (truncated)"
        if len(previous_sql) > 2000:
            previous_sql = previous_sql[:2000] + "\n... (truncated)"
        if len(execution_feedback) > 5000:
            execution_feedback = execution_feedback[:5000] + "\n... (truncated)"
        
        template = cls.REVISION_TEMPLATE_SIMPLE if use_simple else cls.REVISION_TEMPLATE
        
        return template.format(
            schema=schema,
            question=question,
            evidence=evidence_text,
            previous_sql=previous_sql,
            execution_feedback=execution_feedback
        )
    
    @classmethod
    def build_chat_messages(
        cls,
        user_content: str,
        system_prompt: Optional[str] = None
    ) -> list:
        """
        构建聊天消息格式
        
        Args:
            user_content: 用户消息内容
            system_prompt: 系统提示词，None则使用默认值
            
        Returns:
            消息列表
        """
        system = system_prompt or cls.SYSTEM_PROMPT
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_content})
        
        return messages


class SchemaLoader:
    """
    数据库Schema加载器
    
    从dev_tables.json或数据库文件加载schema信息
    """
    
    @staticmethod
    def load_schema_from_tables_json(
        tables_data: Dict,
        db_id: str
    ) -> str:
        """
        从tables.json数据中加载指定数据库的schema
        
        Args:
            tables_data: tables.json的内容（列表格式）
            db_id: 数据库ID
            
        Returns:
            格式化的schema字符串
        """
        # 查找对应的数据库
        db_info = None
        for db in tables_data:
            if db.get("db_id") == db_id:
                db_info = db
                break
        
        if not db_info:
            return f"-- Schema for {db_id} not found"
        
        schema_parts = []
        
        # 获取表名列表
        table_names = db_info.get("table_names_original", [])
        column_names = db_info.get("column_names_original", [])
        column_types = db_info.get("column_types", [])
        primary_keys = db_info.get("primary_keys", [])
        foreign_keys = db_info.get("foreign_keys", [])
        
        # 构建每个表的CREATE TABLE语句
        for table_idx, table_name in enumerate(table_names):
            columns = []
            pk_columns = []
            
            # 收集该表的所有列
            for col_idx, (col_table_idx, col_name) in enumerate(column_names):
                if col_table_idx == table_idx:
                    col_type = column_types[col_idx] if col_idx < len(column_types) else "text"
                    columns.append(f"    {col_name} {col_type}")
                    
                    # 检查是否是主键
                    if col_idx in primary_keys:
                        pk_columns.append(col_name)
            
            if columns:
                # 添加主键约束
                if pk_columns:
                    columns.append(f"    PRIMARY KEY ({', '.join(pk_columns)})")
                
                create_stmt = f"CREATE TABLE {table_name} (\n"
                create_stmt += ",\n".join(columns)
                create_stmt += "\n);"
                
                schema_parts.append(create_stmt)
        
        # 添加外键信息作为注释
        if foreign_keys:
            fk_comments = ["\n-- Foreign Key Relationships:"]
            for fk in foreign_keys:
                if len(fk) >= 2:
                    from_col = column_names[fk[0]] if fk[0] < len(column_names) else ("?", "?")
                    to_col = column_names[fk[1]] if fk[1] < len(column_names) else ("?", "?")
                    
                    from_table = table_names[from_col[0]] if from_col[0] >= 0 and from_col[0] < len(table_names) else "?"
                    to_table = table_names[to_col[0]] if to_col[0] >= 0 and to_col[0] < len(table_names) else "?"
                    
                    fk_comments.append(
                        f"-- {from_table}.{from_col[1]} -> {to_table}.{to_col[1]}"
                    )
            schema_parts.append("\n".join(fk_comments))
        
        return "\n\n".join(schema_parts)
    
    @staticmethod
    def get_schema_from_input_seq(input_seq: str) -> str:
        """
        从input_seq中提取schema部分
        
        Args:
            input_seq: 完整的input_seq
            
        Returns:
            提取的schema字符串
        """
        # 查找 "Database Schema:" 和 "Question:" 之间的内容
        schema_start = input_seq.find("Database Schema:")
        schema_end = input_seq.find("Question:")
        
        if schema_start != -1 and schema_end != -1:
            schema = input_seq[schema_start + len("Database Schema:"):schema_end]
            return schema.strip()
        
        # 备选：查找 CREATE TABLE 语句
        if "CREATE TABLE" in input_seq:
            lines = input_seq.split("\n")
            schema_lines = []
            in_schema = False
            
            for line in lines:
                if "CREATE TABLE" in line:
                    in_schema = True
                if in_schema:
                    schema_lines.append(line)
                    if line.strip().endswith(");"):
                        schema_lines.append("")  # 添加空行分隔
            
            return "\n".join(schema_lines)
        
        return ""
