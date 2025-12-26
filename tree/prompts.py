class Prompts:
    SYSTEM_PROMPT = """You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>
...
</think>
<answer>
...
</answer>"""

    # Initial generation prompt (usually comes from the dataset input_seq, but we might need a template if not provided)
    # The user's README says "node_initial" has "input_seq" which comes from train_bird.json.
    # So we might just use that.

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

## INSTRUCTIONS
1. Analyze the execution feedback and the incorrect SQL.
2. Think step-by-step about how to fix the error.
3. Output the corrected SQL query in a markdown code block.

## OUTPUT
Respond in the following format:
<think>
... reasoning ...
</think>
<answer>
```sql
... corrected SQL ...
```
</answer>
"""
