import sqlite3
import sys
import os
from func_timeout import func_timeout, FunctionTimedOut

class SQLExecutor:
    @staticmethod
    def execute_sql(predicted_sql, ground_truth, db_path, timeout=120.0):
        """
        Executes SQL and returns (is_correct, feedback)
        """
        import time
        
        # Helper to execute a single query
        def _exec_query(sql, path):
            conn = sqlite3.connect(path)
            cursor = conn.cursor()
            cursor.execute(sql)
            res = cursor.fetchall()
            headers = [description[0] for description in cursor.description] if cursor.description else []
            conn.close()
            return res, headers

        # 1. Execute Ground Truth
        print(f"[SQLExecutor] Executing Ground Truth SQL on {os.path.basename(db_path)}...", flush=True)
        start_time = time.time()
        try:
            # Use the default timeout (or a generous one) for Ground Truth to establish baseline
            ground_truth_res, ground_truth_headers = func_timeout(timeout, _exec_query, args=(ground_truth, db_path))
            gt_duration = time.time() - start_time
            print(f"[SQLExecutor] Ground Truth finished in {gt_duration:.2f}s.", flush=True)
        except FunctionTimedOut:
            return False, f"Error: Ground Truth SQL execution timed out ({timeout}s). Cannot evaluate."
        except Exception as e:
            return False, f"Error executing Ground Truth SQL: {e}"

        # 2. Calculate Dynamic Timeout
        # Strategy: At least 10s, otherwise 1.2x GT time + 5s buffer
        # This ensures fast queries have a floor, and slow queries have a proportional buffer.
        dynamic_timeout = max(10.0, gt_duration * 1.2 + 5.0)

        # 3. Execute Predicted SQL
        print(f"[SQLExecutor] Executing Predicted SQL (Timeout limit: {dynamic_timeout:.2f}s)...", flush=True)
        try:
            predicted_res, predicted_headers = func_timeout(dynamic_timeout, _exec_query, args=(predicted_sql, db_path))
            print(f"[SQLExecutor] Predicted SQL finished.", flush=True)
        except FunctionTimedOut:
            return False, f"Timeout: Query execution exceeded the limit of {dynamic_timeout:.2f}s (Ground Truth took {gt_duration:.2f}s)."
        except sqlite3.Error as e:
            error_msg = str(e)
            if "You can only execute one statement at a time" in error_msg:
                return False, "SQL Execution Error: Multiple SQL statements detected. Please combine them into a single SQL query to answer all parts of the question."
            return False, f"SQL Execution Error: {error_msg}"
        except Exception as e:
            return False, f"Error: {str(e)}"

        # 4. Compare results
        # Check if predicted SQL returned any columns (i.e., was it a SELECT statement?)
        if not predicted_headers and not predicted_res:
             # If ground truth also has no headers/results, maybe it's fine? 
             # But usually ground truth is a SELECT.
             if ground_truth_headers:
                 return False, "SQL Execution Error: The query did not return any results or columns (it might be empty or not a SELECT statement)."

        # Compare results
        if set(predicted_res) == set(ground_truth_res):
            return True, "Correct"
        else:
            feedback = ["Incorrect result."]
            
            # Column mismatch
            if len(predicted_headers) != len(ground_truth_headers):
                feedback.append(f"Column count mismatch: Expected {len(ground_truth_headers)} columns ({', '.join(ground_truth_headers)}), but got {len(predicted_headers)} columns ({', '.join(predicted_headers)}).")
            
            # Row count mismatch
            if len(predicted_res) != len(ground_truth_res):
                feedback.append(f"Row count mismatch: Expected {len(ground_truth_res)} rows, but got {len(predicted_res)} rows.")
            
            # Content mismatch details
            missing_rows = list(set(ground_truth_res) - set(predicted_res))
            extra_rows = list(set(predicted_res) - set(ground_truth_res))
            
            if missing_rows:
                feedback.append(f"Missing rows (example): {missing_rows[:3]}")
            if extra_rows:
                feedback.append(f"Unexpected rows (example): {extra_rows[:3]}")
            
            # Sample data
            feedback.append(f"Expected result sample: {ground_truth_res[:3]}")
            feedback.append(f"Actual result sample: {predicted_res[:3]}")
            
            return False, "\n".join(feedback)

    @staticmethod
    def get_db_path(db_root_path, db_id):
        return os.path.join(db_root_path, db_id, f"{db_id}.sqlite")

    @staticmethod
    def get_schema(db_path):
        """
        Dump schema from SQLite database
        """
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
            rows = cursor.fetchall()
            schema = "\n".join([row[0] for row in rows if row[0] is not None])
            return schema
        except Exception as e:
            return f"Error loading schema: {e}"
