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
        def _exec(p_sql, g_sql, path):
            conn = sqlite3.connect(path)
            cursor = conn.cursor()
            
            # Execute Ground Truth first to be sure it works (or to get expected result)
            cursor.execute(g_sql)
            ground_truth_res = cursor.fetchall()
            ground_truth_headers = [description[0] for description in cursor.description] if cursor.description else []
            
            # Execute Predicted SQL
            cursor.execute(p_sql)
            predicted_res = cursor.fetchall()
            predicted_headers = [description[0] for description in cursor.description] if cursor.description else []
            
            return predicted_res, ground_truth_res, predicted_headers, ground_truth_headers

        try:
            print(f"[SQLExecutor] Running query on {os.path.basename(db_path)}...", flush=True)
            predicted_res, ground_truth_res, predicted_headers, ground_truth_headers = func_timeout(timeout, _exec, args=(predicted_sql, ground_truth, db_path))
            print(f"[SQLExecutor] Query finished.", flush=True)
            
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
                
        except FunctionTimedOut:
            return False, "Timeout"
        except sqlite3.Error as e:
            error_msg = str(e)
            if "You can only execute one statement at a time" in error_msg:
                return False, "SQL Execution Error: Multiple SQL statements detected. Please combine them into a single SQL query to answer all parts of the question."
            return False, f"SQL Execution Error: {error_msg}"
        except Exception as e:
            return False, f"Error: {str(e)}"

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
