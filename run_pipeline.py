import subprocess
import sqlite3
import os
import sys

def run_step(step_name, command, success_marker=None, check_func=None):
    print(f"\n{'='*60}")
    print(f"🚀 RUNNING: {step_name}")
    print(f"💻 COMMAND: {' '.join(command)}")
    print(f"{'='*60}\n")
    
    # We use Popen so we can stream the output live to the terminal
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    output_log = ""
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
        sys.stdout.flush() # Force immediate print to terminal
        output_log += line
        
    process.stdout.close()
    return_code = process.wait()
    
    # Check 1: Exit Code
    if return_code != 0:
        print(f"\n❌ HARD ERROR: {step_name} failed with exit code {return_code}")
        sys.exit(1)
        
    # Check 2: Expected Output Marker (Catches silent failures like caught exceptions)
    if success_marker and success_marker not in output_log:
        print(f"\n❌ SILENT ERROR: {step_name} did not produce expected output: '{success_marker}'")
        sys.exit(1)
        
    # Check 3: Data / State Verification Function
    if check_func:
        success, msg = check_func()
        if not success:
            print(f"\n❌ VERIFICATION FAILED: {msg}")
            sys.exit(1)
            
    print(f"\n✅ {step_name} COMPLETED SUCCESSFULLY.")

def check_step1():
    db_path = 'm_pulse.db'
    if not os.path.exists(db_path):
        return False, "Database m_pulse.db was not created."
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT count(*) FROM macro_data")
        macro_c = c.fetchone()[0]
        c.execute("SELECT count(*) FROM micro_data")
        micro_c = c.fetchone()[0]
        conn.close()
        
        if macro_c == 0 and micro_c == 0:
            return False, "Both macro_data and micro_data tables are entirely empty."
            
        print(f"  -> DB Verification Passed: {macro_c} macro records, {micro_c} micro records ingested.")
        return True, ""
    except Exception as e:
        return False, f"Database integrity check failed: {e}"

def check_step2():
    if not os.path.exists("current_context.model"):
        return False, "current_context.model was not saved to disk."
    return True, ""

def check_step4():
    # In the final version, we check for the most recent run folder or the root artifacts
    if not os.path.exists("final_forecast_root.png") and not os.path.exists("runs"):
        return False, "Training outputs were not detected."
    return True, ""

if __name__ == "__main__":
    topic = "Middle East"
    python_exec = sys.executable 
    
    # Step 1
    run_step(
        "Step 1: Data Ingestion Pipeline", 
        [python_exec, "-u", "step1_ingestion.py", topic],
        check_func=check_step1
    )
    
    # Step 2
    run_step(
        "Step 2: Semantic Filtering & Embeddings", 
        [python_exec, "-u", "step2_embeddings.py", topic],
        check_func=check_step2
    )

    # Step 3
    run_step(
        "Step 3: PyTorch Network Architecture Check", 
        [python_exec, "-u", "step3_model.py"],
        success_marker="Architecture structurally sound"
    )

    # Step 4
    run_step(
        "Step 4: Training & Evaluation Loop", 
        [python_exec, "-u", "step4_training.py"],
        success_marker="FINAL RUN COMPLETE",
        check_func=check_step4
    )

    # Step 5
    run_step(
        "Step 5: Local RAG Explainability Node", 
        [python_exec, "-u", "step5_rag.py"],
        success_marker="The pipeline is finished."
    )
    
    print("\n" + "🌟"*25)
    print("🌟 FULL M-PULSE PIPELINE COMPLETED SUCCESSFULLY! 🌟")
    print("🌟"*25 + "\n")
