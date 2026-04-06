import subprocess
import os
import sys

"""
M-PULSE MASTER ORCHESTRATOR
Integrated Research Pipeline v3.0

This script executes the full end-to-end multi-resolution forecasting cycle:
1. Data Ingestion (GDELT + Bluesky)
2. Semantic Embedding (Word2Vec)
3. Architecture Validation
4. Ablation Study & Regression
5. Explainability Node (RAG)
"""

def run_step(name, cmd):
    print(f"\n{'='*70}")
    print(f"🚀 EXECUTING: {name}")
    print(f"{'='*70}")
    # Using -u for unbuffered real-time research logging
    result = subprocess.run([sys.executable, "-u"] + cmd, check=False)
    if result.returncode != 0:
        print(f"\n❌ PIPELINE STALLED: {name} failed.")
        sys.exit(1)

if __name__ == "__main__":
    # Target topic for the final thesis run
    topic = "NVIDIA Blackwell"
    
    print(f"🧪 STARTING M-PULSE EXPERIMENTAL RUN: '{topic}'")
    
    # Pathway 1: Data Acquisition
    run_step("Step 1: Multi-Resolution Ingestion", ["step1_ingestion.py", topic, "2024"])
    
    # Pathway 2: Semantic Analysis
    run_step("Step 2: Ephemeral Vector Mapping", ["step2_embeddings.py", topic])
    
    # Pathway 3: Structural Integrity
    run_step("Step 3: Network Architecture Check", ["step3_model.py"])
    
    # Pathway 4: Empirical Forecasting & Ablation
    run_step("Step 4: Quantitative Ablation Study", ["step4_training.py", topic])
    
    # Pathway 5: Localized Explainability
    run_step("Step 5: RAG Explanation Node", ["step5_rag.py"])
    
    print("\n" + "🌟"*30)
    print("🌟 FULL M-PULSE RESEARCH CYCLE COMPLETED SUCCESSFULLY! 🌟")
    print("🌟"*30 + "\n")
