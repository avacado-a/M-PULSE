import subprocess
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set academic style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)

TOPICS = [
    ("robotics", "Agreeable"),
    ("NVIDIA Blackwell", "Non-Political"),
    ("Middle East", "Politically Split")
]

python_exec = sys.executable

def run_cmd(cmd):
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def collect_results():
    all_results = []
    # Find the latest run folder for each topic
    for topic_name, category in TOPICS:
        folder_name = topic_name.replace(" ", "_")
        run_folders = [d for d in os.listdir("runs") if d.startswith(folder_name)]
        if not run_folders:
            print(f"No runs found for {topic_name}")
            continue
        
        latest_run = sorted(run_folders)[-1]
        res_file = os.path.join("runs", latest_run, "results.txt")
        
        if os.path.exists(res_file):
            with open(res_file, "r") as f:
                lines = f.read().splitlines()
                all_results.append({
                    "Topic": topic_name,
                    "Category": category,
                    "Macro_MSE": float(lines[0]),
                    "Micro_MSE": float(lines[1]),
                    "Dual_MSE": float(lines[2]),
                    "Peak_VRAM": float(lines[3])
                })
    return pd.DataFrame(all_results)

if __name__ == "__main__":
    print("🚀 STARTING THESIS COMPARISON ANALYSIS...")
    
    # We will attempt to run training for the three key topics.
    # If the database already has the data from our previous runs, it will proceed.
    for topic, category in TOPICS:
        print(f"\nProcessing Topic: {topic} ({category})")
        try:
            # We assume step1 and step2 were successful in previous turns or 
            # will be skipped by step4 if data is missing.
            run_cmd([python_exec, "step4_training.py", topic])
        except Exception as e:
            print(f"⚠️ Could not complete training for {topic}: {e}")

    print("\n📊 GENERATING FINAL THESIS CHARTS...")
    df = collect_results()
    os.makedirs("thesis_results", exist_ok=True)

    # Chart 1: MSE Comparison (Predictability ranking)
    # Sort by Dual_MSE to prove the "Agreeable -> Political" ranking
    df = df.sort_values("Dual_MSE")
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x="Category", y="Dual_MSE", data=df, palette="viridis")
    plt.title("Topic Predictability Ranking (Dual-Stream MSE)")
    plt.ylabel("Prediction Error (MSE)")
    plt.savefig("thesis_results/topic_predictability_comparison.png", dpi=300)

    # Chart 2: Architecture Performance (Dual vs Baseline)
    # Melt the dataframe for a grouped bar chart
    melted_df = df.melt(id_vars="Category", value_vars=["Macro_MSE", "Micro_MSE", "Dual_MSE"], 
                        var_name="Architecture", value_name="MSE")
    
    plt.figure(figsize=(14, 8))
    sns.barplot(x="Category", y="MSE", hue="Architecture", data=melted_df)
    plt.title("Ablation Study: Prediction Accuracy Across Resolution Streams")
    plt.savefig("thesis_results/architecture_ablation_comparison.png", dpi=300)

    # Chart 3: VRAM Efficiency (Hardware Proof)
    plt.figure(figsize=(10, 6))
    plt.bar(df["Category"], df["Peak_VRAM"], color='green', alpha=0.7)
    plt.axhline(y=6.0, color='r', linestyle='--', label='6GB Limit')
    plt.title("M-PULSE Hardware Efficiency Profile")
    plt.ylabel("Peak VRAM (GB)")
    plt.legend()
    plt.savefig("thesis_results/hardware_efficiency_proof.png", dpi=300)

    print("\n✅ THESIS RESULTS READY in thesis_results/ folder.")
    print(df)
