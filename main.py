import os
import sys
import logging
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import pandas as pd

from mpulse.data.ingestion import DataIngestor
from mpulse.models.embeddings import SemanticEncoder
from mpulse.training.trainer import ModelTrainer
from mpulse.data.dataset import extract_real_data, create_dataloaders

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

EXPERIMENTAL_TOPICS = [
    {"name": "FIRST Robotics Competition", "db_topic_mac": "robotics", "db_topic_mic": "robotics", "category": "Agreeable"},
    {"name": "NVIDIA Blackwell", "db_topic_mac": "NVIDIA Blackwell", "db_topic_mic": "NVIDIA Blackwell", "category": "Mainstream"},
    {"name": "The Middle East", "db_topic_mac": "Middle East", "db_topic_mic": "Middle East", "category": "Politically Split"}
]

def generate_thesis_figures(all_results, save_dir="results"):
    """
    Generates academic figures driven strictly by real PyTorch evaluation metrics
    and raw database extraction.
    """
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # ---------------------------------------------------------
    # Figure 1: MSE Ranking across Topic Categories
    # ---------------------------------------------------------
    categories = []
    dual_mses = []
    
    for res in all_results:
        categories.append(res['category'])
        dual_mses.append(res['metrics']['Dual-Stream']['mse'])
        
    plt.figure(figsize=(10, 6))
    sns.barplot(x=categories, y=dual_mses, palette="viridis")
    plt.title("Figure 1: Mean Squared Error (MSE) by Topic Category")
    plt.ylabel("Test Set MSE (Lower is Better)")
    plt.xlabel("Topic Structure")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Figure_1_MSE_Comparison.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # Figure 2: Architecture Comparison (Ablation Study)
    # ---------------------------------------------------------
    plot_data = {"Category": [], "Architecture": [], "MSE": []}
    for res in all_results:
        cat = res['category']
        for arch in ["Macro-Only", "Micro-Only", "Dual-Stream"]:
            plot_data["Category"].append(cat)
            plot_data["Architecture"].append(arch)
            plot_data["MSE"].append(res['metrics'][arch]['mse'])
            
    df_ablation = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(12, 7))
    sns.barplot(x="Category", y="MSE", hue="Architecture", data=df_ablation, palette="muted")
    plt.title("Figure 2: Ablation Study - Prediction Accuracy Across Resolution Streams")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.xlabel("Topic Category")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Figure_2_Ablation_Study.png"), dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # Figure 3: Sentiment Lag Analysis (Empirical Database Evaluation)
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    target_data = None
    for res in all_results:
        if res['category'] == "Mainstream":
            target_data = res
            break
            
    if target_data is not None:
        volumes = target_data['actual_volumes']
        sentiments = target_data['actual_sentiments']
        
        plt.plot(range(len(volumes)), volumes, label='Actual Normalized Volume', color='black', linewidth=2)
        plt.plot(range(len(sentiments)), sentiments, label='Lexicon Sentiment Score', color='orange', linestyle='--', linewidth=2)
        plt.title(f"Figure 3: Temporal Lag of Sentiment ({target_data['topic']})")
        plt.ylabel("Relative Intensity / Score")
        plt.xlabel("Timeline (Days)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "Figure_3_Sentiment_Lag.png"), dpi=300)
    plt.close()

    logger.info(f"All experimental figures successfully generated in '{save_dir}/'.")

def run_ingestion_phase():
    """Phase 1: Deep data collection via API scraping."""
    logger.info("=== Starting Data Ingestion Phase ===")
    ingestor = DataIngestor()
    for topic_info in EXPERIMENTAL_TOPICS:
        topic = topic_info["name"]
        logger.info(f"Fetching data for: {topic}")
        # Note: Ingestion uses the exact 'name' string to build the dataset
        ingestor.fetch_macro_gdelt(topic)
        ingestor.fetch_micro_bluesky(topic)
    logger.info("=== Data Ingestion Phase Complete ===")

def run_experiment_phase(window_size):
    """Phase 2: Local model training and evaluation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Initialized M-PULSE framework on device: {device}")
    
    trainer = ModelTrainer(device)
    encoder = SemanticEncoder()
    all_experimental_results = []

    for topic_info in EXPERIMENTAL_TOPICS:
        topic = topic_info["name"]
        cat = topic_info["category"]
        db_mac = topic_info["db_topic_mac"]
        db_mic = topic_info["db_topic_mic"]
        
        logger.info(f"=== Starting Experimental Protocol for: {topic} ({cat}) ===")
        
        # 1. Semantic Embedding (using db strings if needed, but topic is preferred)
        if not encoder.generate_embeddings(topic):
            logger.warning(f"Skipping {topic} due to embedding failure (insufficient corpus).")
            continue
            
        # 2. Data Extraction
        try:
            X_mac_arr, X_mic_arr, Y_arr, actual_volumes, actual_sentiments = extract_real_data(
                topic, window_size=window_size
            )
        except Exception as e:
            logger.error(f"Failed to load data for {topic}: {e}")
            continue
            
        metrics = {}
        
        # 3. Ablation Runs
        p_mac, m_mac = trainer.train_evaluate(X_mac_arr, X_mic_arr, Y_arr, f"{topic}_Macro", True, False)
        metrics["Macro-Only"] = {"preds": p_mac, "mse": m_mac}
        
        p_mic, m_mic = trainer.train_evaluate(X_mac_arr, X_mic_arr, Y_arr, f"{topic}_Micro", False, True)
        metrics["Micro-Only"] = {"preds": p_mic, "mse": m_mic}
        
        p_dual, m_dual = trainer.train_evaluate(X_mac_arr, X_mic_arr, Y_arr, f"{topic}_Dual", True, True)
        metrics["Dual-Stream"] = {"preds": p_dual, "mse": m_dual}
        
        all_experimental_results.append({
            "topic": topic,
            "category": cat,
            "actual_volumes": actual_volumes,
            "actual_sentiments": actual_sentiments,
            "metrics": metrics
        })
        
    # 4. Result Generation
    if all_experimental_results:
        logger.info("Generating aggregated research figures from real evaluation data...")
        generate_thesis_figures(all_experimental_results)
    else:
        logger.error("No valid topics were processed. Ensure database contains sufficient aligned data.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M-PULSE CLI Orchestrator")
    parser.add_argument("--mode", type=str, choices=['ingest', 'experiment', 'all'], default='experiment',
                        help="Select pipeline phase: 'ingest' (pull data), 'experiment' (train models), or 'all'")
    parser.add_argument("--window", type=int, default=3, help="Sequence window length for the experiment")
    args = parser.parse_args()

    if args.mode in ['ingest', 'all']:
        run_ingestion_phase()
        
    if args.mode in ['experiment', 'all']:
        run_experiment_phase(args.window)
