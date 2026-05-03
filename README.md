# M-PULSE: Predicting Media Trends Through Dual-Stream NLP Analysis

M-PULSE is a specialized research framework designed to quantify the predictability of social and conventional media trends across different sociopolitical categories. 

## 🧬 Architecture

This repository utilizes a custom PyTorch architecture designed for temporal multi-resolution forecasting. 

- **Macro-Stream**: Processes institutional baselines via standard LSTM layers.
- **Micro-Stream**: Processes volatile social chatter using 1D-CNN layers for localized feature extraction, fed into an LSTM.
- **Fusion Protocol**: Concatenates latent states for final regressive output.

## 🛠️ Repository Structure
```
M-PULSE/
├── mpulse/                  # Core Python Package
│   ├── data/                # GDELT/Bluesky Ingestion, Lexicon Sentiment & PyTorch Datasets
│   ├── models/              # MPulseNet Architecture & DBSCAN/Word2Vec Semantic Encoder
│   └── training/            # Evaluation loops
├── main.py                  # CLI Orchestrator
```

## ⚙️ Installation & Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your Bluesky credentials as environment variables:
   ```powershell
   $env:BSKY_HANDLE="yourname.bsky.social"
   $env:BSKY_APP_PASSWORD="xxxx-xxxx-xxxx-xxxx"
   ```

## 🚀 Usage Workflow

M-PULSE strictly separates the slow, API-rate-limited data collection phase from the fast, iterative PyTorch experimentation phase.

### Phase 1: Data Ingestion (Run Once)
Collect months of historical news (GDELT) and social media chatter (Bluesky AT Protocol) for the defined research topics. This builds the local `m_pulse.db` SQLite database.

Because GDELT ingestion queries years of data and respects API rate limits (which can take hours), it is highly recommended to pull all the data you need once and then reuse it. You can run the collection script in the background:

```powershell
python main.py --mode ingest
```
*Note: For Windows PowerShell, you can use `Start-Process python -ArgumentList "main.py --mode ingest" -NoNewWindow` to run it in the background.*

### Phase 2: Experimentation (Run Iteratively)
Once the database is populated, you can train the models and generate the ablation study charts in minutes without re-triggering the slow API calls. This phase mathematically evaluates the corpus via DBSCAN, generates localized Word2Vec embeddings, and trains the LSTM network.

```powershell
python main.py --mode experiment
```


To run both phases sequentially:
```powershell
python main.py --mode all
```

## 📊 Evaluation
Running the experiment phase automatically generates research figures in the `results/` directory, providing comparative Mean Squared Error (MSE) metrics across Macro-Only, Micro-Only, and Dual-Stream configurations, as well as Temporal Lag Analysis.
