# M-PULSE: Multi-Parametric Ultra-Lightweight Sentiment Engine

M-PULSE is a specialized research framework designed to quantify the predictability of social and conventional media trends across different sociopolitical categories. 

## 🧬 Research Thesis
The predictability of trends in media (both conventional and social) reveals that forecasting accuracy is most impacted by **Topic Structure**, followed by **Media Bias**, and then **Sentiment**. 

This repository contains the dual-stream neural network architecture and ingestion pipelines used to mathematically prove that localized, multi-resolution data fusion significantly outperforms isolated sentiment analysis.

## 🚀 Key Features
- **Dual-Stream Architecture**: Fuses a Macro (Institutional News via GDELT) pathway with a Micro (Ephemeral Social via Bluesky) pathway.
- **Multi-Resolution Temporal Alignment**: Aligns long-term news baselines with short-term social volatility.
- **Localized Encoding**: Utilizes task-specific Word2Vec embeddings to preserve technical and platform-specific context.
- **Hardware Optimized**: Scientifically verified to run on consumer-grade hardware (3GB VRAM).

## 📊 Results & Visualization
Final research charts and ablation studies are available in the `/thesis_results` folder:
- **Trend #1**: Predictability Ranking (Agreeable vs. Polarized topics).
- **Trend #2**: Social media as a Leading Indicator vs. Narrative Divergence.
- **Trend #3**: Temporal Lag Analysis of Sentiment crystallization.

## 🛠️ Installation & Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your Bluesky credentials as environment variables:
   ```bash
   set BSKY_HANDLE=yourname.bsky.social
   set BSKY_APP_PASSWORD=xxxx-xxxx-xxxx-xxxx
   ```

## 🧪 Usage
To reproduce the full research cycle:
```bash
python run_pipeline.py
```

## 📜 Citation
If utilizing this framework for research, please reference the M-PULSE Multi-Resolution Fusion methodology (2026).
