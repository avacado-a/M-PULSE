# M-PULSE: Micro-Parametric Understanding of Localized Social Events)

M-PULSE is a specialized research framework designed to predict of social and conventional media trends. 

## Research Paper
[Research Paper](https://docs.google.com/document/d/1wNWhK99xXIVXwCNQsg47dklm_DooHbIeQlJDC_OV_oU/edit?tab=t.0)

## Key Features
- **Dual-Stream Architecture**: Fuses a Macro (Institutional News via GDELT) pathway with a Micro (Ephemeral Social via Bluesky) pathway.
- **Localized Encoding**: Utilizes task-specific Word2Vec embeddings to preserve technical and platform-specific context.
- **Hardware Optimized**: Runs on consumer-grade hardware (6GB VRAM).

## Results & Visualization
Final research charts and ablation studies are available in the `/thesis_results` folder

## Installation & Setup
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

## Usage
To reproduce the full research cycle:
```bash
python run_pipeline.py
```

## 📜 Citation
If utilizing this framework for research, please cite me!
