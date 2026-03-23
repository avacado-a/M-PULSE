import ollama
import sqlite3
import pandas as pd
from collections import Counter
import re

def get_top_keywords(db_path="m_pulse.db", limit=5):
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT clean_text FROM micro_data ORDER BY created_utc DESC LIMIT 100", conn)
        conn.close()
        text = " ".join(df['clean_text'].dropna().tolist()).lower()
        words = re.findall(r'\b[a-z]{4,}\b', text)
        stopwords = {'that', 'this', 'with', 'from', 'have', 'they', 'your', 'what', 'about', 'https', 'http', 'com'}
        words = [w for w in words if w not in stopwords]
        common = Counter(words).most_common(limit)
        return [w[0] for w in common] if common else ["peace", "conflict", "negotiations"]
    except Exception as e:
        print(f"Failed to fetch real keywords: {e}")
        return ["peace", "conflict", "summit", "negotiations", "trade"]

def rag_explanation(predicted_spike, keywords):
    print(f"Predicted Trend Shift: {predicted_spike}%")
    
    if predicted_spike > 15:
        print("Spike detected! Triggering Local RAG Explainability Node...")
        prompt = (f"The system predicts a spike. Here are the top keywords from today: "
                  f"[{', '.join(keywords)}]. Write a two-sentence explanation of what is happening.")
        
        print(f"Sending prompt to Ollama (gemma3:1b):\n{prompt}\n")
        try:
            response = ollama.chat(model='gemma3:1b', messages=[
                {'role': 'user', 'content': prompt}
            ])
            explanation = response['message']['content']
            print("\n--- Ollama Output ---")
            print(explanation)
            print("---------------------\n")
            print("The pipeline is finished.")
            
        except Exception as e:
            print(f"Error querying Ollama: {e}")
            print("Ensure Ollama is installed locally and the 'gemma3:1b' model is pulled ('ollama run gemma3:1b').")
    else:
        print("No significant spike detected. Routine monitoring continues.")

if __name__ == "__main__":
    predicted_spike = 25
    keywords = get_top_keywords()
    rag_explanation(predicted_spike, keywords)
