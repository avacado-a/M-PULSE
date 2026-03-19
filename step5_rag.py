import ollama

def rag_explanation(predicted_spike, keywords):
    print(f"Predicted Trend Shift: {predicted_spike}%")
    
    if predicted_spike > 15:
        print("Spike detected! Triggering Local RAG Explainability Node...")
        prompt = (f"The system predicts a spike. Here are the top keywords from today: "
                  f"[{', '.join(keywords)}]. Write a two-sentence explanation of what is happening.")
        
        print(f"Sending prompt to Ollama (phi3:mini):\n{prompt}\n")
        try:
            response = ollama.chat(model='phi3:mini', messages=[
                {'role': 'user', 'content': prompt}
            ])
            explanation = response['message']['content']
            print("\n--- Ollama Output ---")
            print(explanation)
            print("---------------------\n")
            print("The pipeline is finished.")
            
        except Exception as e:
            print(f"Error querying Ollama: {e}")
            print("Ensure Ollama is installed locally and the 'phi3:mini' model is pulled ('ollama run phi3:mini').")
    else:
        print("No significant spike detected. Routine monitoring continues.")

if __name__ == "__main__":
    predicted_spike = 25
    keywords = ["WPILib", "update", "error", "CAN bus"]
    rag_explanation(predicted_spike, keywords)
