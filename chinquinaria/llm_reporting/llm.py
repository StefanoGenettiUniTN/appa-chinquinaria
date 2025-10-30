"""
LLM Reporting Module
"""

import openai
from transformers import pipeline

def generate_summary_open_source(prompt, model_name="mistralai/Mistral-7B-Instruct-v0.2"):
    summarizer = pipeline("text-generation", model=model_name)
    result = summarizer(prompt, max_new_tokens=300, do_sample=True)
    return result[0]["generated_text"]

def generate_summary_proprietary(prompt, model="gpt-4o-mini"):
    client = openai.OpenAI()
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content

def summarize_shap(shap_summary, llm_type="open_source"):
    prompt = (
        f"Explain the pollutant prediction results based on these SHAP values:\n"
        f"{shap_summary}\n"
        f"Write a concise analysis for an environmental scientist."
    )

    if llm_type == "open_source":
        return generate_summary_open_source(prompt)
    else:
        return generate_summary_proprietary(prompt)
    
def generate_final_essay(window_summaries, llm_type="open_source"):
    combined_text = "\n\n".join(window_summaries)
    final_prompt = (
        f"Here are multiple analyses of pollutant behavior across time windows:\n\n"
        f"{combined_text}\n\n"
        f"Please write a coherent essay summarizing key findings, "
        f"trends, and implications for air quality management."
    )
    return summarize_shap(final_prompt, llm_type=llm_type)