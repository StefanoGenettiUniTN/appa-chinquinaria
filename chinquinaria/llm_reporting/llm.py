"""
LLM Reporting Module
"""

import openai
from transformers import pipeline
from chinquinaria.config import CONFIG
from .prompts import build_shap_prompt, build_final_summary_prompt

def generate_summary_open_source(prompt, model_name="mistralai/Mistral-7B-Instruct-v0.2"):
    summarizer = pipeline("text-generation", model=model_name, device_map="auto")
    result = summarizer(prompt, do_sample=True, max_new_tokens=512)
    return result[0]["generated_text"]

def generate_summary_proprietary(prompt, model="gpt-4.1"):
    client = openai.OpenAI(base_url=CONFIG["endpoint"], api_key=CONFIG["token"])
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content

def summarize_shap(shap_summary: str, shap_raw_data: str | None = None) -> str:
    prompt_variant = CONFIG.get("llm_prompt_variant_shap", "v1")
    prompt = build_shap_prompt(
        variant=prompt_variant,
        shap_summary_text=shap_summary,
        shap_raw_data=shap_raw_data
    )

    if CONFIG["debug"]:
        print("LLM Prompt:")
        print(prompt)

    if CONFIG["llm_type"] == "open_source":
        return generate_summary_open_source(prompt)
    elif CONFIG["llm_type"] == "proprietary":
        return generate_summary_proprietary(prompt)
    elif CONFIG["llm_type"] == "fake":
        return shap_summary
    else:
        raise ValueError(f"Unknown LLM type: {CONFIG['llm_type']}")

def generate_final_essay(window_summaries, shap_data: str | None = None) -> str:
    prompt_variant = CONFIG.get("llm_prompt_variant_final", "v1")
    final_prompt = build_final_summary_prompt(
        variant=prompt_variant,
        window_summaries=window_summaries,
        shap_corpus=shap_data
    )
    
    if CONFIG["debug"]:
        print("LLM Prompt:")
        print(final_prompt)

    if CONFIG["llm_type"] == "open_source":
        return generate_summary_open_source(final_prompt)
    elif CONFIG["llm_type"] == "proprietary":
        return generate_summary_proprietary(final_prompt)
    elif CONFIG["llm_type"] == "fake":
        return final_prompt
    else:
        raise ValueError(f"Unknown LLM type: {CONFIG['llm_type']}")