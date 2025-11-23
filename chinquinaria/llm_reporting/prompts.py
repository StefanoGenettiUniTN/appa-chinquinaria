"""
Prompt templates for LLM reporting.
Each prompt family exposes multiple variants to make switching easy.
"""

from typing import List, Optional


# ---------- SHAP window-level report prompts ----------
def build_shap_prompt(
    variant: str,
    shap_summary_text: str,
    shap_raw_data: Optional[str] = None,
) -> str:
    """
    Build a prompt to explain SHAP results for a single window.
    """
    if variant == "v1":
        # Baseline, original prompt.
        base = (
            "Explain the pollutant prediction results based on these SHAP values:\n"
            f"{shap_summary_text}\n\n"
            "Write a concise analysis for an environmental scientist who has to understand "
            "the features which influence the behaviour of pollutant levels."
        )
        if shap_raw_data:
            base += (
                "\n\nAdditional raw SHAP details (use only as evidence, do not repeat verbatim):\n"
                f"{shap_raw_data}"
            )
        return base

    if variant == "v2":
        # Structured, practitioner-oriented output.
        body = [
            "You are assisting an environmental scientist. Using the SHAP information below, produce a clear, well-structured analysis.",
            "",
            "Requirements:",
            "- Start with a 2-3 sentence executive summary highlighting the top drivers.",
            "- Explain directionality (increase/decrease) of key features on pollutant levels.",
            "- Quantify relative importance when possible (e.g., rank top 5 drivers).",
            "- Note any time/seasonal effects or interactions if evident.",
            "- End with 2-3 actionable insights or monitoring suggestions.",
            "",
            "Style and Constraints:",
            "- Avoid mentioning SHAP explicitly; refer instead to 'feature importance' or 'model insights'.",
            "- Be precise and avoid speculation beyond what SHAP supports.",
            "- Prefer bullet points and short paragraphs.",
            "- Keep under 300 words.",
            "",
            "SHAP summary:",
            f"{shap_summary_text}",
        ]
        if shap_raw_data:
            body += [
                "",
                "Raw SHAP data (for grounding claims; do not copy verbatim):",
                f"{shap_raw_data}",
            ]
        return "\n".join(body)

    if variant == "v3":
        # This variant is a more analytic narrative, emphasis on cause-effect interpretability.
        body = [
            "Based on the SHAP analysis provided, write an analytic narrative for domain experts.",
            "",
            "Deliverables:",
            "1) Key Drivers: Identify and rank the most influential features and describe how they affect the target.",
            "2) Mechanisms: Explain plausible mechanisms consistent with SHAP directionality (avoid unsupported claims).",
            "3) Stability: Discuss whether effects seem consistent across the window or likely context-dependent.",
            "4) Implications: Suggest implications for forecasting, monitoring, or mitigation.",
            "",
            "Style: professional, evidence-based, concise. Avoid mentioning SHAP explicitly; refer instead to 'feature importance' or 'model insights'.",
            "",
            "SHAP summary:",
            f"{shap_summary_text}",
        ]
        if shap_raw_data:
            body += [
                "",
                "Raw SHAP evidence (use to ground the analysis):",
                f"{shap_raw_data}",
            ]
        return "\n".join(body)
    
    if variant == "v4":
        # Feature description but optimized
        body = [
        "You are assisting an environmental scientist studying PM10 in the province of Trento (TN). "
        "Using the model insights below, produce a clear, well-structured analysis.",
        "",
        "Requirements:",
        "- Start with a 2-3 sentence executive summary highlighting top drivers, not use bold text",
        "- Explain the direction (increase/decrease) of key features on PM10",
        "- Identify and rank the most influential features and describe their effects",
        "- Note time/seasonal effects or interactions only if evident",
        "- Do not include numeric feature importance values",
        "- Refer to 'feature importance' or 'model insights'; never mention SHAP",
        "- When mentioning places, specify the province in parentheses, note that Bologna is not in the dataset",
        "",
        "Style & Constraints:",
        "- Use bullet points and short paragraphs",
        "- Be precise, avoid speculation beyond provided data",
        "- Keep under 200 words",
        "",
        "Feature information:",
        "- Meteorology features:",
        "  - Humidity at 550 hPa, 950 hPa",
        "  - Air temperature at 550 hPa, 850 hPa, 950 hPa",
        "  - Zonal wind (U) at 550 hPa, 850 hPa, 950 hPa",
        "  - Meridional wind (V) at 550 hPa, 850 hPa, 950 hPa",
        "  - Precipitation (mm), surface air temperature (°C), relative humidity (%), wind speed (m/s), wind direction (°), atmospheric pressure (hPa), total solar radiation (kJ/m²), boundary-layer height",
        "- Representative PM10 levels for each province:",
        "  - BG: PM10 Calusco D'Adda",
        "  - BL: PM10 Parco Città di Bologna, not related with Bologna city/province",
        "  - BS: PM10 Palazzo del Broletto, Brescia",
        "  - CR: PM10 Piazza Cadorna, Cremona",
        "  - FE: PM10 Corso Isonzo, Ferrara",
        "  - LC: PM10 Valmadrera",
        "  - MN: PM10 Ponti sul Mincio",
        "  - MO: PM10 Via Ramesina, Modena",
        "  - PD: PM10 Granze",
        "  - PR: PM10 Via Saragat, Parma",
        "  - RE: PM10 San Rocco, Reggio Emilia",
        "  - RO: PM10 Largo Martiri, Rovigo",
        "  - TV: PM10 Conegliano",
        "  - VE: PM10 Sacca Fisola, Venice",
        "  - VI: PM10 Quartiere Italia, Vicenza",
        "  - VR: PM10 Borgo Milano, Verona",
        "",
        "Model insights data:",
        f"{shap_summary_text}"
    ]

        if shap_raw_data:
            body += [
                "",
                "Raw model data (for grounding claims; do not copy verbatim):",
                f"{shap_raw_data}"
            ]

        return "\n".join(body)

    # Default fallback to baseline if unknown
    return build_shap_prompt("v1", shap_summary_text, shap_raw_data)


# ---------- Final multi-window report prompts ----------
def build_final_summary_prompt(
    variant: str,
    window_summaries: List[str],
    shap_corpus: Optional[str] = None,
) -> str:
    """
    Build a prompt to synthesize multiple window-level analyses into a coherent final report.
    Optionally include a SHAP corpus (e.g., concatenated text summaries or raw stats) for grounding.
    """
    combined_text = "\n\n---\n\n".join(window_summaries)

    if variant == "v1":
        base = (
            "Here are multiple analyses of pollutant behavior across time windows:\n\n"
            f"{combined_text}\n\n"
            "Please write a coherent essay summarizing key findings, trends, and implications "
            "for air quality management."
        )
        if shap_corpus:
            base += (
                "\n\nUse the following SHAP evidence to ground your synthesis (do not repeat verbatim):\n"
                f"{shap_corpus}"
            )
        return base

    if variant == "v2":
        body = [
            "Synthesize the following window-level analyses into a single report.",
            "",
            "Structure:",
            "- Executive Summary (3-5 bullet points)",
            "- Consistent Patterns Across Windows",
            "- Divergences/Anomalies and Possible Context",
            "- Implications for Policy/Operations",
            "- Recommendations (short, prioritized)",
            "",
            "Guidelines:",
            "- Cite specific windows when referencing notable effects.",
            "- Ground claims in provided SHAP evidence; avoid speculation.",
            "- Avoid mentioning SHAP explicitly; refer instead to 'feature importance' or 'model insights'.",
            "- Keep the report under 600 words.",
            "",
            "Window analyses:",
            f"{combined_text}",
        ]
        if shap_corpus:
            body += [
                "",
                "SHAP evidence for grounding (do not copy verbatim):",
                f"{shap_corpus}",
            ]
        return "\n".join(body)

    if variant == "v3":
        body = [
            "Produce a coherent, expert-level synthesis of the following analyses.",
            "",
            "Focus Areas:",
            "1) Feature drivers that are robust across windows vs. those that are context-specific.",
            "2) Temporal dynamics: how driver importance/directionality changes over windows.",
            "3) Practical takeaways for air quality monitoring and intervention prioritization.",
            "",
            "Style: concise, evidence-driven; prefer bullets and short sections. Avoid mentioning SHAP explicitly; refer instead to 'feature importance' or 'model insights'.",
            "",
            "Window analyses:",
            f"{combined_text}",
        ]
        if shap_corpus:
            body += [
                "",
                "Additional SHAP corpus (use as evidential backing):",
                f"{shap_corpus}",
            ]
        return "\n".join(body)
    
    if variant == "v4":
        body = [
        "Here are multiple analyses of pollutant behavior across time windows:"
        f"{combined_text}",
        "Please write a coherent essay summarizing key findings, trends, and implications for air quality management.",
        "Structure:",
        "- Executive summary (3-5 bullet points), avoid bold words.",
        "- Consistent Patterns Across Windows.",
        "- Check for seasonal effects",
        "- Divergences/Anomalies and Possible Context.",
        "",
        "Guidelines:",
        "- Bologna is not present"
        "- Cite specific windows when referencing notable effects.",
        "- Ground claims in provided SHAP evidence; avoid speculation.",
        "- Avoid mentioning SHAP explicitly; refer instead to 'feature importance' or 'model insights', do not show shap values.",
        "- Keep the report under 600 words.",
        ]

        if shap_corpus:
            body += [
                "",
                "Additional SHAP corpus (use as evidential backing):",
                f"{shap_corpus}",
            ]
        return "\n".join(body)

    # Default fallback to baseline if unknown
    return build_final_summary_prompt("v1", window_summaries, shap_corpus)


