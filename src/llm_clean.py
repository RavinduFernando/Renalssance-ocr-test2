

import os
import yaml
from openai import OpenAI


def load_llm_client(config_path: str = "configs/default.yaml"):
    """
    Load LLM client and model name from YAML config.
    Works with LM Studio OpenAI-compatible API or cloud OpenAI.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    llm_cfg = config["llm"]

    provider = llm_cfg.get("provider", "local")
    base_url = llm_cfg["base_url"]
    model_name = llm_cfg["model_name"]

    if provider == "local":
        # LM Studio: API key is unused but required by client
        api_key = "not-needed"
    else:
        # Cloud OpenAI or other provider
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in environment variables.")

    client = OpenAI(base_url=base_url, api_key=api_key)

    return client, model_name, llm_cfg


def clean_ocr_with_llm(
    text: str,
    client: OpenAI,
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    """
    Use the LLM to clean up OCR output.
    """
    prompt = (
        "You are an expert in early modern Spanish and Latin printing.\n"
        "The following text comes from an OCR system and may contain small errors.\n"
        "TASK:\n"
        "- Correct obvious OCR character errors only.\n"
        "- Do NOT add or remove whole words.\n"
        "- Do NOT ask for more text or explain anything.\n"
        "- If the input is very short or you are unsure, just repeat it unchanged.\n"
        "Return only the corrected text, with no explanations.\n\n"
        f"OCR text:\n{text}\n"
    )

    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()




## socrece for notebook 04

import re

def rule_based_fix(text: str) -> str:
    """
    Simple character-level clean-up before/without LLM.
    Adjust rules to match your dataset notes.
    """
    fixed = text

    # Example rules – modify as needed
    fixed = fixed.replace("ç", "z")
    fixed = fixed.replace("Ç", "Z")

    # Collapse repeated letters: 'qqe' -> 'qe'
    fixed = re.sub(r"(.)\1+", r"\1", fixed)

    # Normalize whitespace
    fixed = re.sub(r"\s+", " ", fixed).strip()

    return fixed
