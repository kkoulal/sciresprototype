import httpx
import json
import re
from typing import Dict, Any, List, Union
from .config import (
    AGENT_LLM_URL, AGENT_LLM_MODEL, AGENT_LLM_KEY, AGENT_LLM_TIMEOUT, AGENT_LLM_MODEL_FAST,
    GEMINI_API_KEY, GEMINI_BASE_URL,
)

def _sanitize_llm_output(text: str) -> str:
    """Strips <think> blocks in case the reasoning parser misses them or they are returned."""
    cleaned = text or ""
    cleaned = re.sub(r"(?is)<think>.*</think>", "", cleaned).strip()
    cleaned = re.sub(r"(?is)<think>.*$", "", cleaned, flags=re.DOTALL).strip()
    if "</think>" in cleaned.lower():
        parts = re.split(r"(?is)</think>", cleaned)
        cleaned = parts[-1].strip()
    return cleaned

def _extract_json_from_text(text: str) -> Union[Dict[str, Any], list]:
    """Find and parse a JSON object/array from the LLM output.

    Handles Claude's common shapes: ```json fences, plain ``` fences, and prose
    preamble followed by a bare {...} or [...] block.
    """
    candidates: list[str] = []

    # 1. ```json ... ```  or  plain ``` ... ``` fenced block
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        candidates.append(fence.group(1).strip())

    # 2. The whole text, stripped (model returned bare JSON)
    candidates.append((text or "").strip())

    # 3. From the first { or [ to the last } or ] (prose wrapped around JSON)
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start, end = text.find(open_ch), text.rfind(close_ch)
        if start != -1 and end != -1 and end > start:
            candidates.append(text[start:end + 1])

    for cand in candidates:
        if not cand:
            continue
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue

    print(f"Failed to parse JSON. Content was:\n{(text or '')[:1000]}")
    return {}

def call_llm(
    messages: List[Dict[str, str]],
    temperature: float = 0.1,
    max_tokens: int = 4000,
    extract_json: bool = False,
    stream_callback=None,
    model: str = None,
) -> Union[str, Dict[str, Any]]:
    """
    Calls the configured LLM endpoint (OpenRouter by default).
    Models that start with 'gemini-' are automatically routed to Google's
    OpenAI-compatible endpoint using GEMINI_API_KEY.
    """
    effective_model = model or AGENT_LLM_MODEL

    # Route Gemini models to Google's OpenAI-compatible endpoint
    if effective_model.startswith("gemini-"):
        chat_url = GEMINI_BASE_URL.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {GEMINI_API_KEY}",
            "Content-Type": "application/json",
        }
    else:
        chat_url = f"{AGENT_LLM_URL}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {AGENT_LLM_KEY}",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Scires ProductSimulation",
        }

    payload = {
        "model": effective_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream_callback is not None,
    }

    with httpx.Client(timeout=AGENT_LLM_TIMEOUT, headers=headers) as client:
        try:
            if stream_callback:
                with client.stream("POST", chat_url, json=payload) as response:
                    response.raise_for_status()
                    content = ""
                    for line in response.iter_lines():
                        if line.startswith("data: ") and line != "data: [DONE]":
                            try:
                                chunk = json.loads(line[6:])
                                delta_content = chunk["choices"][0]["delta"].get("content", "")
                                delta_reasoning = chunk["choices"][0]["delta"].get("reasoning_content", "")
                                combined_delta = delta_reasoning + delta_content
                                if combined_delta:
                                    content += combined_delta
                                    stream_callback(content)
                            except Exception:
                                pass
            else:
                response = client.post(chat_url, json=payload)
                response.raise_for_status()
                result = response.json()
                content = result["choices"][0]["message"].get("content", "")

                reasoning = result["choices"][0]["message"].get("reasoning_content", "")
                if reasoning and not content.startswith("<think>"):
                    content = f"<think>\n{reasoning}\n</think>\n{content}"

        except Exception as e:
            print(f"LLM Call Failed: {e}")
            return {} if extract_json else f"Error: {e}"

    sanitized_content = _sanitize_llm_output(content)

    if extract_json:
        return _extract_json_from_text(sanitized_content)

    return sanitized_content
