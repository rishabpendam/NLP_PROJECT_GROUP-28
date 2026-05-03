"""
agents/llm_norep_ollama.py
LLM-NoRep agent backed by Ollama (fully local, no API key needed).

Ollama runs Llama 3 locally — the same model family used in the paper (Llama 3.1).
Start Ollama before running: `ollama serve` in a separate terminal.
Pull the model first: `ollama pull llama3`

This is the BASELINE agent (Section 4.3 — LLM-NoRep):
  - Receives its own private parameters and the dialogue history.
  - No reputation information is provided to it, regardless of the scenario flag.
"""
from __future__ import annotations
from typing import Optional
import requests

from agents.base_agent import BaseAgent
from agents.prompts import (
    BUYER_SYSTEM, SELLER_SYSTEM,
    build_buyer_user_message, build_seller_user_message,
)
from core.schema import Scenario
from core.dialogue import DialogueHistory

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3"


class LLMNoRepOllamaAgent(BaseAgent):
    """
    LLM-NoRep agent using Ollama.
    Reputation score is intentionally ignored even if scenario.rep_visible is True.
    """

    def __init__(self, role: str, model: str = DEFAULT_MODEL, temperature: float = 0.7):
        super().__init__(role)
        self.model = model
        self.temperature = temperature

    def act(
        self,
        scenario: Scenario,
        history: DialogueHistory,
        rep_score: Optional[float],
        opponent_answer: Optional[str],
    ) -> str:
        system_prompt = BUYER_SYSTEM if self.role == "buyer" else SELLER_SYSTEM

        if self.role == "buyer":
            user_msg = build_buyer_user_message(
                scenario, history, rep_score=None,   # NoRep: never pass reputation
                opponent_answer=opponent_answer,
            )
        else:
            user_msg = build_seller_user_message(
                scenario, history, rep_score=None,
                opponent_answer=opponent_answer,
            )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            "stream": False,
            "options": {"temperature": self.temperature},
        }

        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]
        except Exception as e:
            return f"THOUGHT: Error calling Ollama: {e}\nTALK: I need a moment.\nACTION: QUIT"
