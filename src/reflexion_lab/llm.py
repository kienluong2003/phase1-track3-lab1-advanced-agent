from __future__ import annotations
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from .schemas import QAExample, JudgeResult, ReflectionEntry
from .utils import normalize_answer
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM

# Load environment variables from tests/.env
load_dotenv("tests/.env")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> str:
    context_str = "\n".join([f"Title: {c.title}\nText: {c.text}" for c in example.context])
    
    prompt = f"Context:\n{context_str}\n\nQuestion: {example.question}"
    
    if agent_type == "reflexion" and reflection_memory:
        reflections_str = "\n".join([f"- {r}" for r in reflection_memory])
        prompt += f"\n\nPast Reflection Lessons:\n{reflections_str}\n\nPlease learn from these lessons to provide a better answer."

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": ACTOR_SYSTEM},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    return response.choices[0].message.content.strip()

def evaluator(example: QAExample, answer: str) -> JudgeResult:
    # First, a simple check
    if normalize_answer(example.gold_answer) == normalize_answer(answer):
        return JudgeResult(score=1, reason="Answer matches gold answer exactly.")

    # If not exact match, use LLM to judge
    prompt = f"""Gold Answer: {example.gold_answer}
Student Answer: {answer}

Please judge if the student answer is semantically equivalent to the gold answer."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": EVALUATOR_SYSTEM},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )
    
    data = json.loads(response.choices[0].message.content)
    
    # Sanitize null values for list fields to avoid Pydantic ValidationError
    if data.get("missing_evidence") is None:
        data["missing_evidence"] = []
    if data.get("spurious_claims") is None:
        data["spurious_claims"] = []
        
    return JudgeResult(**data)

def reflector(example: QAExample, attempt_id: int, answer: str, judge: JudgeResult) -> ReflectionEntry:
    context_str = "\n".join([f"Title: {c.title}\nText: {c.text}" for c in example.context])
    
    prompt = f"""Question: {example.question}
Context: {context_str}
Previous Attempt: {answer}
Feedback: {judge.reason}
Missing Evidence: {judge.missing_evidence}
Spurious Claims: {judge.spurious_claims}

Please analyze why the attempt failed and provide a lesson and a new strategy."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": REFLECTOR_SYSTEM},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.1
    )
    
    data = json.loads(response.choices[0].message.content)
    return ReflectionEntry(attempt_id=attempt_id, **data)
