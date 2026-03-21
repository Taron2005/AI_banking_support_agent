"""
User-side prompt templates for Gemini grounded answer synthesis.

Canonical architecture: `runtime/rag_prompts.py` (system EN + Armenian voice preamble + comparison +
refusal templates + optional JSON intent). `runtime/llm.py` imports the English system string from there.
"""

from .rag_prompts import voice_answer_preamble_with_footnote

GROUNDING_SYSTEM_PROMPT = "Evidence-only Armenian banking assistant (see runtime.llm.RAG_SYSTEM_MESSAGE)."

# Single canonical footnote appended post-process if the model omits it (must match template below).
STANDARD_AI_FOOTNOTE_LINE = (
    "Նշում․ պատասխանը կազմված է արհեստական բանականությամբ՝ բացառապես վերևում նշված "
    "պաշտոնական կայքի հատվածների հիման վրա։"
)

LLM_USER_ANSWER_INSTRUCTIONS = voice_answer_preamble_with_footnote(STANDARD_AI_FOOTNOTE_LINE).strip()
