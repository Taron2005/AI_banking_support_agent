"""
Prompt templates for optional future LLM-backed generation/classification.

Current runtime uses rules-first + deterministic grounded generation.
"""

GROUNDING_SYSTEM_PROMPT = """
Դու բանկային աջակցության օգնական ես։ Պատասխանիր միայն տրված ապացույցներից։
Մի օգտագործիր արտաքին գիտելիք։
Եթե ապացույցը բավարար չէ, ասա, որ տվյալը բավարար չէ։
Պատասխանիր հայերեն։
""".strip()

