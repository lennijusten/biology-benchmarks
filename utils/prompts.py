# utils/prompts.py

# Copied from Inspect implementation: https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/src/inspect_ai/solver/_multiple_choice.py
SINGLE_ANSWER_TEMPLATE = r"""
Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.

{question}
{choices}
""".strip()

# Copied from Inspect implementation: https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/src/inspect_ai/solver/_multiple_choice.py
MULTIPLE_ANSWER_TEMPLATE = r"""
Answer the following multiple choice question where multiple answers may be correct. The entire content of your response should be of the following format: 'ANSWER: $LETTERS' (without quotes) where LETTERS is one or more of {letters}.
{question}
{choices}
""".strip()

# Copied from Inspect implementation: https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/src/inspect_ai/solver/_prompt.py
DEFAULT_COT_TEMPLATE = r"""
{prompt}
Before answering, reason in a step-by-step manner as to get the right answer. Provide your answer at the end on its own line in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the question.
""".strip()

MULTIPLE_CHOICE_TEMPLATE_COT = r"""
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}. Before answering, reason in a step-by-step manner as to get the right answer.

{question}

{choices}
""".strip()

FEWSHOT_EXAMPLE_TEMPLATE = "\n{question}\n\n{choices}\n\nANSWER: {target}\n".strip()

MULTIPLE_CHOICE_TEMPLATE_FEWSHOT = "{examples}\n\n{question}"