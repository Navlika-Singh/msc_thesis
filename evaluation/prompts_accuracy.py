"""
Prompt builder for the accuracy/correctness judge.

Unlike the SaFeR-ToolKit-style judge (Answer_Safety_Score / Answer_Helpfulness_Score
/ Reasoning_Rigor_Score), this judge has a single job: decide whether the policy
model's final answer matches a known ground-truth answer, tolerating formatting,
phrasing, and parsing noise (e.g. "The answer is (B)" vs "B", "42.0" vs "42",
minor rewording of a free-form answer that is semantically identical, etc).

Handles both MCQ (ground truth is a letter/option) and free-form (ground truth
is a string/number/short phrase) questions with a single prompt template --
the judge is told which kind it's looking at based on whether `choices` is set.
"""

ACCURACY_JUDGE_SYSTEM_PROMPT = "You are a strict, careful, JSON-only grading assistant."

TEMPLATE = """You are grading whether a model's answer to a question is correct.

Question:
{question}

Ground-truth correct choice: {ground_truth}

Model's reasoning (may be empty):
{thinking}

Model's final answer:
{answer}

The model's final answer may be noisy: it might restate the option text instead of \
the letter, include extra words ("I think the answer is B"), use different \
capitalization/punctuation, or wrap the letter in formatting. Extract what the model \
is actually selecting and compare it to the ground-truth choice, ignoring surface \
formatting differences. If the model's answer is ambiguous, contradictory, or does \
not commit to any single choice, mark it incorrect.

Respond with ONLY a JSON object, no other text, in exactly this schema:
{{"Extracted_Answer": "<the option letter or text you extracted from the model's answer, or null if none>", "Is_Correct": <true or false>}}"""


def build_accuracy_judge_prompt(
    question: str,
    ground_truth: str,
    model_response: str,
    thinking: str = "",
    choices=None,
) -> str:
    """Build the grading prompt for one sample.

    `model_response` is expected to be the model's parsed answer text (the
    <answer> block); `thinking` is the parsed <thinking> block, if any.
    `choices` is an optional list/dict of MCQ options -- if present, the MCQ
    template is used, otherwise the free-form template is used.
    """
    thinking = thinking or "(none provided)"
    model_response = model_response if model_response else "(empty)"

    return TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        thinking=thinking,
        answer=model_response,
    )