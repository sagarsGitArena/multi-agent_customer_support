"""
LLM-as-judge groundedness/consistency evaluator.

Location: src/customer_support/evaluation/groundedness.py

Grades a single agent response on two things, both found as real bugs
during manual testing (see README "Bugs found and fixed"):

1. Groundedness -- every factual claim in the response must be
   supported by the conversation context (tool results, or something
   the customer actually said), not asserted as fact from an
   unstated inference. Catches things like catalog_agent asserting a
   "diverse taste" characterization it read off an earlier invoice
   answer rather than something the customer stated.
2. Self-consistency -- the response must not contradict itself, e.g.
   declining to help with something and then helping with it anyway
   in the same response. Catches the invoice/catalog mixed-intent
   contradiction bug.

`score_response` is the low-level scorer (context + response strings
in, a verdict out) -- usable standalone, e.g. in tests. `evaluate_groundedness`
wraps it in the shape LangSmith's `evaluate()` expects (a `run`/`example`
pair in, a scored dict out).
"""

import logging

from pydantic import BaseModel, Field

from customer_support.config import get_llm

logger = logging.getLogger(__name__)


class GroundednessVerdict(BaseModel):
    grounded: bool = Field(
        description=(
            "True if every factual claim in the response is either directly "
            "supported by the context (a tool result, or something the "
            "customer explicitly said), or a clearly-labeled inference "
            "('based on your recent purchases...'). False if the response "
            "asserts something unsupported as settled fact, or if it "
            "contradicts itself (e.g. declining to answer something and "
            "then answering it anyway)."
        )
    )
    reasoning: str = Field(description="One or two sentences explaining the verdict.")


GROUNDEDNESS_JUDGE_PROMPT = """You are grading a customer support \
agent's response for a music store, on two things:

1. Groundedness: every factual claim in the RESPONSE must be either
   - directly supported by information in the CONTEXT (a tool result, a \
database lookup, or a fact the customer stated themselves), or
   - a reasonable inference the response itself clearly flags as such \
(e.g. "based on your recent purchases...").
   A claim the response asserts as settled fact, but that was never \
actually stated by the customer or returned by a tool in the CONTEXT, \
is NOT grounded -- even if it happens to be a defensible guess.

2. Self-consistency: the response must not contradict itself -- e.g. \
declining to help with something ("I can't provide information about \
X") and then immediately providing X anyway in the same response.

Grade strictly. If in doubt, mark it NOT grounded and explain why in \
one or two sentences."""

judge_llm = get_llm().with_structured_output(GroundednessVerdict)


def score_response(context: str, response: str) -> GroundednessVerdict:
    """Grades one response against the conversation context that
    preceded it. Pure function of (context, response) -- no LangSmith
    dependency, so it's directly unit-testable."""

    verdict: GroundednessVerdict = judge_llm.invoke(
        [
            {"role": "system", "content": GROUNDEDNESS_JUDGE_PROMPT},
            {
                "role": "user",
                "content": f"CONTEXT:\n{context or '(no prior context)'}\n\nRESPONSE:\n{response}",
            },
        ]
    )
    logger.info(
        "score_response: grounded=%s reasoning=%s", verdict.grounded, verdict.reasoning
    )
    return verdict


def evaluate_groundedness(run, example) -> dict:
    """LangSmith evaluator entry point: run.outputs must contain
    "answer" and "context" (see evaluation/dataset.py's target
    function, which produces exactly that shape)."""

    outputs = run.outputs or {}
    verdict = score_response(outputs.get("context", ""), outputs.get("answer", ""))
    return {
        "key": "groundedness",
        "score": 1 if verdict.grounded else 0,
        "comment": verdict.reasoning,
    }
