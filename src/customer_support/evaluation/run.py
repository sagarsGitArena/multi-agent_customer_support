"""
Runnable entry point: builds the LangSmith eval dataset (once) and
runs the groundedness/consistency evaluator against it, logging an
Experiment visible in LangSmith's Tracing -> Evaluators view.

Location: src/customer_support/evaluation/run.py

Usage:
    python -m customer_support.evaluation.run

Requires LANGCHAIN_API_KEY (or LANGSMITH_API_KEY) in .env -- the same
key already used for tracing. Importing dataset.py pulls in
customer_support.config, which calls load_dotenv(), so by the time
the langsmith.Client() below is constructed the key is already in the
process environment.
"""

import logging

from langsmith import Client, evaluate

from customer_support.evaluation.dataset import EXAMPLES, target
from customer_support.evaluation.groundedness import evaluate_groundedness

logger = logging.getLogger(__name__)

DATASET_NAME = "multi-agent-customer-support-groundedness"


def ensure_dataset(client: Client) -> None:
    """Creates the dataset with its examples on first run. Later runs
    reuse whatever's already there -- this doesn't try to sync example
    edits, so if EXAMPLES changes meaningfully, delete the dataset in
    LangSmith first to pick up the new set."""

    if client.has_dataset(dataset_name=DATASET_NAME):
        logger.info("ensure_dataset: %s already exists, reusing", DATASET_NAME)
        return

    client.create_dataset(
        DATASET_NAME,
        description=(
            "Regression cases for agent response quality: groundedness (no "
            "unstated inferences asserted as fact) and self-consistency (no "
            "declining-then-answering contradictions). Built from real bugs "
            "found during manual testing -- see README 'Bugs found and "
            "fixed'."
        ),
    )
    client.create_examples(dataset_name=DATASET_NAME, examples=EXAMPLES)
    logger.info("ensure_dataset: created %s with %d examples", DATASET_NAME, len(EXAMPLES))


def main() -> None:
    client = Client()
    ensure_dataset(client)

    results = evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[evaluate_groundedness],
        client=client,
        experiment_prefix="groundedness",
        description="Agent response quality: groundedness + self-consistency.",
    )
    print(results)


if __name__ == "__main__":
    main()
