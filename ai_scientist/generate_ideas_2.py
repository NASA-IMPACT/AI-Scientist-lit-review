import glob
import json
import os
import os.path as osp
import pickle
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Union

import backoff
import numpy as np
import requests
from loguru import logger
from paperqa import Answer, Context, Doc, Docs, Settings, Text, ask
from tqdm import tqdm

from ai_scientist.llm import extract_json_between_markers, get_response_from_llm


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            # Convert datetime to string in ISO format
            return obj.isoformat()
        elif isinstance(obj, set):
            # Convert set to list
            return list(obj)
        return super().default(obj)


def group_contexts_by_doc(contexts: List[dict]) -> Dict[str, Dict]:
    grouped_contexts = defaultdict(lambda: {"doc_details": None, "contexts": []})

    for context in contexts:
        doc_key = context["doc"]["key"]

        # If doc_details are not set for this doc_key, assign them
        if grouped_contexts[doc_key]["doc_details"] is None:
            grouped_contexts[doc_key]["doc_details"] = context["doc"]

        # Add the context to the list of contexts for this document
        grouped_contexts[doc_key]["contexts"].append(
            {
                "context": context["context"],
                "score": context["score"],
                "text": context["text"],
            },
        )

    res = {}
    for doc_key, items in dict(grouped_contexts).items():
        res[doc_key] = items["doc_details"]
    return res


def _flatten_pqa_answer(answer: Answer) -> dict:
    res = answer.model_dump()
    contexts = []
    for context in answer.contexts:
        dct = dict(
            context=context.context,
            score=context.score,
            text=context.text.text,
            doc=context.text.doc.model_dump(),
        )
        contexts.append(dct)
    res["documents"] = group_contexts_by_doc(contexts)
    res.pop("contexts", [])
    res.pop("id", None)
    return res


def _compress_dict(data_dict: dict):
    compressed_dict = {
        "question": data_dict.get("question"),
        "answer": data_dict.get("answer"),
        "documents": {},
    }

    # filter out documents, keeping only 'context' and 'citation'
    if "documents" in data_dict:
        for doc_key, doc_data in data_dict["documents"].items():
            compressed_dict["documents"][doc_key] = {
                "context": data_dict.get("context"),
                "citation": doc_data.get("citation"),
            }

    return compressed_dict


def minify_summary(summary: List[dict]) -> List[dict]:
    res = map(_compress_dict, summary)
    return list(res)


def generate_summary(
    ideas,
    base_dir,
    client,
    model,
    paper_qa_obj,
    compress_summary: bool = True,
    auto_save: bool = True,
    summary_file: str = "summary.json",
    debug: bool = False,
) -> List[dict]:
    res = []
    total_cost = 0
    token_counts = 0

    # if ideas not passed, try loading
    if not ideas:
        logger.debug(f"Loading ideas from ideas.json")
        with open(osp.join(base_dir, "ideas.json")) as f:
            ideas = json.load(f)

    unwanted_keys = ["formatted_answer", "config_md5"]
    for idx, idea in tqdm(enumerate(ideas)):
        logger.info(f"Idea [{idx+1}/{len(ideas)}] | {idea}")
        answer = _flatten_pqa_answer(paper_qa_obj.query(idea))
        for _uk in unwanted_keys:
            answer.pop(_uk, None)
        logger.info(f"Idea[{idx+1}/{len(ideas)}] | {idea} | answer={answer['answer']}")
        res.append(answer)
        total_cost += answer["cost"]
        token_counts += np.sum(list(answer["token_counts"].values()))
        if debug:
            logger.debug(
                f"Currently cost at {round(total_cost, 3)} and {token_counts} tokens.",
            )
    logger.debug(f"summary json compression = {compress_summary}")
    if compress_summary:
        res = minify_summary(res)
    if auto_save:
        summary_path = osp.join(base_dir, summary_file)
        logger.info(f"Dumping summary to {summary_path}")
        with open(summary_path, "w") as f:
            json.dump(res, f, indent=4, cls=CustomJSONEncoder)
    logger.info(
        f"Total cost :: {round(total_cost, 3)} | Token Counts :: {token_counts}",
    )
    return res


def build_paperqa_index(
    paths: List[str],
    index_path: Optional[os.PathLike] = None,
) -> Docs:
    docs = Docs()

    # Check if the index path is provided and if the index file exists
    if index_path and osp.isdir(index_path):
        index_path = osp.join(index_path, "pqa.pkl")
    if index_path and osp.isfile(index_path):
        logger.info(f"Loading from saved index :: {index_path}")
        with open(index_path, "rb") as f:
            docs = pickle.load(f)
        return docs

    # If index_path does not exist or file doesn't exist, build a new index
    logger.info(f"Building new index for {len(paths)} files")
    for path in paths:
        if path.startswith(("http", "www.")):
            docs.add_url(path)
        else:
            docs.add(path)

    # Save the new index if index_path is provided
    if index_path:
        # Default to pqa.pkl if index_path is a directory
        if osp.isdir(index_path):
            index_path = osp.join(index_path, "pqa.pkl")

        # Log saving operation (assuming logger is defined elsewhere)
        logger.info(f"Saving to {index_path}")

        # Save the docs object to the index_path file
        with open(index_path, "wb") as f:
            pickle.dump(docs, f)

    return docs


if __name__ == "__main__":
    MAX_NUM_GENERATIONS = 32
    NUM_REFLECTIONS = 5
    import argparse

    parser = argparse.ArgumentParser(description="Generate AI scientist ideas")
    # add type of experiment (nanoGPT, Boston, etc.)
    parser.add_argument(
        "--experiment",
        type=str,
        default="nanoGPT",
        help="Experiment to run AI Scientist on.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-05-13",
        choices=[
            "claude-3-5-sonnet-20240620",
            "gpt-4o-2024-05-13",
            "deepseek-coder-v2-0724",
            "llama3.1-405b",
        ],
        help="Model to use for AI Scientist.",
    )
    parser.add_argument(
        "--skip-idea-generation",
        action="store_true",
        help="Skip idea generation and use existing ideas.",
    )
    parser.add_argument(
        "--check-novelty",
        action="store_true",
        help="Check novelty of ideas.",
    )
    args = parser.parse_args()

    # Create client
    if args.model == "claude-3-5-sonnet-20240620":
        import anthropic

        print(f"Using Anthropic API with model {args.model}.")
        client_model = "claude-3-5-sonnet-20240620"
        client = anthropic.Anthropic()
    elif args.model.startswith("bedrock") and "claude" in args.model:
        import anthropic

        # Expects: bedrock/<MODEL_ID>
        client_model = args.model.split("/")[-1]

        print(f"Using Amazon Bedrock with model {client_model}.")
        client = anthropic.AnthropicBedrock()
    elif args.model.startswith("vertex_ai") and "claude" in args.model:
        import anthropic

        # Expects: vertex_ai/<MODEL_ID>
        client_model = args.model.split("/")[-1]

        print(f"Using Vertex AI with model {client_model}.")
        client = anthropic.AnthropicVertex()
    elif args.model == "gpt-4o-2024-05-13":
        import openai

        print(f"Using OpenAI API with model {args.model}.")
        client_model = "gpt-4o-2024-05-13"
        client = openai.OpenAI()
    elif args.model == "deepseek-coder-v2-0724":
        import openai

        print(f"Using OpenAI API with {args.model}.")
        client_model = "deepseek-coder-v2-0724"
        client = openai.OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
        )
    elif args.model == "llama3.1-405b":
        import openai

        print(f"Using OpenAI API with {args.model}.")
        client_model = "meta-llama/llama-3.1-405b-instruct"
        client = openai.OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
    else:
        raise ValueError(f"Model {args.model} not supported.")

    base_dir = osp.join("templates", args.experiment)
    results_dir = osp.join("results", args.experiment)
    paper_paths = glob.glob(osp.join(base_dir, "papers", "*.pdf"))

    questions = []
    with open(osp.join(base_dir, "ideas.json")) as f:
        questions = json.load(f)

    pqa_docs = build_paperqa_index(paper_paths, index_path=base_dir)
    summary = generate_summary(
        ideas=questions,
        base_dir=base_dir,
        client=client,
        model=client_model,
        paper_qa_obj=pqa_docs,
        debug=True,
    )
