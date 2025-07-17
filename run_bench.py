import argparse
import datetime
import json
import re

import torch
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
from transformers import AutoModelForCausalLM, AutoTokenizer, logging, pipeline


logging.set_verbosity_error()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--bench_subset", default="bfcl_simple", type=str)
    parser.add_argument("--batch_size", default=4, type=int)
    args, _ = parser.parse_known_args()
    return args


def read_bfcl_data(bench, subset="simple"):
    if bench == "bfcl":
        home = "Berkeley-Function-Calling-Leaderboard"
        source = f"BFCL_v3_{subset}"
    elif bench == "booking":
        home = "booking-benchmark"
        source = subset

    with open(f"{home}/{source}.json", "r") as f:
        data = [json.loads(x.strip()) for x in f.readlines()]

    with open(f"{home}/possible_answer/{source}.json", "r") as f:
        labels = [json.loads(x.strip()) for x in f.readlines()]

    return data, labels

def prepare_data(data):
    tokenizer_tool = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", padding_side="left")
    dataset = []
    for x in data:
        text = tokenizer_tool.apply_chat_template(
            x["question"][0],
            tools=[{"type": "function", "function": f} for f in x["function"]],
            add_generation_prompt=True,
            tokenize=False
        )
        text = text.replace("You are a helpful assistant", f"You are a helpful assistant. Today is {datetime.date.today().__str__()}")
        dataset.append(text)
    return dataset


def try_parse_tool_calls(content: str):
    """Try parse the tool calls."""
    tool_calls = []
    offset = 0
    for i, m in enumerate(re.finditer(r"<tool_call>\n(.+)?\n</tool_call>", content)):
        if i == 0:
            offset = m.start()
        try:
            func = json.loads(m.group(1))
            tool_calls.append({"type": "function", "function": func})
            if isinstance(func["arguments"], str):
                func["arguments"] = json.loads(func["arguments"])
        except json.JSONDecodeError as e:
            # print(f"Failed to parse tool calls: the content is {m.group(1)} and {e}")
            pass
    if tool_calls:
        if offset > 0 and content[:offset].strip():
            c = content[:offset]
        else:
            c = ""
        return {"role": "assistant", "content": c, "tool_calls": tool_calls}
    return {"role": "assistant", "content": re.sub(r"<\|im_end\|>$", "", content), "tool_calls": [{}]}


def eval_ast(spec, pred, true, test_category="simple"):
    return ast_checker(
        spec,
        [{pred["function"]["name"]: pred["function"]["arguments"]}],
        true,
        language="Python",
        test_category=test_category,
        model_name="gorilla-openfunctions-v2"  # unused
    )


def eval_bfcl(data, labels, responses, subset="simple"):
    total = 0
    correct = 0
    for example, label, response in zip(data, labels, responses):
        total += 1
        try:
            response_parsed = try_parse_tool_calls(response[0]["generated_text"])["tool_calls"][0]
            eva = eval_ast(example["function"], response_parsed, label["ground_truth"], test_category=subset)
            correct += eva["valid"]
        except Exception:
            pass
    return correct / total


def main(args):
    bench, subset = args.bench_subset.split("_")
    data, labels = read_bfcl_data(bench, subset)
    data_prepared = prepare_data(data)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    responses = generator(data_prepared, max_new_tokens=128, do_sample=False, batch_size=args.batch_size)
    accuracy = eval_bfcl(data, labels, responses, subset=subset)
    print(bench, subset, args.model_id, accuracy)


if __name__ == "__main__":
    args = get_args()
    main(args)
