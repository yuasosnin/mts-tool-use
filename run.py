import argparse
import json
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


class ChatCLI:
    def __init__(self, model_name: str, system_prompt: str = None):
        self.conversation: List[Dict[str, str]] = []

        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        if system_prompt:
            self.conversation.append({"role": "system", "content": system_prompt})

    def generate_response(self, user_input: str) -> str:
        self.conversation.append({"role": "user", "content": user_input})

        prompt = self.tokenizer.apply_chat_template(
            self.conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=2048,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                streamer=streamer
            )

        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True).strip()
        self.conversation.append({"role": "assistant", "content": response})

        return response

    def run(self):
        print("Chat started. Type 'quit' to exit.\n")

        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    break

                if user_input:
                    print("Assistant: ", end="", flush=True)
                    self.generate_response(user_input)
                    print()

            except KeyboardInterrupt:
                break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="HuggingFace model name")

    with open("booking-benchmark/multiple.json") as f:
        tools = json.loads(f.readline())["function"]

    tools_string = "\n".join(json.dumps(a) for a in tools)
    system = 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n' + tools_string + '\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'

    args = parser.parse_args()

    chat = ChatCLI(args.model, system)
    chat.run()

if __name__ == "__main__":
    main()
