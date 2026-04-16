import argparse
import re
import os
import json
import logging
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from EasyChatTemplating.util_tools import convert_userprompt_transformers, skip_special_tokens_transformers

result_pattern = r'\{.*\}'
label_pattern = r'\[\[(.*?)\]\]'
model_path_dict = {"llama3-chat": "../../pretrained_models/llama3-chat"}
dataset_path_dict = {"conll2003": "./datasets/conll2003",
                     "ace04": "./datasets/ace04",
                     "ace05": "./datasets/ace05",
                     "genia": "./datasets/genia"}


conll2003_rule_prompt = """Task: Please identify Person, Organization, Location and Miscellaneous Entity from the given text and rules. 
The rules provide an entity category followed by a list of patterns that match that category.

Rules:
{Rules}
Please note: Patterns not included in the above are not entities.

Examples:
Input Text: EU rejects German call to boycott British lamb.
Given the Input Text and Rules, only classify text as an entity if it matches a pattern; otherwise, it should not be classified as an entity. 
The Output is: [["EU", "organization"], ["German", "miscellaneous"], ["British", "miscellaneous"]]
Input Text: S&P = DENOMS ( K ) 1-10-100 SALE LIMITS US / UK / CA
Given the Input Text and Rules, only classify text as an entity if it matches a pattern; otherwise, it should not be classified as an entity. 
The Output is: [["Iraq", "location"], ["Saddam", "person"], ["Russia", "location"], ["Zhirinovsky", "person"]]
Input Text: -- E. Auchard , Wall Street bureau , 212-859-1736
Given the Input Text and Rules, only classify text as an entity if it matches a pattern; otherwise, it should not be classified as an entity. 
The Output is: [["E. Auchard", "person"], ["Wall Street bureau", "organization"]]

Identify Entities for: 
Input Text: {input_text}
Given the Input Text and Rules, only classify text as an entity if it matches a pattern; otherwise, it should not be classified as an entity. 
The Output is:
"""


def get_summary_rule(summary_rules_file):
    rule_summary = None
    final_rule = ""
    with open(summary_rules_file, 'r', encoding='utf8') as f:
        rule_summary = json.loads(f.readlines()[0])

    if isinstance(rule_summary, dict):
        grouped_rules = rule_summary
    else:
        grouped_rules = {}
        for item in rule_summary:
            if not isinstance(item, list) or len(item) != 3:
                continue
            entity_type, rule_text, _ = item
            if entity_type not in grouped_rules:
                grouped_rules[entity_type] = []
            grouped_rules[entity_type].append(rule_text)

    for k, v in grouped_rules.items():
        final_rule += k.capitalize() + ": "
        final_rule += str(v)
        final_rule += "\n"
    return final_rule


def predict_batch(outputs, tokenizer, fw, texts, labels):
    for j, output in enumerate(outputs):
        clean_text = skip_special_tokens_transformers(tokenizer, output.outputs[0].text)
        result = re.search(label_pattern, clean_text, re.DOTALL)
        
        result_dict = {}
        result_dict["text"] = texts[j]
        result_dict["labels"] = labels[j]
        
        # If llm generate the right result
        if result is not None:
            try:
                result = eval(result.group())
                result_dict["status"] = "success"
                result_dict["predicted_labels"] = result
            except:
                result_dict["status"] = "eval_wrong"
                result_dict["predicted_labels"] = []
        # if llm generate the wrong result or generate nothing
        else:
            result_dict["status"] = "none_wrong"
            result_dict["predicted_labels"] = []
        try:
            fw.write(json.dumps(result_dict))
            fw.write("\n")
            fw.flush()
        except:
            result_dict["status"] = "write_wrong"
            result_dict["predicted_labels"] = []
            fw.write(json.dumps(result_dict))
            fw.write("\n")
            fw.flush()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name',
                        default='conll2003',
                        choices=["conll2003", "ace04", "ace05", "genia"])
    parser.add_argument('--model_name',
                        default='llama3-chat')
    parser.add_argument('--temperature',
                        default=0.8,
                        type=float),
    parser.add_argument('--top_p',
                        default=0.95,
                        type=float),
    batch_size = 32
    args = parser.parse_args()
    
    model_path = model_path_dict[args.model_name]
    dataset_path = dataset_path_dict[args.dataset_name]
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path)
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=256)
    
    reuslt_file_name = os.path.join(dataset_path, f"{args.model_name}_withrule_result_detail.txt")
    fw = open(reuslt_file_name, "a", encoding='utf8')
    
    messages = []
    texts = []
    labels = []
    
    task_prompt = eval(f"{args.dataset_name}_rule_prompt")
    
    summary_rules_file = os.path.join(dataset_path, f"{args.model_name}_summaryrules.txt")
    summary_rule = get_summary_rule(summary_rules_file)
        
    
    
    with open(os.path.join(dataset_path, "test.jsonl"), "r", encoding='utf8') as f:
        for i, line in tqdm(enumerate(f)):
            line_json = json.loads(line)

            text = line_json["text"]
            entity_labels = line_json["entity_labels"]
            

            prompt_predict = task_prompt.format(Rules=summary_rule,
                                                input_text=text)
            message = convert_userprompt_transformers(tokenizer, prompt_predict, add_generation_prompt=True)
            
            if len(messages) < batch_size - 1:
                texts.append(text)
                labels.append(entity_labels)
                messages.append(message)
            else:
                texts.append(text)
                labels.append(entity_labels)
                messages.append(message)
                
                outputs = llm.generate(messages, sampling_params)
                
                predict_batch(outputs, tokenizer, fw, texts, labels)
                
                messages = []
                texts = []
                labels = []
        
        if len(messages) > 0:
            outputs = llm.generate(messages, sampling_params)
            predict_batch(outputs, tokenizer, fw, texts, labels)
        
  
    
if __name__ == "__main__":
    main()
