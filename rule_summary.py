import argparse
import re
import os
import json
import logging
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from EasyChatTemplating.util_tools import convert_userprompt_transformers, skip_special_tokens_transformers

# 输出文件中的规则格式速查：
# 1. *_rules.txt
#    每行一个 JSON 记录，predicted_rules 是“类别 -> 规则列表”：
#    {"text": "...", "labels": [["EU", "organization"]], "status": "success",
#     "predicted_rules": {"organization": ["union"]}}
# 2. *_validrules.txt
#    每行一个 JSON 记录，right_rules / wrong_rules 是“单条规则”的列表，
#    列表中的每个元素都是 {类别: 规则文本}：
#    {"text": "...", "label": [["EU", "organization"]],
#     "orignal_rules": {"organization": ["union"]},
#     "right_rules": [{"organization": "union"}],
#     "wrong_rules": [],
#     "status": "success",
#     "predict_labels": [["EU", "organization"]]}
# 3. *_summaryrules.txt
#    整个文件是一个 JSON 对象，表示每个类别最终保留的高频规则列表：
#    {"organization": ["union", "financial institution"],
#     "person": ["name"],
#     "location": ["country"]}
# 4. *_wrongsummaryrules.txt
#    形式与 *_summaryrules.txt 相同，但统计的是 wrong_rules：
#    {"organization": ["bureau"], "person": ["journalist"]}

"""
本脚本是 GuideNER 的“规则构建”主入口，整体分为三步：
1. 读取训练集 train.jsonl，让 LLM 根据文本和实体标注总结候选规则；
2. 再把候选规则回填到识别任务中，检查这些规则是否真的能帮助识别原句中的实体；
3. 统计验证通过的高频规则，输出为后续推理脚本 run_withrule.py 使用的 summary rules。

运行前至少需要确认：
1. model_path_dict 中的本地模型路径存在；
2. datasets/<dataset_name>/train.jsonl 和 labels.jsonl 存在；
3. 机器环境已经安装 transformers、vllm、tqdm 等依赖。

运行后会在对应数据集目录下追加写出三个文件：
1. <model_name>_rules.txt：训练集样本对应的原始候选规则；
2. <model_name>_validrules.txt：回代验证后区分 right/wrong 的规则；
3. <model_name>_summaryrules.txt：最终汇总后的类别规则摘要。

注意：脚本使用追加模式 "a" 写文件，重复运行会把新结果继续追加到旧文件后面。
如果你希望得到一次全新的实验结果，通常需要先手动清理旧结果文件。
"""

# 从模型输出中抓取 JSON 形式的规则结果，例如 {"organization": ["union"]}。
result_pattern = r'\{.*\}'
# 从验证阶段输出中抓取实体列表，例如 [["EU", "organization"], ...]。
valid_pattern = r'\[\[(.*?)\]\]'

# 命令行中的 model_name 会映射到仓库外部的本地模型目录 ../model/<model_name>。
MODEL_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "model"))
DEFAULT_MODEL_NAME = "Llama-3.1-8B-Instruct"
model_path_dict = {
    "bge-m3": os.path.join(MODEL_ROOT, "bge-m3"),
    "Llama-3.1-8B-Instruct": os.path.join(MODEL_ROOT, "Llama-3.1-8B-Instruct"),
    "Ministral-3-8B-Instruct-2512": os.path.join(MODEL_ROOT, "Ministral-3-8B-Instruct-2512"),
    "Qwen2.5-7B-Instruct": os.path.join(MODEL_ROOT, "Qwen2.5-7B-Instruct"),
}
# 命令行中的 dataset_name 会映射到对应数据目录。
# 当前仓库实际只看到了 conll2003，其他数据集路径更像是预留接口。
dataset_path_dict = {"conll2003": "./datasets/conll2003",
                     "ace04": "./datasets/ace04",
                     "ace05": "./datasets/ace05",
                     "genia": "./datasets/genia"}

# 规则抽取 prompt：输入一条训练样本及其标注，让 LLM 总结“泛化规则”而不是复述具体实体名。
conll2003_prompt = """Task: Summarize the generic rules for each named entity category for the named entity recognition task based on the provided text and their corresponding annotations. The output must be structured in JSON format, where the keys represent the entity categories, and the values are lists of rules that have been summarized from the input text and their annotations.

Guidelines: 
(1) Avoid including specific entity names in the output and instead describe general patterns or features. 
(2) Only summarize rules for the entity categories that appear in the provided annotations. Do not include rules for any other categories.
(3) For each annotation provided, generate exactly one summarized rule corresponding to that label.
(4) The order of the summarized rules should strictly correspond to the order of the annotations, and the number of summarized rules must match the number of annotations.

Examples: 
Input Text: EU rejects German call to boycott British lamb . 
Annotations: [["EU", "organization"], ["German", "miscellaneous"], ["British", "miscellaneous"]]. 
Output: {{"organization": ["union"], "miscellaneous": ["ethnic groups", "ethnic groups"]}}
Input Text: Iraq 's Saddam meets Russia 's Zhirinovsky .
Annotations: [["Iraq", "location"], ["Saddam", "person"], ["Russia", "location"], ["Zhirinovsky", "person"]]
Output: {{"location": ["country", "country"], "person": ["name", "name"]}}
Input Text: S&P = DENOMS ( K ) 1-10-100 SALE LIMITS US / UK / CA
Annotations: [["S&P", "organization"], ["US", "location"], ["UK", "location"], ["CA", "location"]]
Output: {{"organization": ["financial institution"], "location": ["country", "country", "country"]}}

Summarize for:
Input Text: {input_text}
Annotations: {input_annotations}
Output:
"""

conll2003_valid_prompt= """Task: Please identify Person, Organization, Location and Miscellaneous Entity from the given text and rules.
The rules are in JSON format where the key is the entity category and the value is the schema contained in that category.

Examples:
Input Text: EU rejects German call to boycott British lamb.
Rules: {{"organization": ["union"], "miscellaneous": ["nationality"]}}
Output: [["EU", "organization"], ["German", "miscellaneous"], ["British", "miscellaneous"]]
Input Text: S&P = DENOMS ( K ) 1-10-100 SALE LIMITS US / UK / CA
Rules: {{"organization": ["financial institution"], "location": ["country", "country", "country"]}}
Output: [["Iraq", "location"], ["Saddam", "person"], ["Russia", "location"], ["Zhirinovsky", "person"]]
Input Text: -- E. Auchard , Wall Street bureau , 212-859-1736
Rules: {{"person": ["journalist"], "organization": ["newspaper bureau"]}}
Output: [["E. Auchard", "person"], ["Wall Street bureau", "organization"]]

Instructions:

Input Text: {input_text}
Rules: {summarized_rules}
Output:
"""

# 检查真实标注 labels 与 LLM 预测出的规则字典 result 是否在“数量”和“类别分布”上对齐。
# 这里不检查规则内容是否合理，只检查每个实体类别的条数是否能一一对应。
# 这样后面才能安全地把“第 n 个同类实体”与“第 n 条同类规则”配对起来。
def type_num_equal(labels, result):
    labels_len = len(labels)
    result_len = 0
    for k, v in result.items():
        result_len += len(v)
    if labels_len != result_len:
        return False
    
    tmp_dict = {}
    for label in labels:
        label_type = label[-1]
        if label_type not in tmp_dict:
            tmp_dict[label_type] = 0
        tmp_dict[label_type] += 1
        
    tmp_dict2 ={}
    for k,v in result.items():
        if k not in tmp_dict2:
            tmp_dict2[k] = 0
        tmp_dict2[k] += len(v)
    
    for k,v in tmp_dict.items():
        if k not in tmp_dict2:
            return False
        if v != tmp_dict2[k]:
            return False
    
    return True
    
    
# 将标注列表与规则字典做顺序对齐，得到 [label, rule] 的对应关系。
# 例如某句有两个 location，则分别取 result["location"] 中第 1 条和第 2 条规则。
def correspondings(labels, result):
    label_type_dict = {}
    final_result = []
    for label in labels:
        label_type = label[-1]
        if label_type not in label_type_dict:
            label_type_dict[label_type] = 0
        label_type_dict[label_type] += 1
        
        idx = label_type_dict[label_type] - 1
        rule = result[label_type][idx]

        final_result.append([label, rule])
    return final_result
        



def summary(rule_file_name, label_file, fw, top_k=20):
    # summary 系列文件最终写出的不是“按行记录”，而是单个 JSON 对象：
    # {
    #   "organization": ["rule_a", "rule_b", ...],
    #   "person": ["rule_c", ...],
    #   "location": ["rule_d", ...]
    # }
    # value 中只保留规则文本本身，不保留频次；频次仅用于内部排序选出 top_k。
    # result_dict 先按标签文件初始化实体类型，再统计每种规则出现的频次。
    result_dict = {}
    with open(label_file, 'r', encoding='utf8') as f:
        labels_dict = f.readlines()[0]
        labels_dict = json.loads(labels_dict)
        for k in labels_dict:
            if "geo" in k:
                k = "geo-political entity"
            result_dict[k] = {}

        
    with open(rule_file_name, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            line_json = json.loads(line)
            # 只有验证成功、且含有 right_rules 的记录才参与最终规则汇总。
            if "right_rules" not in line_json:
                continue
            right_rules = line_json["right_rules"]
            if len(right_rules) == 0:
                continue
            for right_rule in right_rules:
                for k, v in right_rule.items():
                    if "geo" in k:
                        k = "geo-political entity" 
                    entity_type = k
                    rule = v
                    if rule not in result_dict[entity_type]:
                        result_dict[entity_type][rule] = 0
                        result_dict[entity_type][rule] += 1   

    # 对每个实体类别，保留频次最高的若干条规则，作为最终摘要规则。
    rules_dict = {} 
    for k in result_dict:
        rules_dict[k] = []
        tmp_list = sorted(result_dict[k].items(), key=lambda x:x[-1], reverse=True)
        for j, tmp in enumerate(tmp_list):
            if j > top_k:
                break
            rules_dict[k].append(tmp[0])
     
    fw.write(json.dumps(rules_dict))
    fw.close()

            

def predict_batch(outputs, tokenizer, fw, texts, labels):
    # 处理“规则抽取阶段”的一批模型输出，并把原始候选规则落盘。
    for j, output in enumerate(outputs):
        clean_text = skip_special_tokens_transformers(tokenizer, output.outputs[0].text)
        result = re.search(result_pattern, clean_text, re.DOTALL)
        
        result_dict = {}
        result_dict["text"] = texts[j]
        result_dict["labels"] = labels[j]
        
        # 如果模型输出中能解析出 JSON 规则，就记为 success。
        if result is not None:
            try:
                result = eval(result.group())
                result_dict["status"] = "success"
                result_dict["predicted_rules"] = result
            except:
                result_dict["status"] = "eval_wrong"
                result_dict["predicted_rules"] = []
        # 如果没有抽到合法 JSON，则记录为空结果，供后续分析失败样本。
        else:
            result_dict["status"] = "none_wrong"
            result_dict["predicted_rules"] = []
        try:
            fw.write(json.dumps(result_dict))
            fw.write("\n")
            fw.flush()
        except:
            result_dict["status"] = "write_wrong"
            result_dict["predicted_rules"] = []
            fw.write(json.dumps(result_dict))
            fw.write("\n")
            fw.flush()
            
def valid_batch(outputs, tokenizer, fw, texts, labels, rules_list):
    """
    处理“规则验证阶段”的一批模型输出，并将每条规则标记为有效或无效。

    参数含义：
    - outputs: 当前 batch 的模型生成结果。每个元素对应一条验证 prompt 的输出。
    - tokenizer: 用于清理模型输出中的特殊 token。
    - fw: 写出验证结果的文件句柄。
    - texts: 当前 batch 中的原始句子列表。
    - labels: 当前 batch 中每条句子的真实实体标注列表。
    - rules_list: 当前 batch 中每条句子的候选规则 dict。

    核心思路：
    1. 先把每条样本的真实实体与候选规则做“按类别、按顺序”的一一对应；
    2. 再看模型在“原句 + 这些规则”的条件下，是否真的识别回了这些实体；
    3. 若某实体被识别回来了，就认为与它绑定的那条规则在该样本上是有效的，
       记入 right_rules，否则记入 wrong_rules。
    """
    # 处理“规则验证阶段”的一批模型输出。
    # 思路是：把某条训练样本的规则再喂回模型，看能否重新识别出原标注实体。
    for j, output in enumerate(outputs):
        text = texts[j]
        real_labels = labels[j]
        rules = rules_list[j]
        # corres 中每一项是 [真实实体标注, 与之顺序对齐的一条规则]。
        # 例如：
        # real_labels = [["Iraq", "location"], ["Saddam", "person"], ["Russia", "location"]]
        # rules = {"location": ["country", "country"], "person": ["name"]}
        # 则 corres 会变成：
        # [
        #   [["Iraq", "location"], "country"],
        #   [["Saddam", "person"], "name"],
        #   [["Russia", "location"], "country"]
        # ]
        corres = correspondings(real_labels, rules)
        right_rules = []
        wrong_rules = []
        
        result_dict = {}

        clean_text = skip_special_tokens_transformers(tokenizer, output.outputs[0].text)
        result = re.search(valid_pattern, clean_text, re.DOTALL)
        # 这里落盘的是“验证阶段”的记录，而不是第一阶段原始规则抽取结果。
        result_dict["text"] = text
        result_dict["label"] = real_labels
        result_dict["orignal_rules"] = rules
        
        if result is not None:
            try:
                # 验证阶段期望模型输出实体列表，如 [[实体名, 实体类型], ...]。
                result = eval(result.group())
                # 统一地名类别写法，避免 "geo" 与 "geo-political entity" 混用导致比较失败。
                for i in range(len(result)):
                    if "geo" in result[i][-1]:
                        result[i][-1] = "geo-political entity"
                    
                # 对每条“标注-规则”对应关系，检查该实体是否在模型识别结果中出现。
                # 出现则认为该规则对这个样本有效，加入 right_rules；否则进 wrong_rules。
                for k, cor in enumerate(corres):
                    label = cor[0]
                    type = label[-1]
                    if "geo" in type:
                        type = "geo-political entity"
                        label[-1] = "geo-political entity"
                    # 每条规则最终被记录成 {类别: 规则内容} 的形式，便于后续统计。
                    rules = {type:cor[-1]}
                    # 这里的判断非常直接：
                    # 只要这条真实标注 label 出现在模型预测结果 result 中，
                    # 就把与该实体绑定的规则视为“本样本上的有效规则”。
                    if label in result:
                        right_rules.append(rules)
                    else:
                        wrong_rules.append(rules)
                
                result_dict["right_rules"] = right_rules
                result_dict["wrong_rules"] = wrong_rules
                result_dict["status"] = "success"
                result_dict["predict_labels"] = result
            except:
                result_dict["status"] = "eval_wrong"
                result_dict["predict_labels"] = []
        else:
            # 如果模型输出里连实体列表都没抽出来，这条验证记录记为 none_wrong。
            result_dict["status"] = "none_wrong"
            result_dict["predict_labels"] = []
            
        try:
            fw.write(json.dumps(result_dict))
            fw.write("\n")
            fw.flush()
        except:
            # 写文件失败时，仍然尽量把错误状态写出去，避免整批结果丢失。
            result_dict["status"] = "write_wrong"
            result_dict["predict_labels"] = []
            fw.write(json.dumps(result_dict))
            fw.write("\n")
            fw.flush()
            
def valied_rules(fr, fw, batch_size, valid_prompt, tokenizer, llm, sampling_params):
    """
    对第一阶段生成的候选规则做过滤，并把可验证的样本送入第二阶段验证。

    这个函数主要做两件事：
    1. 从 rules.txt 中筛掉结构不可靠、无法一一对齐的规则记录；
    2. 将剩余记录重新组织成 NER 验证输入，分批交给模型生成，
       再由 valid_batch(...) 判断哪些规则真正帮助识别出了原实体。

    为什么这里要先过滤？
    因为后续会依赖 correspondings(...) 按“同类别、同顺序”把实体和规则绑定。
    如果规则不是字典，或者规则数目与真实实体分布对不上，就无法安全地完成这种绑定。
    """
    # 读取规则抽取阶段生成的文件，只保留“结构可对齐”的规则记录进入验证阶段。
    messages = []
    texts = []
    labels = []
    rules_list = []
    for i, line in enumerate(fr):
        line_json = json.loads(line)
        result_dict = {}
        text = line_json["text"]
        entity_labels = line_json["labels"]
        rules = line_json["predicted_rules"]
        
        # 跳过无法做稳定验证的样本：
        # 1. 原始标签为空：没有参照答案，就谈不上验证规则有效性；
        # 2. 规则不是字典：第一阶段输出没有遵守“类别 -> 规则列表”的约定结构；
        # 3. 规则条数和类别数量对不上真实标注：后面无法把“第 n 个实体”稳定映射到“第 n 条规则”。
        if len(entity_labels) == 0:
            continue
        if not isinstance(rules, dict):
            continue
        if not type_num_equal(entity_labels, rules):
            continue
        
        # 把规则重新放回 NER 任务，让模型在“原句 + 规则”的条件下再次做实体识别。
        # 随后 valid_batch(...) 会比较预测实体和真实实体，
        # 决定每条规则应进入 right_rules 还是 wrong_rules。
        prompt_predict = valid_prompt.format(except_rules="", input_text = text, summarized_rules = rules)
        message = convert_userprompt_transformers(tokenizer, prompt_predict, add_generation_prompt=True)
        
        # 先攒满一个 batch 再统一生成，减少模型调用次数。
        if len(messages) < batch_size - 1:
            texts.append(text)
            labels.append(entity_labels)
            messages.append(message)
            rules_list.append(rules)
        else:
            texts.append(text)
            labels.append(entity_labels)
            messages.append(message)
            rules_list.append(rules)
           
            outputs = llm.generate(messages, sampling_params)
            valid_batch(outputs, tokenizer, fw, texts, labels, rules_list)
            messages = []
            texts = []
            labels = []
            rules_list = []
    
    # 别遗漏最后一个不足 batch_size 的尾批次。
    if len(messages) > 0:
        outputs = llm.generate(messages, sampling_params)
        valid_batch(outputs, tokenizer, fw, texts, labels, rules_list)
    
    fw.close()
            
            
        


def summary(rule_file_name, label_file, fw, top_k=20):
    # Override the earlier implementation and emit triples:
    # [entity_type, rule_text, support_examples]
    result_dict = {}
    with open(label_file, 'r', encoding='utf8') as f:
        labels_dict = json.loads(f.readlines()[0])
        for k in labels_dict:
            if "geo" in k:
                k = "geo-political entity"
            result_dict[k] = {}

    with open(rule_file_name, 'r', encoding='utf8') as f:
        for line in f:
            line_json = json.loads(line)
            if "orignal_rules" not in line_json or "label" not in line_json or "predict_labels" not in line_json:
                continue

            rules = line_json["orignal_rules"]
            labels = line_json["label"]
            predict_labels = line_json["predict_labels"]
            if not isinstance(rules, dict):
                continue

            normalized_predict_labels = {
                (label[0], "geo-political entity" if "geo" in label[-1] else label[-1])
                for label in predict_labels
            }

            for label, rule in correspondings(labels, rules):
                entity_text = label[0]
                entity_type = "geo-political entity" if "geo" in label[-1] else label[-1]
                if (entity_text, entity_type) not in normalized_predict_labels:
                    continue

                if rule not in result_dict[entity_type]:
                    result_dict[entity_type][rule] = {"count": 0, "support_examples": []}

                result_dict[entity_type][rule]["count"] += 1
                if entity_text not in result_dict[entity_type][rule]["support_examples"]:
                    result_dict[entity_type][rule]["support_examples"].append(entity_text)

    summary_rules = []
    for entity_type in result_dict:
        tmp_list = sorted(
            result_dict[entity_type].items(),
            key=lambda x: x[-1]["count"],
            reverse=True
        )
        for j, (rule_text, rule_info) in enumerate(tmp_list):
            if j >= top_k:
                break
            summary_rules.append([entity_type, rule_text, rule_info["support_examples"]])

    fw.write(json.dumps(summary_rules, ensure_ascii=False))
    fw.close()


def main():
    # 该脚本通常直接运行：
    # python rule_summary.py --dataset_name conll2003 --model_name Llama-3.1-8B-Instruct
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name',
                        default='conll2003',
                        choices=["conll2003", "ace04", "ace05", "genia"])
    parser.add_argument('--model_name',
                        default=DEFAULT_MODEL_NAME,
                        choices=list(model_path_dict.keys()))
    parser.add_argument('--temperature',
                        default=0.8,
                        type=float),
    parser.add_argument('--top_p',
                        default=0.95,
                        type=float),
    batch_size = 32
    args = parser.parse_args()
    
    # 根据命令行参数解析出模型目录和数据目录。
    model_path = model_path_dict[args.model_name]
    dataset_path = dataset_path_dict[args.dataset_name]
    
    # tokenizer 用于 chat template 与输出清洗，vLLM 用于实际生成。
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model=model_path,
        enforce_eager=True,
        max_model_len=8192,
    )
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=256)
    
    # 三类输出文件：
    # 1. rules.txt：原始候选规则；
    # 2. validrules.txt：验证后的规则；
    # 3. summaryrules.txt：最终供推理使用的摘要规则。
    rule_file_name = os.path.join(dataset_path, f"{args.model_name}_rules.txt")
    valid_rule_file_name = os.path.join(dataset_path, f"{args.model_name}_validrules.txt")
    label_file = os.path.join(dataset_path, "labels.jsonl")
    summary_file_name = os.path.join(dataset_path, f"{args.model_name}_summaryrules.txt")
    fw = open(rule_file_name, "a", encoding='utf8')
    
    messages = []
    texts = []
    labels = []
    
    task_prompt = eval(f"{args.dataset_name}_prompt")
    valid_prompt = eval(f"{args.dataset_name}_valid_prompt")
    
    # 第一步：遍历训练集，为每条样本生成候选规则。
    with open(os.path.join(dataset_path, "train.jsonl"), "r", encoding='utf8') as f:
        for i, line in tqdm(enumerate(f)):
            line_json = json.loads(line)
            
            text = line_json["text"]
            entity_labels = line_json["entity_labels"]
            
            # 无实体样本对规则总结帮助有限，这里直接跳过。
            if len (entity_labels) == 0:
                continue
            
            prompt_predict = task_prompt.format(input_text = text, input_annotations = entity_labels)
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
        
        fw.close()
    
    # 第二步：验证 rules.txt 中的规则是否真的能帮助识别原句实体。
    fr = open(rule_file_name, 'r', encoding='utf8')
    fw = open(valid_rule_file_name, 'a', encoding='utf8')
    
    valied_rules(fr, fw, batch_size, valid_prompt, tokenizer, llm, sampling_params)
    
    # 第三步：从验证通过的规则中统计高频规则，得到最终摘要规则。
    fw = open(summary_file_name, 'a', encoding='utf8')
    summary(valid_rule_file_name, label_file, fw)
    
    
if __name__ == "__main__":
    main()
