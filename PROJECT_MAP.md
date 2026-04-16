# 项目概览

- 项目名称：GuideNER，对应论文 *GuideNER: Annotation Guidelines Are Better than Examples for In-Context Named Entity Recognition*（README 标注为 AAAI 2025）。
- 项目主要目标：利用训练集标注样本自动总结实体识别规则（guidelines/rules），再将这些规则注入大语言模型提示词中，完成命名实体识别（NER）推理。
- 当前仓库主要完成的工作：
  - 基于训练集样本抽取并归纳实体类别规则。
  - 对抽取出的规则进行一次“回代验证”，筛出更可靠的规则。
  - 将汇总后的规则用于测试集推理。
  - 对推理结果进行离线评测，计算 Precision / Recall / F1。
- 当前仓库未体现的能力：
  - 未看到常规意义上的参数训练、微调、优化器配置、checkpoint 管理等逻辑。
  - 未看到 Web/API 服务化接口。
  - 未看到独立的实验配置系统（如 YAML/JSON 配置树）。

# 目录结构总览

## 一级目录

- `datasets/`
  - 数据目录。
  - 当前仓库实际提供了 `conll2003` 数据，包含原始 CoNLL 文本格式与处理后的 JSONL 格式。
  - 也是脚本默认写入规则抽取结果、规则汇总结果、推理结果的目录。
- `EasyChatTemplating/`
  - 聊天模板与输出清洗工具模块。
  - 为 `rule_summary.py` 与 `run_withrule.py` 提供 prompt 格式转换和特殊 token 清洗能力。
- `../model/`
  - 仓库外部的本地模型目录，不在当前版本库内。
  - 当前按用户提供信息，`rule_summary.py` 与 `run_withrule.py` 约定从这里读取四个模型：
    - `bge-m3`
    - `Llama-3.1-8B-Instruct`
    - `Ministral-3-8B-Instruct-2512`
    - `Qwen2.5-7B-Instruct`

## 根目录关键文件

- `README.md`
  - 项目总说明，概述环境依赖、数据来源限制和基本运行顺序。
- `rule_summary.py`
  - 训练集规则抽取、规则验证、规则汇总的核心脚本。
- `run_withrule.py`
  - 使用汇总规则进行测试集推理的核心脚本。
- `ner_evaluate.py`
  - 对推理结果文件进行离线评测的脚本。
- `README copy.md`
  - 与 `README.md` 内容一致，疑似备份文件。
- `image.png`
  - README 中引用的框架示意图，不参与运行逻辑。

## 明显重要的二级目录

- `datasets/conll2003/`
  - 当前仓库最重要的数据子目录。
  - 包含：
    - `train.jsonl` / `dev.jsonl` / `test.jsonl`：处理后的样本数据。
    - `train.txt` / `valid.txt` / `test.txt`：原始 CoNLL 风格标注文本。
    - `labels.jsonl`：实体标签集合与编号映射。
    - `metadata`：数据集说明元信息。
  - 运行脚本后，理论上还会在该目录下追加生成：
    - `<model_name>_rules.txt`
    - `<model_name>_validrules.txt`
    - `<model_name>_summaryrules.txt`
    - `<model_name>_withrule_result_detail.txt`
- `EasyChatTemplating/__pycache__/`
  - Python 自动生成缓存目录，不是核心逻辑入口。

# 关键文件说明

## 训练相关

> 说明：本仓库没有“参数训练/微调”脚本。这里的“训练相关”指使用训练集样本抽取和归纳规则的流程。

### `rule_summary.py`

- 文件路径：`rule_summary.py`
- 主要作用：
  - 定义训练集规则抽取 prompt（`conll2003_prompt`）。
  - 调用 vLLM 对 `train.jsonl` 中带标注样本进行规则总结。
  - 将模型生成的规则写入 `<model_name>_rules.txt`。
  - 通过验证 prompt 将规则再用于识别同一样本，筛出命中的 `right_rules`，写入 `<model_name>_validrules.txt`。
  - 统计高频规则，输出汇总版 `<model_name>_summaryrules.txt`。
- 谁会调用它 / 它会调用什么：
  - 通常由开发者直接命令行运行。
  - 内部调用 `transformers.AutoTokenizer`、`vllm.LLM`、`EasyChatTemplating.util_tools` 中的模板与清洗函数。
  - 读取 `datasets/<dataset>/train.jsonl`、`labels.jsonl`，并写入多个结果文件。
- 高风险修改级别：高。
  - 该文件定义了核心实验方法的规则抽取、规则验证、汇总口径。
  - prompt、规则筛选逻辑、输出文件格式一旦变化，会直接影响后续推理和评测结果。

## 推理相关

### `run_withrule.py`

- 文件路径：`run_withrule.py`
- 主要作用：
  - 读取 `rule_summary.py` 产出的 `<model_name>_summaryrules.txt`。
  - 将汇总规则格式化为提示词中的 `Rules` 段。
  - 读取 `datasets/<dataset>/test.jsonl` 并批量调用 LLM 推理。
  - 将预测结果写入 `<model_name>_withrule_result_detail.txt`。
- 谁会调用它 / 它会调用什么：
  - 通常由开发者在完成规则汇总后直接运行。
  - 调用 `get_summary_rule()` 读取规则摘要。
  - 调用 `EasyChatTemplating.util_tools` 生成聊天模板并清洗输出。
  - 依赖 `vllm.LLM` 与 `transformers.AutoTokenizer`。
- 高风险修改级别：高。
  - 它是最终推理主入口。
  - prompt 文本、结果解析正则、输出结构一旦改变，会影响评测可用性和结果可复现性。

## 评测相关

### `ner_evaluate.py`

- 文件路径：`ner_evaluate.py`
- 主要作用：
  - 读取 `<model_name>_withrule_result_detail.txt`。
  - 按预测实体与真实标注的精确匹配统计 `correct_preds`、`total_preds`、`total_correct`。
  - 输出 Precision、Recall、F1，以及非成功解析样本数量。
- 谁会调用它 / 它会调用什么：
  - 通常由开发者在推理完成后直接运行。
  - 读取 `datasets/<dataset>/labels.jsonl` 获取合法标签集合。
  - 读取推理结果文件进行逐样本统计。
- 高风险修改级别：高。
  - 该文件定义最终评测口径。
  - 大小写归一化、`geo` 标签映射、非法预测过滤方式都会影响结果对比。

## 数据处理相关

### `datasets/conll2003/train.jsonl`

- 文件路径：`datasets/conll2003/train.jsonl`
- 主要作用：
  - 规则抽取阶段的主输入数据。
  - 每行包含 `text` 和 `entity_labels`，供 `rule_summary.py` 读取。
- 谁会调用它 / 它会调用什么：
  - 被 `rule_summary.py` 读取。
  - 不调用其他模块。
- 高风险修改级别：高。
  - 属于实验数据格式定义的一部分，字段名或标签格式变化会直接破坏流程。

### `datasets/conll2003/dev.jsonl`

- 文件路径：`datasets/conll2003/dev.jsonl`
- 主要作用：验证集/开发集数据。
- 谁会调用它 / 它会调用什么：
  - 当前仓库脚本中未直接使用。
- 高风险修改级别：中。
  - 当前主流程未使用，但可能用于后续实验扩展。
- 状态说明：当前用途在仓库脚本中未落实，属于“待确认”。

### `datasets/conll2003/test.jsonl`

- 文件路径：`datasets/conll2003/test.jsonl`
- 主要作用：
  - 推理阶段主输入数据。
- 谁会调用它 / 它会调用什么：
  - 被 `run_withrule.py` 读取。
- 高风险修改级别：高。

### `datasets/conll2003/train.txt` / `valid.txt` / `test.txt`

- 文件路径：`datasets/conll2003/*.txt`
- 主要作用：
  - 保存原始 CoNLL 风格序列标注文本。
  - 更适合人工核验数据来源与原始标注格式。
- 谁会调用它 / 它会调用什么：
  - 当前 Python 主流程未直接读取这些文件。
- 高风险修改级别：中。
- 状态说明：更像原始数据保留件，而非当前代码路径中的直接输入。

### `datasets/conll2003/labels.jsonl`

- 文件路径：`datasets/conll2003/labels.jsonl`
- 主要作用：
  - 定义实体类型及编号映射，例如 `person`、`organization`、`location`、`miscellaneous`。
  - 为规则汇总初始化类别集合，并为评测阶段提供合法标签范围。
- 谁会调用它 / 它会调用什么：
  - 被 `rule_summary.py` 和 `ner_evaluate.py` 读取。
- 高风险修改级别：高。
  - 这是标签口径定义文件，修改会影响规则汇总与评测一致性。

### `datasets/conll2003/metadata`

- 文件路径：`datasets/conll2003/metadata`
- 主要作用：
  - 记录数据集格式、访问限制、名称等元信息。
- 谁会调用它 / 它会调用什么：
  - 当前脚本未直接读取。
- 高风险修改级别：低。

## 配置文件

> 本仓库没有独立配置文件体系，参数主要硬编码在 Python 脚本中。

### `rule_summary.py` / `run_withrule.py` 中的 `model_path_dict` 与相关脚本中的 `dataset_path_dict`

- 文件路径：
  - `rule_summary.py`
  - `run_withrule.py`
- 主要作用：
  - 通过字典维护模型名到本地模型路径、数据集名到目录路径的映射。
  - 当前这两个脚本将模型路径统一映射到仓库外部兄弟目录 `../model/<model_name>`。
  - 当前显式支持的模型名为：`bge-m3`、`Llama-3.1-8B-Instruct`、`Ministral-3-8B-Instruct-2512`、`Qwen2.5-7B-Instruct`。
- 谁会调用它 / 它会调用什么：
  - 在各自 `main()` 中通过 `args.model_name`、`args.dataset_name` 查询。
- 高风险修改级别：高。
  - 是运行入口的基础配置。
  - 目前声明了 `ace04`、`ace05`、`genia`，但仓库中并未看到这些目录，修改前需要确认是否仅为论文完整版本遗留。

## 启动脚本 / shell 脚本

- 当前仓库未发现 `.sh`、`.bat`、`Makefile`、任务编排脚本。
- 实际运行方式由 `README.md` 给出：
  - 先执行 `python rule_summary.py --model_name Llama-3.1-8B-Instruct`
  - 再执行 `python run_withrule.py --model_name Llama-3.1-8B-Instruct`
  - 如需评测，再执行 `python ner_evaluate.py`

## 工具函数 / 公共模块

### `EasyChatTemplating/util_tools.py`

- 文件路径：`EasyChatTemplating/util_tools.py`
- 主要作用：
  - `convert_userprompt_transformers()`：将普通用户提示转成 Hugging Face tokenizer 的 chat template 文本。
  - `skip_special_tokens_transformers()`：对 vLLM 输出进行解码，去除特殊 token。
- 谁会调用它 / 它会调用什么：
  - 被 `rule_summary.py` 与 `run_withrule.py` 调用。
  - 依赖 `transformers.PreTrainedTokenizer`。
- 高风险修改级别：中到高。
  - 是 prompt 进入模型前和输出解析前的公共桥梁层。
  - 改动后可能同时影响规则抽取和最终推理。

### `EasyChatTemplating/examples.py`

- 文件路径：`EasyChatTemplating/examples.py`
- 主要作用：
  - 演示如何调用 `util_tools.py` 结合 vLLM 与 tokenizer 完成一次简单问答推理。
- 谁会调用它 / 它会调用什么：
  - 通常供开发者手动运行验证环境。
  - 调用 `EasyChatTemplating.util_tools`。
- 高风险修改级别：低。
  - 更偏示例脚本，不属于核心实验逻辑。

### `EasyChatTemplating/__init__.py`

- 文件路径：`EasyChatTemplating/__init__.py`
- 主要作用：当前为空文件，用于将目录声明为 Python 包。
- 谁会调用它 / 它会调用什么：
  - 被 Python 导入机制隐式使用。
- 高风险修改级别：低。

## 文档说明文件

### `README.md`

- 文件路径：`README.md`
- 主要作用：
  - 给出论文标题、环境依赖、数据说明和最简运行顺序。
- 谁会调用它 / 它会调用什么：
  - 供人阅读，不参与程序调用。
- 高风险修改级别：低。
  - 但它是新人第一入口，建议保持与代码流程同步。

### `README copy.md`

- 文件路径：`README copy.md`
- 主要作用：与 `README.md` 内容一致，疑似副本。
- 谁会调用它 / 它会调用什么：
  - 未见任何脚本引用。
- 高风险修改级别：低。
- 状态说明：是否保留、是否冗余，待确认。

# 核心工作流

## 1. 数据预处理与数据组织流程

- 当前仓库没有提供原始文本到 JSONL 的转换脚本。
- 从仓库现状看，数据预处理结果已经落盘在 `datasets/conll2003/*.jsonl`。
- 相关关键文件：
  - `datasets/conll2003/train.jsonl`
  - `datasets/conll2003/dev.jsonl`
  - `datasets/conll2003/test.jsonl`
  - `datasets/conll2003/labels.jsonl`
  - `datasets/conll2003/train.txt` / `valid.txt` / `test.txt`
- 结论：
  - 本仓库更像“实验运行仓库”，而不是“完整数据构建仓库”。
  - 原始数据如何转换为 JSONL，当前仓库内待确认。

## 2. 规则抽取流程

- 目标：从训练集标注样本中让 LLM 总结类别规则，而不是直接做参数学习。
- 关键步骤：
  - `rule_summary.py` 读取 `train.jsonl`。
  - 使用 `conll2003_prompt` 让模型针对每个带标注样本生成规则。
  - 将原始规则预测保存到 `<model_name>_rules.txt`。
- 相关关键文件：
  - `rule_summary.py`
  - `datasets/conll2003/train.jsonl`
  - `EasyChatTemplating/util_tools.py`

## 3. 规则验证与汇总流程

- 目标：过滤不可靠规则，并形成更稳定的类别规则摘要。
- 关键步骤：
  - `rule_summary.py` 读取 `<model_name>_rules.txt`。
  - 用 `conll2003_valid_prompt` 将规则重新应用到原句。
  - 根据是否能正确回识别实体，将规则划分为 `right_rules` / `wrong_rules`。
  - 统计高频 `right_rules`，输出 `<model_name>_summaryrules.txt`。
- 相关关键文件：
  - `rule_summary.py`
  - `datasets/conll2003/labels.jsonl`
  - `<model_name>_rules.txt`（运行产物）
  - `<model_name>_validrules.txt`（运行产物）
  - `<model_name>_summaryrules.txt`（运行产物）

## 4. 推理流程

- 目标：基于汇总规则在测试集上进行 NER 推理。
- 关键步骤：
  - `run_withrule.py` 读取 `<model_name>_summaryrules.txt`。
  - 将规则拼接进 `conll2003_rule_prompt`。
  - 对 `test.jsonl` 批量推理。
  - 输出到 `<model_name>_withrule_result_detail.txt`。
- 相关关键文件：
  - `run_withrule.py`
  - `datasets/conll2003/test.jsonl`
  - `EasyChatTemplating/util_tools.py`
  - `<model_name>_summaryrules.txt`（运行产物）

## 5. 评测流程

- 目标：计算最终识别质量。
- 关键步骤：
  - `ner_evaluate.py` 读取 `<model_name>_withrule_result_detail.txt`。
  - 对成功解析的样本执行标签合法性过滤、大小写归一化和精确匹配统计。
  - 输出 P/R/F1。
- 相关关键文件：
  - `ner_evaluate.py`
  - `datasets/conll2003/labels.jsonl`
  - `<model_name>_withrule_result_detail.txt`（运行产物）

## 6. API 实验流程

- 当前仓库未发现 API 调用、在线服务、HTTP 客户端或远程推理接口逻辑。
- 模型调用方式是本地 `vllm.LLM(model=...)`。

# 重点入口

## 最值得优先阅读的文件

- `README.md`
  - 快速了解项目目标、依赖和官方推荐运行顺序。
- `rule_summary.py`
  - 理解项目方法论的第一核心文件。
  - 规则抽取、验证、汇总三段逻辑都在这里。
- `run_withrule.py`
  - 理解项目最终推理方式的核心入口。
- `ner_evaluate.py`
  - 理解结果如何被评测、哪些预测会被计入统计。
- `EasyChatTemplating/util_tools.py`
  - 理解 prompt 是如何转换为 chat template，以及输出如何被清洗。

## 新人理解项目时的推荐阅读顺序

1. `README.md`
2. `datasets/conll2003/labels.jsonl`
3. `datasets/conll2003/train.jsonl`（抽样看几行即可）
4. `rule_summary.py`
5. `run_withrule.py`
6. `ner_evaluate.py`
7. `EasyChatTemplating/util_tools.py`
8. `datasets/conll2003/train.txt` / `test.txt`（如需核对原始标注风格）

# 修改建议

## 适合经常修改的文件

- `rule_summary.py` 中的 prompt 文本
  - 适合做提示词实验、规则总结口径调整。
- `run_withrule.py` 中的推理 prompt
  - 适合做推理模板、输出格式约束实验。
- `EasyChatTemplating/util_tools.py`
  - 适合在模型模板或输出清洗方式变化时进行适配。
- `README.md`
  - 适合补充真实运行命令、依赖说明和实验说明。

## 修改前需要特别谨慎的文件

- `datasets/conll2003/labels.jsonl`
  - 会影响标签体系、规则汇总类别初始化和评测合法性判断。
- `rule_summary.py`
  - 改动会同时影响规则抽取、验证和汇总三环节。
- `run_withrule.py`
  - 改动会直接影响最终推理结果文件格式。
- `ner_evaluate.py`
  - 改动会影响实验指标口径，容易破坏结果可比性。
- `datasets/conll2003/*.jsonl`
  - 数据字段、标签拼写、大小写格式变化都可能破坏脚本。

## 不应作为核心逻辑修改入口的内容

- `EasyChatTemplating/__pycache__/`
  - 自动生成缓存，应忽略。
- `image.png`
  - 仅为文档配图。
- 运行后生成的 `*_rules.txt`、`*_validrules.txt`、`*_summaryrules.txt`、`*_withrule_result_detail.txt`
  - 更适合作为实验输出或中间产物，不适合作为核心源码修改入口。
  - 另外，这些文件在脚本中以追加模式（`"a"`）写入，重复运行可能累积旧结果，使用前应谨慎核对。

# 待确认项

- `../model`
  - 当前文档与 `rule_summary.py`、`run_withrule.py` 已按用户说明维护为仓库外兄弟目录。
  - 在本次工作区检查中，`D:\Projects\model` 尚未实际出现；若实际部署路径不同，需要同步更新脚本中的 `MODEL_ROOT` 约定。
- `dataset_path_dict` 中声明的 `ace04`、`ace05`、`genia`
  - 代码支持这些数据集名称，但仓库内未看到对应目录。
  - 可能是论文完整版本中的预留接口，也可能是未提交的数据目录。
- `datasets/conll2003/dev.jsonl`
  - 数据存在，但当前主流程未直接使用，其实验定位待确认。
- `datasets/conll2003/train.txt` / `valid.txt` / `test.txt`
  - 当前脚本不直接读取，更多像原始数据备份；是否仍作为人工核验基准待确认。
- `README copy.md`
  - 与主 README 内容完全重复，是否为历史备份或误提交文件待确认。
- `result_pattern = r'\{.*\}'`（见 `run_withrule.py`）
  - 变量已定义但当前推理结果解析实际使用的是 `label_pattern`，`result_pattern` 似乎未被使用，是否为遗留代码待确认。
- `import logging`
  - 在多个脚本中导入，但未见有效日志配置与调用，可能是预留依赖或遗留导入。
