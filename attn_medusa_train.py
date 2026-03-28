import os
import json
import pathlib
from typing import Dict, Optional
from dataclasses import dataclass, field
import torch
from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from safetensors.torch import save_file
import transformers
from transformers import Trainer, TrainerCallback
from transformers.trainer_pt_utils import LabelSmoother

from attn_medusa_model import AttnMedusaModel, MedusaConfig
from train_settings import Config

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

class CustomizedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        medusa = model.medusa

        # 此时 logits 的形状应该为: (batch_size, medusa_heads, seq_len, vocab_size)
        logits = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        labels = inputs["labels"]

        loss = 0.0 * logits.sum() 
        loss_fct = CrossEntropyLoss()
        log = {}
        
        for i in range(medusa):
            medusa_logits = logits[:, i, : -(2 + i), :].contiguous()    # 避免不知情的维度变化导致切到其它维度
            medusa_labels = labels[..., 2 + i :].contiguous()
            
            medusa_logits = medusa_logits.view(-1, logits.shape[-1])
            medusa_labels = medusa_labels.view(-1)
            medusa_labels = medusa_labels.to(medusa_logits.device)

            not_ignore = medusa_labels.ne(IGNORE_TOKEN_ID)

            if not_ignore.sum() == 0:
                log[f"medusa{i}_loss"] = 0.0
                log[f"medusa{i}_top1"] = 0.0
                continue

            loss_i = loss_fct(medusa_logits, medusa_labels)

            # TODO 是否需要根据位置加权
            loss += loss_i
            
            medusa_labels = medusa_labels[not_ignore]

            for k in range(1, 2):
                _, topk = medusa_logits.topk(k, dim=-1)
                topk = topk[not_ignore]
                correct = topk.eq(medusa_labels.unsqueeze(-1)).any(-1)
                log[f"medusa{i}_top{k}"] = correct.float().mean().item()

            log[f"medusa{i}_loss"] = loss_i.item()
            
        self.log(log)
        return (loss, logits) if return_outputs else loss
    

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        model_to_save = self.model.module if hasattr(self.model, "module") else self.model        
        
        # Safetensors 只支持扁平字典。直接遍历模型中需要梯度的参数保存即可
        save_state_dict = {
            name: param for name, param in model_to_save.named_parameters() if param.requires_grad
        }

        save_file(
            save_state_dict,
            os.path.join(output_dir, "attn_medusa_model.safetensors"),
        )        
        if hasattr(model_to_save, "config"):
            model_to_save.config.save_pretrained(output_dir)
            
        print(f"\n[Trainer] 已保存 attn_medusa_model.safetensors 到: {output_dir}")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=Config.BASE_MODEL_PATH)
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Load in 4 bit."},
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load in 8 bit."},
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default="sharegpt_clean.json",
        metadata={"help": "Path to the training data."},
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = True


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    report_to: Optional[str] = "none"
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    medusa_num_heads: int = field(
        default=4,
        metadata={"help": "Number of Medusa heads."},
    )
    medusa_num_layers: int = field(
        default=1,
        metadata={"help": "Number of layers for each Medusa head."},
    )


class JsonlLogCallback(TrainerCallback):
    """
    训练日志实时写入 JSONL 文件的 Callback
    """
    def __init__(self, log_path):
        self.log_path = log_path
        # 确保目录存在
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # 加入当前的全局步数，方便对齐
            logs["global_step"] = state.global_step
            with open(self.log_path, "a") as f:
                f.write(json.dumps(logs) + "\n")


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Preprocesses conversation data and tokenizes it for model input.

    Args:
        sources: A list of conversation sources.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenization.

    Returns:
        Dict: A dictionary containing tokenized inputs, labels, and attention mask.
    """

    # Apply prompt templates
    conversations = []
    prompts = []
    # # import pdb; pdb.set_trace()
    for i, conversation in enumerate(sources):
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
        prompts.append(prompt)
        conversations.append(conversation)

    # Tokenize conversations
    encoding = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
    )
    # Set everything to be ignored, except the assistant part
    targets = torch.full_like(encoding.input_ids, IGNORE_TOKEN_ID)
    input_ids = encoding.input_ids

    # Mask targets. Only compute loss on the assistant outputs.
    for conv_index, (conversation, target, prompt) in enumerate(zip(conversations, targets, prompts)):

        search_start = 0

        for turn in conversation:
            if turn["role"] == "assistant":
                content = turn["content"]
                stripped_content = content.strip()
                
                # 如果内容为空，直接跳过
                if not stripped_content:
                    continue

                try:
                    # 从上次结束的地方开始找，并增加异常捕获
                    start = prompt.index(stripped_content, search_start)
                    stop = start + len(stripped_content)
                    
                    # 更新下次搜索的起始位置
                    search_start = stop
                    
                    indices = []
                    for tok_index, (tok_start, tok_stop) in enumerate(encoding.offset_mapping[conv_index]):
                        if tok_stop >= start and tok_start < stop:
                            indices.append(tok_index)
                    
                    if indices:
                        target[indices] = encoding.input_ids[conv_index][indices]
                        
                except ValueError:
                    # 如果找不到匹配字符串，打印警告并跳过该轮对话，防止崩溃
                    # print(f"Warning: Skipped masking for a turn in conv {conv_index} due to string mismatch.")
                    continue

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning.

    Args:
        raw_data (list): A list of raw data examples.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
    """

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = raw_data
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Lazy dataset for supervised fine-tuning.

    This dataset loads data on-the-fly when requested, which can be memory-efficient but slower.

    Args:
        raw_data (list): A list of raw data examples.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
    """

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]], self.tokenizer)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
        data_args: Data arguments.

    Returns:
        dict: A dictionary containing train and eval datasets.
    """
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    config.use_cache = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    if tokenizer.chat_template is None:
        print("Warning: chat_template not found. Injecting default Llama-2 template.")
        tokenizer.chat_template = (
            "{% for message in messages %}"
                "{% if message['role'] == 'system' %}"
                    "<<SYS>>\n{{ message['content'] }}\n<</SYS>>\n\n"
                "{% elif message['role'] == 'user' or message['role'] == 'human' %}"
                    "[INST] {{ message['content'] }} [/INST] "
                "{% elif message['role'] == 'assistant' or message['role'] == 'gpt' %}"
                    "{{ message['content'] }} </s>"
                "{% endif %}"
            "{% endfor %}"
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=Config.DTYPE,
    )

    # 注意: 不能用 model.base_model，必须完全冻结 AutoModelForCausalLM 的所有参数（包括 model 和 lm_head）
    for param in model.parameters():
        param.requires_grad = False

    attn_medusa_model = AttnMedusaModel(
        model,
        medusa_num_heads=training_args.medusa_num_heads,
        medusa_num_layers=training_args.medusa_num_layers,
        base_model_name_or_path=model_args.model_name_or_path,
    )

    medusa_config = MedusaConfig(
        medusa_num_heads=training_args.medusa_num_heads,
        medusa_num_layers=training_args.medusa_num_layers,
        base_model_name_or_path=model_args.model_name_or_path,
    )
    medusa_config.save_pretrained(training_args.output_dir)

    # 初始化 Callback 并传入 Trainer
    log_file_path = os.path.join(training_args.output_dir, "training_logs.jsonl")
    jsonl_callback = JsonlLogCallback(log_file_path)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    trainer = CustomizedTrainer(
        model=attn_medusa_model, 
        args=training_args, 
        callbacks=[jsonl_callback], 
        **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    print("训练结束.")
    model.config.use_cache = True
    
    if hasattr(attn_medusa_model, "module"):
        attn_medusa_model = attn_medusa_model.module
    
    # 使用相同的拍平逻辑提取最终状态
    save_state_dict = {
        name: param for name, param in attn_medusa_model.named_parameters() if param.requires_grad
    }

    if local_rank == 0 or local_rank == -1:
        tokenizer.encode("Test", truncation=None, padding="do_not_pad")
        tokenizer.save_pretrained(training_args.output_dir)
        
        save_file(save_state_dict, os.path.join(training_args.output_dir, "attn_medusa_model.safetensors"))
        print(f"attn_medusa_model 已成功导出至: {training_args.output_dir}")

if __name__ == "__main__":
    train()
