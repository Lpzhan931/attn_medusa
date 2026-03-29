"""
python benchmark_sps_qwen3.py \
    --draft-model-path /home/pzli/Project/Spec/SpS/models/qwen3-tiny-l8/ \
    --base-model-path /home/share/models/Qwen3-8B/ \
    --bench-name gsm8k --num-samples 1 \
    --gamma 4 \
    --show-first-sample

python benchmark_sps_qwen3.py \
    --draft-model-path /home/pzli/Project/Spec/SpS/models/qwen3-tiny-ep3 \
    --base-model-path /home/share/models/Qwen3-8B/ \
    --bench-name gsm8k --num-samples 10 \
    --gamma 4

"""

import torch
import time
import json
import os
from tqdm import tqdm
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

THINKING = False
DATA_DIR = "./data"       # from eagle repo
LOG_DIR = "./evaluation/logs"

def trim_kv_cache(past_key_values, keep_len):
    """
    安全裁剪 KV Cache, 兼容新版 DynamicCache 与 Tuple
    """
    if past_key_values is None: 
        return None
    
    # 1. Transformers 新版 DynamicCache (最佳做法：调用官方 crop)
    if hasattr(past_key_values, "crop"):
        past_key_values.crop(keep_len)
        return past_key_values
    
    # 2. 传统 tuple 格式 (基础模型的多层)
    if isinstance(past_key_values, tuple) and isinstance(past_key_values[0], tuple):
        new_past = []
        for layer in past_key_values:
            k = layer[0][:, :, :keep_len, :]
            v = layer[1][:, :, :keep_len, :]
            new_past.append((k, v))
        return tuple(new_past)
        
    return past_key_values


@torch.no_grad()
def benchmark_sps_generate(base_model, draft_model, tokenizer, input_ids, max_new_tokens=512, gamma=4, debug_log_file=None):
    device = base_model.device
    
    stats = {
        "draft_time": 0,      
        "verify_time": 0,     
        "draft_count": 0,
        "verify_count": 0,
        "accepted_lengths": [],
        "position_attempts": torch.zeros(gamma).to(device), 
        "position_accepts": torch.zeros(gamma).to(device),
        "new_tokens": 0
    }

    terminators = [tokenizer.eos_token_id]
    if hasattr(tokenizer, "convert_tokens_to_ids"):
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end is not None and isinstance(im_end, int):
            terminators.append(im_end)

    prompt_len = input_ids.shape[1]
    
    if debug_log_file:
        debug_log_file.write("="*60 + "\n")
        debug_log_file.write(f"🔍 [DEBUG] SpS 投机解码详细过程追踪 (Sample 0)\n")
        debug_log_file.write("="*60 + "\n")
        debug_log_file.write(f"📝 Prompt 长度: {prompt_len} tokens\n\n")

    seq = input_ids[0].tolist()
    
    # ==========================================
    # 0. 预填充 (Prefill)
    # ==========================================
    t0 = time.perf_counter()
    
    # Base Model Prefill
    b_out = base_model(input_ids.to(device), use_cache=True)
    target_kv = b_out.past_key_values
    base_token = torch.argmax(b_out.logits[0, -1, :]).item()
    
    seq.append(base_token)
    stats["new_tokens"] += 1
    stats["verify_time"] += (time.perf_counter() - t0)
    stats["verify_count"] += 1
    
    # Draft Model Prefill
    d_out = draft_model(input_ids.to(device), use_cache=True)
    draft_kv = d_out.past_key_values

    step_counter = 0

    while stats["new_tokens"] < max_new_tokens:
        if seq[-1] in terminators:
            break
        step_counter += 1
        
        # 当前 target_kv 和 draft_kv 包含的长度 (不含 base_token)
        L_before_kv = len(seq) - 1 

        # ==========================================
        # 1. Drafting 阶段 (Draft Model 自回归)
        # ==========================================
        torch.cuda.synchronize()
        t_d_start = time.perf_counter()
        
        draft_tokens = []
        curr_d_in = torch.tensor([[base_token]], device=device)
        
        for i in range(gamma):
            s_out = draft_model(curr_d_in, past_key_values=draft_kv, use_cache=True)
            draft_kv = s_out.past_key_values
            
            next_tok = torch.argmax(s_out.logits[0, -1, :]).item()
            draft_tokens.append(next_tok)
            stats["position_attempts"][i] += 1
            
            if next_tok in terminators:
                break
            curr_d_in = torch.tensor([[next_tok]], device=device)
            
        torch.cuda.synchronize()
        step_draft_time = time.perf_counter() - t_d_start
        stats["draft_time"] += step_draft_time
        stats["draft_count"] += 1
        
        actual_gamma = len(draft_tokens)

        # ==========================================
        # 2. Verification 阶段 (Base Model 并行验证)
        # ==========================================
        torch.cuda.synchronize()
        t_v_start = time.perf_counter()
        
        # 组装验证输入: [Base_Token] + [Draft_Tokens]
        verify_inputs = [base_token] + draft_tokens
        t_in = torch.tensor([verify_inputs], device=device)
        
        t_out = base_model(t_in, past_key_values=target_kv, use_cache=True)
        target_kv = t_out.past_key_values
        
        # 获取预测结果
        target_preds = torch.argmax(t_out.logits[0], dim=-1).tolist()
        
        # 贪心匹配 (Greedy Matching)
        accept_length = 0
        for i in range(actual_gamma):
            if target_preds[i] == draft_tokens[i]:
                accept_length += 1
                stats["position_accepts"][i] += 1
            else:
                break
                
        stats["accepted_lengths"].append(accept_length)
        
        # 被接受的 Token + 1个大模型纠正的 Bonus Token
        accepted_ids = draft_tokens[:accept_length]
        bonus_token = target_preds[accept_length]
        
        tokens_to_add = accepted_ids + [bonus_token]
        seq.extend(tokens_to_add)
        stats["new_tokens"] += len(tokens_to_add)

        # ==========================================
        # 3. KV Cache 裁剪与状态对齐 (核心逻辑)
        # ==========================================
        # 我们需要保留的 KV 长度为: L_before_kv + 1(base_token) + accept_length
        keep_len = L_before_kv + 1 + accept_length
        
        target_kv = trim_kv_cache(target_kv, keep_len)
        draft_kv = trim_kv_cache(draft_kv, keep_len)
        
        # 更新下一个 Step 的起点
        base_token = bonus_token

        torch.cuda.synchronize()
        step_verify_time = time.perf_counter() - t_v_start
        stats["verify_time"] += step_verify_time
        stats["verify_count"] += 1

        # ==========================================
        # 4. 打印 Debug 信息
        # ==========================================
        if debug_log_file:
            # 1. 真正被拒绝的 Token
            truly_rejected_id = []
            if accept_length < actual_gamma:
                truly_rejected_id = [draft_tokens[accept_length]]
            
            # 2. 因前者被拒而从未被评估的后续草稿 Token
            unseen_draft_ids = []
            if accept_length + 1 < actual_gamma:
                unseen_draft_ids = draft_tokens[accept_length + 1:]

            bonus_id = [bonus_token]

            # 解码为文本
            draft_str = tokenizer.decode(draft_tokens)
            accepted_str = tokenizer.decode(accepted_ids) if accepted_ids else "(None)"
            rejected_str = tokenizer.decode(truly_rejected_id) if truly_rejected_id else "(None - Draft fully matched)"
            unseen_str = tokenizer.decode(unseen_draft_ids) if unseen_draft_ids else "(None)"
            bonus_str = tokenizer.decode(bonus_id)
            merged_str = tokenizer.decode(tokens_to_add)
            
            debug_log_file.write(f"Step {step_counter} | Accept Length: {accept_length}/{actual_gamma}\n")
            debug_log_file.write(f"\tTime: Verify {step_verify_time*1000:.2f} ms | Draft {step_draft_time*1000:.2f} ms\n")
            debug_log_file.write(f"\t[Draft]   '{draft_tokens}' | {draft_str}\n")
            debug_log_file.write(f"\t[Accept]  '{accepted_ids}' | {accepted_str}\n")
            debug_log_file.write(f"\t[Reject]  '{truly_rejected_id}' | {rejected_str}\n")
            debug_log_file.write(f"\t[Bonus]   '{bonus_id}' | {bonus_str}\n")
            debug_log_file.write(f"\t[Unseen]  '{unseen_draft_ids}' | {unseen_str}\n")
            debug_log_file.write(f"\t[Final]   '{tokens_to_add}' | {merged_str}\n")
            debug_log_file.write("-" * 80 + "\n")
            debug_log_file.flush()

    if debug_log_file:
        debug_log_file.write("\n" + "="*60 + "\n")
        debug_log_file.write("📄 [DEBUG] 最终完整生成的回答:\n")
        debug_log_file.write("="*60 + "\n")
        debug_log_file.write(tokenizer.decode(seq[prompt_len:]) + "\n")
        debug_log_file.write("="*60 + "\n\n")

    return stats


def run_benchmark(args):
    print("Loading Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_path, torch_dtype=torch.bfloat16, device_map="auto")
    base_model.eval()

    print("Loading Draft Model...")
    draft_model = AutoModelForCausalLM.from_pretrained(args.draft_model_path, torch_dtype=torch.bfloat16, device_map="auto")
    draft_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)

    questions = []
    question_file = os.path.join(DATA_DIR, f"{args.bench_name}/question.jsonl")
    with open(question_file, "r") as f:
        for line in f: questions.append(json.loads(line))

    # --- 1. 预热阶段 ---
    print("🔥 Warming up...")
    dummy_input = tokenizer(["Hello, who are you?"], return_tensors="pt").input_ids.to(base_model.device)
    benchmark_sps_generate(base_model, draft_model, tokenizer, dummy_input, max_new_tokens=16, gamma=args.gamma)

    # --- 2. 纯粹的 Debug 阶段 ---
    if args.show_first_sample:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(LOG_DIR, f"sps_benchmark_{timestamp}.log")
        os.makedirs("logs", exist_ok=True)
        print(f"\n📝 正在进行单样本详细解码追踪 (耗时不计入最终评测)...")
        print(f"📂 详细过程将写入日志文件: \033[96m{log_filename}\033[0m")
        
        q0 = questions[0]
        messages = [{"role": "user", "content": q0["turns"][0]}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=THINKING)
        input_ids = tokenizer([prompt], return_tensors="pt").input_ids.to(base_model.device)
        
        with open(log_filename, "w", encoding="utf-8") as log_file:
            benchmark_sps_generate(base_model, draft_model, tokenizer, input_ids, max_new_tokens=args.max_new_tokens, gamma=args.gamma, debug_log_file=log_file)
            
        print(f"✅ 追踪完成！\n")

    # --- 3. 正式的评测阶段 ---
    all_stats = []
    print("🚀 Starting benchmark...")
    
    for q in tqdm(questions[:args.num_samples], total=args.num_samples):
        messages = [{"role": "user", "content": q["turns"][0]}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=THINKING)
        input_ids = tokenizer([prompt], return_tensors="pt").input_ids.to(base_model.device)
        
        start_w = time.perf_counter()
        stat = benchmark_sps_generate(base_model, draft_model, tokenizer, input_ids, max_new_tokens=args.max_new_tokens, gamma=args.gamma)
        stat["total_wall_time"] = time.perf_counter() - start_w
        all_stats.append(stat)

    # --- 4. 结果统计 ---
    total_steps = sum(len(s["accepted_lengths"]) for s in all_stats)
    total_accepts = sum(sum(s["accepted_lengths"]) for s in all_stats)
    avg_accept_len = total_accepts / total_steps if total_steps > 0 else 0
    
    total_tokens = sum(s["new_tokens"] for s in all_stats)
    total_time = sum(s["total_wall_time"] for s in all_stats)
    speed = total_tokens / total_time
    
    total_draft_time = sum(s["draft_time"] for s in all_stats)
    total_verify_time = sum(s["verify_time"] for s in all_stats)
    draft_ratio = total_draft_time / (total_draft_time + total_verify_time)

    # 平均每步耗时
    avg_draft_time = total_draft_time / sum(s["draft_count"] for s in all_stats)
    avg_verify_time = total_verify_time / sum(s["verify_count"] for s in all_stats)

    print(f"\n====== SpS 测试结果 ======")
    print(f"Base Model: {args.base_model_path}")
    print(f"Draft Model: {args.draft_model_path}")
    print(f"1. 解码速度: {speed:.2f} tokens/s")
    print(f"2. 平均接受长度: {avg_accept_len:.2f}")
    print(f"3. 平均延时: 投机模块 {avg_draft_time*1000:.2f}ms ({draft_ratio*100:.1f}%) | 验证模块 {avg_verify_time*1000:.2f}ms ({(1-draft_ratio)*100:.1f}%)")

    # pos_accepts = torch.stack([s["position_accepts"] for s in all_stats]).sum(dim=0)
    # pos_attempts = torch.stack([s["position_attempts"] for s in all_stats]).sum(dim=0)
    # pos_rates = (pos_accepts / torch.clamp(pos_attempts, min=1)).cpu().tolist()
    # print(f"4. 逐位置接受率: {[round(r, 4) for r in pos_rates]}")

    # 提取绝对接受次数
    pos_accepts_list = torch.stack([s["position_accepts"] for s in all_stats]).sum(dim=0).cpu().tolist()
    print("4. 逐位置接受率 (条件接受率 / Conditional Acceptance Rate):")
    # Token 1 的分母是总验证步数 (Total Steps)
    current_denominator = total_steps
    for i, accepts in enumerate(pos_accepts_list):
        if current_denominator > 0:
            rate = accepts / current_denominator
            print(f"   Token {i+1}: {rate*100:.2f}% ({int(accepts)}/{int(current_denominator)})")
        else:
            print(f"   Token {i+1}: 0.00% (0/0)")
        # 下一个位置的条件尝试分母，等于当前位置的接受分子
        current_denominator = accepts

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--draft-model-path", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--bench-name", type=str, default="mt_bench")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--gamma", type=int, default=4, help="草稿模型每次生成的 token 数量")
    parser.add_argument("--show-first-sample", action="store_true", help="打印第一条数据的投机解码详细过程")
    args = parser.parse_args()
    run_benchmark(args)
