import os
import copy
import torch
import torch.nn as nn
from transformers import PretrainedConfig, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from train_settings import Config


class MedusaConfig(PretrainedConfig):
    def __init__(
        self,
        medusa_num_heads=4,
        medusa_num_layers=1,
        base_model_name_or_path=Config.BASE_MODEL_PATH,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path

class ResBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))
    

class AttnMedusaModel(nn.Module):
    def __init__(
        self,
        base_model,
        medusa_num_heads=4,
        medusa_num_layers=1,
        base_model_name_or_path=Config.BASE_MODEL_PATH,
    ):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.config.hidden_size
        self.vocab_size = base_model.config.vocab_size
        self.medusa = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        except:
            self.tokenizer = None

        if hasattr(self.base_model.model, 'layers'):
            self.decoder_layer = copy.deepcopy(self.base_model.model.layers[-1])
        else:
            raise ValueError("Base Model does not have '.model.layers'. Check model arch.")
        
        for param in self.decoder_layer.parameters():
            param.requires_grad = True

        self.medusa_head = nn.ModuleList(
            [
                nn.Sequential(
                    *([ResBlock(self.hidden_size)] * medusa_num_layers),
                )
                for _ in range(medusa_num_heads)
            ]
        )

        self.reduction_layers = nn.ModuleList(
            [
                nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
                for _ in range(medusa_num_heads)
            ]
        )
        for layer in self.reduction_layers:
            torch.nn.init.xavier_uniform_(layer.weight)

        self.decoder_layer.to(self.base_model.dtype).to(self.base_model.device)
        self.medusa_head.to(self.base_model.dtype).to(self.base_model.device)
        self.reduction_layers.to(self.base_model.dtype).to(self.base_model.device)

    def get_tokenizer(self):
        return self.tokenizer
    
    @classmethod
    def from_pretrained(
        cls,
        medusa_head_name_or_path,
        base_model=None,
        medusa_num_heads=None,
        **kwargs,
    ):
        medusa_config = MedusaConfig.from_pretrained(medusa_head_name_or_path)
        if medusa_num_heads is not None:
            medusa_config.medusa_num_heads = medusa_num_heads
        if base_model is not None:
            medusa_config.base_model_name_or_path = base_model
            
        print(f"Loading base model: {medusa_config.base_model_name_or_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            medusa_config.base_model_name_or_path, 
            torch_dtype=Config.DTYPE,
            device_map=Config.DEVICE,
            **kwargs
        )

        model = cls(
            base_model,
            medusa_config.medusa_num_heads,
            medusa_config.medusa_num_layers,
            medusa_config.base_model_name_or_path,
        )

        spec_model_path = os.path.join(medusa_head_name_or_path, "attn_medusa_model.safetensors")
        if not os.path.exists(spec_model_path):
            spec_model_path = os.path.join(medusa_head_name_or_path, "attn_medusa_model.pt")
        if os.path.exists(spec_model_path):
            filename = spec_model_path
        else:
            filename = hf_hub_download(medusa_head_name_or_path, "attn_medusa_model.pt")
            
        print(f"Loading AttnMedusa Model from {filename}")
        
        if filename.endswith(".safetensors"):
            from safetensors.torch import load_file
            spec_model_state_dict = load_file(filename)
        else:
            spec_model_state_dict = torch.load(filename, map_location=base_model.device)
            
        # 直接用 strict=False 恢复扁平化保存的字典，它会自动匹配 decoder_layer.xxx 等前缀
        model.load_state_dict(spec_model_state_dict, strict=False)

        return model
    

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        medusa_past_key_values=None,
        output_orig=False,
        position_ids=None,
        return_latencies=False, # profile
    ):
        import time
        latencies = {}

        # [Time 1 begin]
        if return_latencies:
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        with torch.no_grad():
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                output_hidden_states=True,
                use_cache=True,
            )

            if hasattr(outputs, "hidden_states"):
                base_hidden_states = outputs.hidden_states[-1]
            else:
                base_hidden_states = outputs[0]

        # 需要 base_logits 来提取 embed，所以必须计算 orig
        orig = self.base_model.lm_head(base_hidden_states)

        # [Time 1 end]
        if return_latencies:
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies["base_model"] = t1 - t0


        base_hidden_states = base_hidden_states.clone()

        if position_ids is None:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
            else:
                seq_length = base_hidden_states.shape[1]
                position_ids = torch.arange(seq_length, dtype=torch.long, device=base_hidden_states.device)
                position_ids = position_ids.unsqueeze(0).expand(base_hidden_states.shape[0], -1)

        layer_attention_mask = attention_mask
        
        is_flash_attn = getattr(self.config, "_attn_implementation", "") == "flash_attention_2"
        if not is_flash_attn and base_hidden_states.shape[1] > 1 and attention_mask is not None and attention_mask.dim() == 2:
            seq_len = base_hidden_states.shape[1]
            causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=base_hidden_states.device, dtype=torch.bool))
            causal_4d = causal_mask[None, None, :, :].expand(base_hidden_states.shape[0], 1, seq_len, seq_len)
            
            padding_mask = attention_mask[:, None, None, :].expand(-1, 1, seq_len, -1).bool()
            combined_mask = causal_4d & padding_mask
            
            layer_attention_mask = torch.zeros_like(combined_mask, dtype=base_hidden_states.dtype)
            layer_attention_mask.masked_fill_(~combined_mask, torch.finfo(base_hidden_states.dtype).min)

        decoder_kwargs = {}
        if hasattr(self.base_model.model, "rotary_emb"):
            position_embeddings = self.base_model.model.rotary_emb(base_hidden_states, position_ids)
            decoder_kwargs["position_embeddings"] = position_embeddings

        import inspect
        forward_signature = inspect.signature(self.decoder_layer.forward)
        if "cache_position" in forward_signature.parameters:
            decoder_kwargs["cache_position"] = position_ids[0]

        # [Time 2 begin]
        if return_latencies:
            torch.cuda.synchronize()
            t2 = time.perf_counter()

        decoder_outputs = self.decoder_layer(
            base_hidden_states,
            attention_mask=layer_attention_mask,
            position_ids=position_ids,
            past_key_value=medusa_past_key_values,
            use_cache=True,
            **decoder_kwargs 
        )

        # [Time 2 end] decoder layer
        if return_latencies:
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            latencies["decoder_layer"] = t3 - t2
        

        if isinstance(decoder_outputs, tuple):
            d_hidden_state_raw = decoder_outputs[0]
            new_medusa_past_key_values = decoder_outputs[1] if len(decoder_outputs) > 1 else None
        else:
            d_hidden_state_raw = decoder_outputs  
            new_medusa_past_key_values = None

        d_hidden_state = d_hidden_state_raw

        # [Time 3 begin]: Medusa Heads loop
        # ==========================================
        if return_latencies:
            torch.cuda.synchronize()
            t4 = time.perf_counter()

        # 逐个获取上一个 Token 的 Embedding
        medusa_logits = []
        
        # 提取模型底层的 Embedding 层及其权重
        embed_layer = self.base_model.get_input_embeddings()
        
        # 初始的 prev_logits 来自 Base Model 的输出
        prev_logits = orig 

        for i in range(self.medusa):
            # 1. 获取 embedding
            if self.training:
                # 训练时 Teacher Forcing
                # 第 t 步 head i 预测 t+i+2 的词, 因此输入需要是 ground truth 中的 t+i+1 的词
                shift_amount = i + 1

                # 左移 input_ids, 在末尾补 0 保持序列长度不变 (Trainer 自动切片末尾垃圾数据)
                shifted_input_ids = torch.cat([
                    input_ids[:, shift_amount:],
                    torch.zeros((input_ids.size(0), shift_amount), dtype=input_ids.dtype, device=input_ids.device)
                ], dim=1)

                prev_embed = embed_layer(shifted_input_ids)
            else:
                # 推理时自回归
                # 使用 argmax 拿到离散 Token ID 直接查表
                token_ids = torch.argmax(prev_logits, dim=-1)
                prev_embed = embed_layer(token_ids)
                # print("Warning!")
            
            
            # 2. 拼接：上一个词的 embedding + 当前步的 d_hidden_state
            concat_state = torch.cat([prev_embed, d_hidden_state], dim=-1)
            
            # 3. 降维
            reduced_state = self.reduction_layers[i](concat_state)
            
            # 4. Medusa Head 提取特征
            m_hidden_state = self.medusa_head[i](reduced_state)
            
            # 5. RMSNorm 对齐分布 (Post)
            if hasattr(self.base_model.model, "norm"):
                m_hidden_state_normed = self.base_model.model.norm(m_hidden_state)
            else:
                m_hidden_state_normed = m_hidden_state 
                
            # 6. 过 lm_head 得到本阶段的 logits
            mlogits = self.base_model.lm_head(m_hidden_state_normed)
            
            # 记录结果，将当前的 mlogits 赋值给 prev_logits 传给下一轮循环
            medusa_logits.append(mlogits)
            prev_logits = mlogits

        medusa_logits_stack = torch.stack(medusa_logits, dim=1)     # (batch_size, medusa_num_head, seq_len, vocab_size)

        # [Time 3 end]
        if return_latencies:
            torch.cuda.synchronize()
            t5 = time.perf_counter()
            latencies["medusa_heads"] = t5 - t4
    
        ret = [medusa_logits_stack]
        if output_orig:
            ret.extend([outputs, orig, new_medusa_past_key_values])
        if return_latencies:
            ret.append(latencies)
            
        return tuple(ret) if len(ret) > 1 else ret[0]

        
    