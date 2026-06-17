from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer

from modules.visual_extractor import VisualExtractor
from modules.caption_model import CaptionModel


class R2GenLlama(CaptionModel):
    def __init__(self, args, tokenizer):
        super(R2GenLlama, self).__init__()

        self.args = args
        self.r2_tokenizer = tokenizer
        self.tokenizer = tokenizer

        self.bos_idx = args.bos_idx
        self.eos_idx = args.eos_idx
        self.pad_idx = args.pad_idx

        self.visual_extractor = VisualExtractor(args)

        self.llama_model_name = getattr(
            args, "llama_model_name", "llSourcell/medllama2_7b"
        )

        self.llama_tokenizer = AutoTokenizer.from_pretrained(
            self.llama_model_name,
            use_fast=True
        )
        if self.llama_tokenizer.pad_token is None:
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        self.llama = AutoModelForCausalLM.from_pretrained(
            self.llama_model_name,
            load_in_8bit=getattr(args, "llama_load_in_8bit", True),
            device_map="auto",
            attn_implementation="eager"
        )
        self.llama.resize_token_embeddings(len(self.llama_tokenizer))
        self.llama_dtype = next(self.llama.parameters()).dtype

        d_vis = args.d_vf
        d_llm = self.llama.config.hidden_size
        self.visual_proj = nn.Linear(d_vis, d_llm)

        self.max_seq_length = args.max_seq_length
        self.vocab_size = self.llama.config.vocab_size
        self.eos_token_id = self.llama_tokenizer.eos_token_id

        if getattr(args, "freeze_llama", True):
            for p in self.llama.parameters():
                p.requires_grad = False

    def encode_image(self, images):
        features = self.visual_extractor(images)
        if isinstance(features, tuple):
            features = features[0]
        visual_emb = self.visual_proj(features)
        visual_emb = visual_emb.to(dtype=self.llama_dtype)
        return visual_emb

    def _build_train_inputs(self, reports_text, visual_emb):
        B, L, D = visual_emb.size()

        prompt = getattr(
            self.args,
            "llama_prompt",
            "Patient chest X-ray image. Write a detailed radiology report.\n\nReport:\n",
        )
        texts = [prompt + r for r in reports_text]

        tok = self.llama_tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=getattr(self.args, "max_txt_len", 256),
            return_tensors="pt",
        ).to(visual_emb.device)

        input_ids = tok["input_ids"]
        attention_mask = tok["attention_mask"]

        text_emb = self.llama.model.embed_tokens(input_ids)
        text_emb = text_emb.to(dtype=self.llama_dtype)
        inputs_embeds = torch.cat([visual_emb, text_emb], dim=1)

        prefix_mask = torch.ones(B, L, device=attention_mask.device, dtype=attention_mask.dtype)
        attn_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        labels = input_ids.clone()
        pad_prefix = torch.full((B, L), -100, dtype=labels.dtype, device=labels.device)
        labels = torch.cat([pad_prefix, labels], dim=1)

        return inputs_embeds, attn_mask, labels

    def ids_to_text(self, report_ids):
        if isinstance(report_ids, torch.Tensor):
            ids = report_ids.cpu().numpy()
        else:
            ids = np.array(report_ids)

        if ids.shape[1] > 1:
            ids_nobos = ids[:, 1:]
        else:
            ids_nobos = ids

        texts = self.r2_tokenizer.decode_batch(ids_nobos)
        return texts

    def _train(self, images, report_ids, **kwargs):
        return self._forward(images, report_ids, **kwargs)

    def _forward(self, images, report_ids, **kwargs):
        visual_emb = self.encode_image(images)
        reports_text = self.ids_to_text(report_ids)
        inputs_embeds, attn_mask, labels = self._build_train_inputs(reports_text, visual_emb)

        out = self.llama(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            labels=labels,
        )
        return out

    def _sample(self, images, **kwargs):
        B = images.size(0)
        visual_emb = self.encode_image(images)

        prompt = getattr(
            self.args,
            "llama_prompt",
            "Patient chest X-ray image. Write a detailed radiology report.\n\nReport:\n",
        )
        dummy = [prompt for _ in range(B)]
        tok = self.llama_tokenizer(
            dummy,
            padding="longest",
            return_tensors="pt",
        ).to(visual_emb.device)

        text_emb = self.llama.model.embed_tokens(tok.input_ids)
        text_emb = text_emb.to(dtype=self.llama_dtype)
        inputs_embeds = torch.cat([visual_emb, text_emb], dim=1)

        prefix_mask = torch.ones(B, visual_emb.size(1), device=tok.attention_mask.device, dtype=tok.attention_mask.dtype)
        attn_mask = torch.cat([prefix_mask, tok.attention_mask], dim=1)

        gen_ids = self.llama.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=getattr(self.args, "max_new_tokens", 128),
            do_sample=False,
        )
        texts = self.llama_tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        return texts

    def get_visual_attention(self, images, report_ids):
        device = images.device
        B = images.size(0)
        assert B == 1

        visual_emb = self.encode_image(images)
        B, L, D = visual_emb.size()

        reports_text = self.ids_to_text(report_ids)
        report_text = reports_text[0]

        prompt = getattr(
            self.args,
            "llama_prompt",
            "Patient chest X-ray image. Write a detailed radiology report.\n\nReport:\n",
        )
        full_text = prompt + report_text

        tok = self.llama_tokenizer(
            [full_text],
            padding="longest",
            truncation=True,
            max_length=getattr(self.args, "max_txt_len", 256),
            return_tensors="pt",
        ).to(device)

        input_ids = tok["input_ids"]
        attention_mask = tok["attention_mask"]

        text_emb = self.llama.model.embed_tokens(input_ids)
        text_emb = text_emb.to(dtype=self.llama_dtype)
        inputs_embeds = torch.cat([visual_emb, text_emb], dim=1)

        prefix_mask = torch.ones(B, L, device=device, dtype=attention_mask.dtype)
        attn_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        outputs = self.llama(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            output_attentions=True,
            use_cache=False
        )
        attns = outputs.attentions

        last_layer_attn = attns[-1][0]
        mean_attn = last_layer_attn.mean(0)

        S_total = mean_attn.size(0)
        S_text = input_ids.size(1)
        assert S_total == L + S_text

        text_to_vis = mean_attn[L:, :L]

        text_to_vis = text_to_vis / (text_to_vis.sum(dim=-1, keepdim=True) + 1e-8)

        return text_to_vis.detach().cpu().numpy(), full_text