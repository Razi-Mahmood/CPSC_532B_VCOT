# Code from test_nb notebook, but in a script
import re

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

LETTERS = "ABCDE"

def load_model(model_id="Qwen/Qwen2.5-VL-7B-Instruct"):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype="auto", 
        device_map="cuda"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

def build_prompt(ex, instruction):
    """Question, then hint/context, then lettered options, then the instruction (direct, CoTs)"""
    prompt = ex["question"]
    if ex["context"]:
        prompt += f"\nHint: {ex['context']}"
    prompt += "\nOptions: " + "".join(f"{LETTERS[i]}) {c} " for i, c in enumerate(ex["choices"]))
    return prompt + "\n" + instruction

def answer(model, processor, image_path, text_prompt, max_new_tokens, generation_kwargs):
    """Ask one question (`text_prompt`) about one image (`image_path`) and return the model's reply."""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": text_prompt},
    ]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, **generation_kwargs)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

@torch.no_grad()
def answer_given_rationale(model, processor, image_path, text_prompt, rationale, n_choices):
    """Prefill the rationale (`rationale`) as the model's own words, return its probability for each option."""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": text_prompt},
    ]}]

    # rationale is a list of steps, put them together amd prepend before 'answer'
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    body = "\n".join(f"{i}. {s}" for i, s in enumerate(rationale, 1))
    text += (body + "\n\n" if body else "") + "Answer:"

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    logits = model(**inputs).logits[0, -1].float()  # scores for the next token after "Answer:"
    letter_ids = [processor.tokenizer.encode(" " + L, add_special_tokens=False)[0] for L in LETTERS[:n_choices]]
    probs = torch.softmax(logits[letter_ids], dim=-1).tolist()  # renormalised over the valid options
    return {"pred": LETTERS[probs.index(max(probs))], "probs": dict(zip(LETTERS, probs))}

def parse_answer(text):
    """Letter after 'Answer:' (e.g. 'Answer: A) yes' -> 'A')."""
    m = re.search(r"Answer:\s*\(?([A-E])\b", text)
    return m.group(1) if m else None

def split_steps(text):
    """Splits the numbered rationale before 'Answer:' in the model's outout. """
    body = text.split("Answer:")[0]
    return [s.strip() for s in re.findall(r"^\s*\d+\.\s*(.+)$", body, flags=re.M)]
