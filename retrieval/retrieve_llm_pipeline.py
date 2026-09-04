import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    pipeline,
)
from langchain_huggingface import HuggingFacePipeline

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
generation_config = GenerationConfig(
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
)

retrieve_llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    generation_config=generation_config,
    return_full_text=False,
)

llm = HuggingFacePipeline(
    pipeline=retrieve_llm_pipeline
)