import modal
from modal import Volume, Image

# Setup - define our infrastructure with code!
app = modal.App("pricer-service")
image = Image.debian_slim().pip_install(
    "huggingface", "torch", "transformers", "bitsandbytes", "accelerate", "peft"
)

# This collects the HuggingFace secret from Modal
secrets = [modal.Secret.from_name("huggingface-secret")]

# GPU and model configuration
GPU = "T4"
BASE_MODEL = "meta-llama/Llama-3.2-3B"
PROJECT_NAME = "price"
HF_USER = "EmilIsmayilzada0707"
RUN_NAME = "2026-03-28_15.33.16"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
REVISION = "4a5b7394a5dfcece5f8c103326e6460042e733a9"
FINETUNED_MODEL = f"{HF_USER}/{PROJECT_RUN_NAME}"
CACHE_DIR = "/cache"

# 0 = cold start (free, but slow first call)
# 1 = always warm (costs money but instant responses)
MIN_CONTAINERS = 0

PREFIX = "Price is $"
QUESTION = "What does this cost to the nearest dollar?"

# Persistent volume to cache model weights between calls
hf_cache_volume = Volume.from_name("hf-hub-cache", create_if_missing=True)


@app.cls(
    image=image.env({"HF_HUB_CACHE": CACHE_DIR}),
    secrets=secrets,
    gpu=GPU,
    timeout=1800,
    min_containers=MIN_CONTAINERS,
    volumes={CACHE_DIR: hf_cache_volume},
)
class Pricer:
    @modal.enter()
    def setup(self):
        """Runs once when the container starts — loads the model into GPU memory."""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=quant_config, device_map="auto"
        )
        self.fine_tuned_model = PeftModel.from_pretrained(
            self.base_model, FINETUNED_MODEL, revision=REVISION
        )

    @modal.method()
    def price(self, description: str) -> float:
        """Takes a product description, returns predicted price."""
        import re
        import torch
        from transformers import set_seed

        set_seed(42)
        prompt = f"{QUESTION}\n\n{description}\n\n{PREFIX}"

        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.fine_tuned_model.generate(inputs, max_new_tokens=5)
        result = self.tokenizer.decode(outputs[0])
        contents = result.split("Price is $")[1]
        contents = contents.replace(",", "")
        match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
        return float(match.group()) if match else 0