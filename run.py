import json
from pathlib import Path
from argparse import ArgumentParser
from transformers import AutoTokenizer
from pre_process.kasual_processor import KasualProcessor
from pre_process.kasual_dataset import KasualDataset, create_dataloader
from model.embeddings import KasualInputEmbedding
import torch

ROOT = Path(__file__).parent
TOKENIZER_CONFIG = ROOT / "configs" 


class ConfigFile:
    def __init__(self, path):
        self.path = Path(path)
        self.file = json.loads(self.path.with_suffix('.json').read_text())
    
    def get(self, key, default=None):
        return self.file.get(key, default)

def read_json_file(path):
    return json.loads(Path(path.with_suffix('.json')).read_text())

def load_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(
        config["name_or_path"],
        revision=config.get("revision"),
        trust_remote_code=config.get("trust_remote_code", False),
    )
    for role, token in config.get("special_tokens", {}).items():
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"{role} {token!r} is {len(ids)} tokens in {config['name_or_path']}, not one")
        setattr(tokenizer, role, token)
    return tokenizer


if __name__ == "__main__":
    
    # Initialize args and configs
    args = ArgumentParser()
    args.add_argument("tokenizer")
    vars = args.parse_args()
    TOKENIZER_CONFIG_PATH =  TOKENIZER_CONFIG / "tokenizers" / vars.tokenizer
    tokenizer_config = ConfigFile(TOKENIZER_CONFIG_PATH)
    tokenize_file = tokenizer_config.file
    
    # Tokenize inputs
    tokenizer = load_tokenizer(tokenize_file)
    processor = KasualProcessor(tokenizer=tokenizer, pause_s=tokenize_file.get("preprocessor", {}).get("pause_s", 3), manual_config=tokenize_file)
    batch = processor(ROOT / "test_data/srt/Louis.C.K.Hilarious.2010.1080p.AMZN.WEB-DL.DD+2.0.x264-Cinefeel.eng.srt")
    
    # Create dataset with inputs
    window_size = tokenize_file.get("preprocessor", {}).get("window_size", 0)
    stride = tokenize_file.get("preprocessor", {}).get("stride", 0)
    
    loader = create_dataloader(batch["input_ids"], window_size, stride)
    
    ## embedding_layer
    torch.manual_seed(9981)
    embedding_dim = tokenize_file.get("preprocessor", {}).get("embedding_dim", 0)
    emb_layer = KasualInputEmbedding(vocab_size=len(tokenizer), emb_dim=embedding_dim)
    
    inputs, targets = next(iter(loader))
    vectors = emb_layer(inputs)
    
    
    
    
    

