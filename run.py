import json
from pathlib import Path
from argparse import ArgumentParser
from transformers import AutoTokenizer
from pre_process.kasualconfig import KasualConfig
from pre_process.kasual_processor import KasualProcessor
from pre_process.kasual_dataset import create_dataloader
from model.embeddings import KasualInputEmbedding
from model.kasual_attention import KasualAttention
import torch

ROOT = Path(__file__).parent
TOKENIZER_CONFIG = ROOT / "configs" 



if __name__ == "__main__":
    
    # Initialize args and configs
    args = ArgumentParser()
    args.add_argument("tokenizer")
    vars = args.parse_args()
    TOKENIZER_CONFIG_PATH =  TOKENIZER_CONFIG / "tokenizers" / vars.tokenizer
    config = KasualConfig.from_json(TOKENIZER_CONFIG_PATH)
    
    # Tokenize inputs
    tokenizer = tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, revision=config.revision,
                                          trust_remote_code=config.trust_remote_code)
    processor = KasualProcessor(tokenizer=tokenizer, pause_s=config.preprocessor.pause_s, manual_config=config)
    batch = processor([ROOT / "test_data/srt/Louis.C.K.Hilarious.2010.1080p.AMZN.WEB-DL.DD+2.0.x264-Cinefeel.eng.srt", ROOT / "test_data/txt/1.txt"])
    
    # Create dataset with inputs
    window_size = config.preprocessor.window_size
    stride = config.preprocessor.stride
    
    loader = create_dataloader(batch["input_ids"], window_size, stride, return_document_ids=True)
    
    ## embedding_layer
    torch.manual_seed(9981)
    embedding_dim = config.preprocessor.embedding_dim
    emb_layer = KasualInputEmbedding(vocab_size=len(tokenizer), emb_dim=embedding_dim)
    
    inputs, targets, document_ids = next(iter(loader))
    vectors = emb_layer(inputs)
    attention = KasualAttention(config)
    out, weights = attention(vectors, return_weight=True, document_ids=document_ids)
    
    
    
    
    
    
    

