import json
from dataclasses import dataclass
from pathlib import Path
import torch.nn.functional as F 
# Plain dict that pydantic reads when it's installed (dev) and Python ignores otherwise. No import needed.
PYDANTIC = {"strict": True, "extra": "forbid"}




@dataclass(frozen=True,  kw_only=True)
class SpecialTokens:
    __pydantic_config__ = PYDANTIC
    bos_token: str
    eos_token: str
    pad_token: str
    unk_token: str
    
    def __post_init__(self):
        for role, token in self.__dict__.items():
            if not isinstance(token, str) or not token:
                raise ValueError(f"{role} must be a non-empty string, got {token!r}")



@dataclass(frozen=True,  kw_only=True)
class PreProcessor:
    __pydantic_config__ = PYDANTIC
    stride: int
    window_size: int
    
    ### With some https://www.opensubtitles.org/ files, the transcripts contains a lot of pauses and noise 
    pause_s: int
    embedding_dim: int
    
    def __post_init__(self):
        
        if min(self.stride, self.window_size, self.pause_s, self.embedding_dim) <=0:
             raise ValueError(f"values for the processor must be positive")


@dataclass(frozen=True,  kw_only=True)
class Attention:
    __pydantic_config__ = PYDANTIC
    num_heads: int
    attention_dropout: float
    implementation: str = 'eager'
    
    def __post_init__(self):
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if not (0.0 <= self.attention_dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.attention_dropout}")
        if self.implementation not in ['eager', 'sdpa']:
            raise ValueError(f"implementation must be 'eager' or 'sdpa', got {self.implementation!r}")
        
@dataclass(frozen=True,  kw_only=True)
class KasualConfig:
    """A kasual.ai config file (configs/tokenizers/<name>.json), loaded and validated.
    """
    path: Path
    tokenizer_path: str
    special_tokens: SpecialTokens
    preprocessor: PreProcessor
    attention: Attention
    revision: str | None = None
    trust_remote_code: bool = False
    
    # rules that span sections go here
    def __post_init__(self):
        if self.preprocessor.embedding_dim % self.attention.num_heads != 0:
            raise ValueError(f"embedding_dim {self.preprocessor.embedding_dim} must be divisible by num_heads {self.attention.num_heads}")
    
    @property
    def head_dims_split(self):
        return self.preprocessor.embedding_dim // self.attention.num_heads
    
   
    @classmethod
    def from_json(cls, path):
        """Load and validate a kasual.ai config file (configs/tokenizers/<name>.json)."""
        path = Path(path.with_suffix('.json'))
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        data = json.loads(path.read_text())
        return cls(path=path,**{**data,
                            "special_tokens": SpecialTokens(**data["special_tokens"]),
                            "preprocessor": PreProcessor(**data["preprocessor"]),
                            "attention": Attention(**data["attention"])})

