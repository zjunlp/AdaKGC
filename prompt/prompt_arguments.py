from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PromptArguments:
    src_seq_ratio: float = field(
        default=0, metadata={"help": "src seq ratio."}
    )
    length_penalty: bool = field(
        default=True,
        metadata={"help": "length penalty."},
    )
    use_ssi: bool = field(
        default=True,
        metadata={"help": "use SSI."},
    )
    use_prompt: bool = field(
        default=True, 
        metadata={"help": "ues prompt"},
    )
    use_task: bool = field(
        default=True, 
        metadata={"help": "ues task"},
    )
    learn_weights: bool = field(
        default=True,
        metadata={"help": "learn weights"},
    )
    prompt_len: int = field(
        default=80,
        metadata={"help": "prompt len."},
    )
    prompt_dim: int = field(
        default=800,
        metadata={"help": "prompt dim."},
    )
    init_prompt: bool = field(
        default=False,
        metadata={"help": "Whether init prompt with spot asoc tokens."},
    )
    record2: str = field(
        default=None, 
        metadata={"help": "record2"},
    )
    negative_ratio: float = field(
        default=0.7, metadata={"help": "The keep rate of negative spot or asoc."}
    )
    other_ratio: float = field(
        default=0., metadata={"help": "The noise rate of null asoc."}
    )

    
    