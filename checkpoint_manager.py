import os
from checkpoint import save_checkpoint as _save_checkpoint_local
from checkpoint import load_checkpoint as _load_checkpoint_local

_config = {
    'local_dir': './checkpoints',
    'enable_s3': False,
    's3_bucket': None,
    's3_prefix': None,
    's3_client': None
}