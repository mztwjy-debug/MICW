#!/usr/bin/env python3
"""PatchTST and iTransformer classification adapters."""

from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
VENDOR_DIR = SCRIPT_DIR.parent / "vendor" / "tslib_modern_baselines"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sys.path.insert(0, str(VENDOR_DIR))
from models.PatchTST import Model as PublicPatchTST  # noqa: E402
from models.iTransformer import Model as PublicITransformer  # noqa: E402


class Adapter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, values, feature_masks, token_padding_mask, valid_token_counts):
        del token_padding_mask, valid_token_counts
        return self.model(torch.cat((values, feature_masks), dim=-1), None, None, None)


def cfg(input_dim, seq_len, out_dim):
    return SimpleNamespace(
        task_name="classification",
        seq_len=seq_len,
        pred_len=0,
        d_model=64,
        n_heads=4,
        d_ff=128,
        e_layers=2,
        dropout=0.2,
        factor=5,
        activation="gelu",
        enc_in=input_dim * 2,
        num_class=out_dim,
        embed="timeF",
        freq="h",
    )


def builder(name, input_dim, sequence_length, objective):
    out_dim = 2 if objective == "ordinal" else 3
    c = cfg(input_dim, sequence_length, out_dim)
    if name == "patchtst":
        return Adapter(PublicPatchTST(c, patch_len=4, stride=2))
    if name == "itransformer":
        return Adapter(PublicITransformer(c))
    raise ValueError(name)
