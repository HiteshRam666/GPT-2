import torch 
import torch.nn as nn 
import numpy as np 
from .GPT_Model import GPTModel
import json 
import os

# Loading GPT Config file 
# Get the directory of the current file (Model_Weights_Loading.py)
module_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root if needed, or adjust as necessary
project_root = os.path.dirname(module_dir)
config_path = os.path.join(project_root, "GPT_Model_Configuration", "GPT_config_355M.json")

with open(config_path, 'r') as f:
    GPT_CONFIG = json.load(f)

# Model Initialize
gpt = GPTModel(GPT_CONFIG)

# Model Assigning function
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

# Loading weights into our model
def load_weights_into_gpt(model, params):
    # Embedding Layers
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    
    # Transformer Blocks (Loop over all layers)
    for b in range(len(params["blocks"])):
        # Attention Weights (Q, K, V)
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_block[b].att.W_query.weight = assign(
            gpt.trf_block[b].att.W_query.weight, q_w.T)
        gpt.trf_block[b].att.W_key.weight = assign(
            gpt.trf_block[b].att.W_key.weight, k_w.T)
        gpt.trf_block[b].att.W_value.weight = assign(
            gpt.trf_block[b].att.W_value.weight, v_w.T)
        
        # Attention Biases (Q, K, V)
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_block[b].att.W_query.bias = assign(
            gpt.trf_block[b].att.W_query.bias, q_b)
        gpt.trf_block[b].att.W_key.bias = assign(
            gpt.trf_block[b].att.W_key.bias, k_b)
        gpt.trf_block[b].att.W_value.bias = assign(
            gpt.trf_block[b].att.W_value.bias, v_b)
        
        # Attention Output Projection
        gpt.trf_block[b].att.out_proj.weight = assign(
            gpt.trf_block[b].att.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_block[b].att.out_proj.bias = assign(
            gpt.trf_block[b].att.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])
        
        # Feedforward (MLP) Layers
        gpt.trf_block[b].ffn.layers[0].weight = assign(
            gpt.trf_block[b].ffn.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_block[b].ffn.layers[0].bias = assign(
            gpt.trf_block[b].ffn.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_block[b].ffn.layers[2].weight = assign(
            gpt.trf_block[b].ffn.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_block[b].ffn.layers[2].bias = assign(
            gpt.trf_block[b].ffn.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])
        
        # LayerNorm Parameters
        gpt.trf_block[b].norm1.scale = assign(
            gpt.trf_block[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_block[b].norm1.shift = assign(
            gpt.trf_block[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_block[b].norm2.scale = assign(
            gpt.trf_block[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_block[b].norm2.shift = assign(
            gpt.trf_block[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])
    
    # Final LayerNorm and Output Head
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
