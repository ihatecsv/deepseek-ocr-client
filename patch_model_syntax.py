#!/usr/bin/env python3
"""
Patch DeepSeek OCR model to fix syntax error in line 914
Fixes: with torch.autocast(...), dtype=...) -> with torch.autocast(..., dtype=...)
"""
import os
import re
from pathlib import Path

# Path to the cached model file
model_cache_path = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules" / "deepseek-ai" / "DeepSeek-OCR"

print("=" * 60)
print("DeepSeek OCR Model Syntax Fix")
print("=" * 60)
print()

# Find the most recent version (skip __pycache__)
if model_cache_path.exists():
    versions = [d for d in model_cache_path.iterdir() if d.is_dir() and d.name != '__pycache__']
    if versions:
        latest_version = max(versions, key=lambda p: p.stat().st_mtime)
        model_file = latest_version / "modeling_deepseekocr.py"
        
        print(f"Found model file: {model_file}")
        
        if model_file.exists():
            with open(model_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            patches_applied = []
            
            # Fix 1: torch.autocast syntax error
            # WRONG: with torch.autocast(device)), dtype=torch.bfloat16):
            # RIGHT: with torch.autocast(device, dtype=torch.bfloat16):
            
            # Search for the problematic pattern
            bad_pattern = r'with torch\.autocast\((str\(self\.device\)\.split\(":"\)\[0\])\),\s*dtype=(torch\.\w+)\):'
            if re.search(bad_pattern, content):
                content = re.sub(bad_pattern, r'with torch.autocast(\1, dtype=\2):', content)
                patches_applied.append("Fixed torch.autocast syntax (comma placement)")
                print("[FIXING] Found and fixing torch.autocast syntax error...")
            
            # Fix 2: Fix dtype mismatch in masked_scatter_
            # The issue: inputs_embeds is float16, but images_in_this_batch might be float32
            # Line 648: inputs_embeds[idx].masked_scatter_(images_seq_mask[idx].unsqueeze(-1).to(self.device), images_in_this_batch)
            dtype_fix_pattern = r'inputs_embeds\[idx\]\.masked_scatter_\(images_seq_mask\[idx\]\.unsqueeze\(-1\)\.to\(self\.device\), images_in_this_batch\)'
            dtype_fix_replacement = r'inputs_embeds[idx].masked_scatter_(images_seq_mask[idx].unsqueeze(-1).to(self.device), images_in_this_batch.to(inputs_embeds.dtype))'
            
            if re.search(dtype_fix_pattern, content):
                content = re.sub(dtype_fix_pattern, dtype_fix_replacement, content)
                patches_applied.append("Fixed dtype mismatch in masked_scatter_ (line 648)")
                print("[FIXING] Found and fixing dtype mismatch in masked_scatter_...")
            
            # Fix 3: Ensure all tensor operations respect model dtype
            # Replace any remaining .cuda() calls with .to(device)
            if '.cuda()' in content and 'def ' in content:
                # Only replace in method contexts where self.device might be available
                content = re.sub(
                    r'(\s+)(\w+)\.cuda\(\)',
                    r'\1\2.to(self.device if hasattr(self, "device") else "cuda")',
                    content
                )
                patches_applied.append("Updated .cuda() calls for device compatibility")
            
            if content != original_content:
                # Backup original file
                backup_file = model_file.with_suffix('.py.syntax_backup')
                if not backup_file.exists():
                    with open(backup_file, 'w', encoding='utf-8') as f:
                        f.write(original_content)
                    print(f"[OK] Backed up original to {backup_file.name}")
                
                # Write patched file
                with open(model_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print()
                print("[SUCCESS] Model file patched!")
                for patch in patches_applied:
                    print(f"  - {patch}")
                print()
                print("The model should now load correctly.")
            else:
                print("[INFO] Model file appears to already be patched")
        else:
            print(f"[ERROR] Model file not found: {model_file}")
    else:
        print(f"[ERROR] No model versions found in {model_cache_path}")
else:
    print(f"[INFO] Cache path not found: {model_cache_path}")
    print("Model will be downloaded on first use - patch will be applied after download")

print()
print("=" * 60)

