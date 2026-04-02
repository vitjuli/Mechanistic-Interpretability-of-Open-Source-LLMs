#!/usr/bin/env python3
"""Fix resume branch issues in 04_train_sae.py"""

#Read the current file
with open("scripts/04_train_sae.py", "r") as f:
    content = f.read()

# Split into lines for easier manipulation
lines = content.split('\n')

# Find and fix the resume branch
new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    
    # Fix #3: Clean up torch.load line (remove inline comment)
    if 'checkpoint = torch.load(resume_from' in line:
        new_lines.append('        checkpoint = torch.load(resume_from, map_location="cpu")')
        new_lines.append('')
        new_lines.append('        # Restore RNG states FIRST (before any random operations)')
        new_lines.append('        if "rng_states" in checkpoint:')
        new_lines.append('            set_rng_states(checkpoint["rng_states"])')
        new_lines.append('            print(f"  ✓ Restored RNG states")')
        new_lines.append('')
        new_lines.append('        # Extract states for later (after SAE/trainer creation)')
        new_lines.append('        start_step = int(checkpoint.get("step", 0))')
        new_lines.append('        model_state_dict = checkpoint.get("model_state")')
        new_lines.append('        optimizer_state_dict = checkpoint.get("optimizer_state")  # Might be None for best.pt')
        
        # Skip old extraction lines and RNG lines
        i += 1
        while i < len(lines):
            if 'if "train_indices" in checkpoint' in lines[i]:
                break
            i += 1
        i -= 1  # Back up to add train_indices line
    
    # Fix #4: Add n_train/n_val recalculation in fallback
    elif 'train_data = activations[val_indices]' in line and i > 0 and 'Fallback' in lines[i-5]:
        new_lines.append(line)
        new_lines.append('')
        new_lines.append('            # Recalculate from actual data')
        new_lines.append('            n_train = train_data.shape[0]')
        new_lines.append('            n_val = val_data.shape[0]')
    
    # Skip old RNG restore block (after split)
    elif 'Restore RNG states FIRST' in line and i > 200:
        # Skip this and next 3 lines
        i += 4
        continue
    
    else:
        new_lines.append(line)
    
    i += 1

# Write back
with open("scripts/04_train_sae.py", "w") as f:
    f.write('\n'.join(new_lines))

print("✅ All resume fixes applied via Python script")
