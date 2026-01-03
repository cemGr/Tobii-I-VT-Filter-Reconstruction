#!/usr/bin/env python3
"""
Script to translate German comments to English in Python files.
"""
import re
import os
from pathlib import Path

# Translation dictionary for common German programming terms
TRANSLATIONS = {
    # Common patterns
    r"# Go left": "# Go left",
    r"# Go right": "# Go right",
    r"# Candidates on the left": "# Candidates on the left",
    r"# Candidates on the right": "# Candidates on the right",
    r"# If no valid samples found": "# If no valid samples found",
    r"# No valid neighbor found": "# No valid neighbor found",
    r"# ignored, only for interface compatibility": "# ignored, only for interface compatibility",
    r"# too far away": "# too far away",
    r"# only use positive differences": "# only use positive differences",
    r"# Only consider valid values": "# Only consider valid values",
    r"# Check if ALL samples are valid": "# Check if ALL samples are valid",
    r"# All valid": "# All valid",
    r"# At least one invalid sample": "# At least one invalid sample",
    r"# Keep original value": "# Keep original value",
    r"# Collect valid samples": "# Collect valid samples",
    r"# If not enough valid samples": "# If not enough valid samples",
    r"# Calculate median": "# Calculate median",
    r"# Calculate mean": "# Calculate mean",
    r"# Not enough valid samples": "# Not enough valid samples",
    r"# GT-based": "# GT-based",
    r"# GT no longer needed": "# GT no longer needed",
    r"# GT check skipped": "# GT check skipped",
    r"# Test different window sizes": "# Test different window sizes",
    r"# Fixed threshold for fair comparison": "# Fixed threshold for fair comparison",
    r"# No smoothing": "# No smoothing",
    r"# Sort by": "# Sort by",
    r"# Bei (\d+) Hz": r"# At \1 Hz",
    r"# Uncomment to run additional tests": "# Uncomment to run additional tests",
    r"# For traditional metrics": "# For traditional metrics",
    r"# Base agreement": "# Base agreement",
    r"# True Positives / False Negatives",
    r"# Full confusion matrix": "# Full confusion matrix",
    r"# Sample-based data": "# Sample-based data",
    r"# Find saccade blocks": "# Find saccade blocks",
    r"# For Ray3D we need": "# For Ray3D we need",
    r"# Select strategy": "# Select strategy",
    r"# Calculate angle": "# Calculate angle",
    r"# Select strategy": "# Select rounding strategy",
    r"# New column for": "# New column for",
}

def translate_file(filepath: Path) -> int:
    """Translate German comments in a single file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    changes = 0
    
    for pattern, replacement in TRANSLATIONS.items():
        new_content, n = re.subn(pattern, replacement, content, flags=re.IGNORECASE)
        if n > 0:
            content = new_content
            changes += n
    
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return changes
    
    return 0

def main():
    """Translate German comments in all Python files."""
    root = Path(".")
    total_changes = 0
    files_changed = 0
    
    for pattern in ["ivt_filter/**/*.py", "*.py", "examples/**/*.py"]:
        for filepath in root.glob(pattern):
            if filepath.is_file():
                changes = translate_file(filepath)
                if changes > 0:
                    print(f"✅ {filepath}: {changes} translations")
                    total_changes += changes
                    files_changed += 1
    
    print(f"\n📊 Total: {total_changes} translations in {files_changed} files")

if __name__ == "__main__":
    main()
