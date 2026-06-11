#!/usr/bin/env python3
"""
Batch update all pipeline scripts
==================================
"""

from pathlib import Path

# Common path patterns to replace
REPLACEMENTS = [
    # Pattern for scripts 002-010 that have similar structure
    (
        '''PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "志愿填报辅助系统" / "上海高考录取数据17-23年"''',
        '''import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from gaokao_recommender.paths import *

DATA_DIR = DATA_RAW_DIR'''
    ),
    
    # Alternative pattern
    (
        '''PROJECT_ROOT = Path(__file__).resolve().parents[1]''',
        '''import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from gaokao_recommender.paths import *'''
    ),
    
    # OUTPUT_DIR patterns
    (
        '''OUTPUT_DIR = DATA_DIR / "用到的"''',
        '''OUTPUT_DIR = DATA_PROCESSED_DIR'''
    ),
    
    (
        '''OUTPUT_DIR.mkdir(parents=True, exist_ok=True)''',
        '''OUTPUT_DIR.mkdir(parents=True, exist_ok=True)'''
    ),
]

def update_script(script_path: Path):
    """Update a single script"""
    content = script_path.read_text(encoding='utf-8')
    original = content
    
    # Apply all replacements
    for old, new in REPLACEMENTS:
        content = content.replace(old, new)
    
    # Only write if changed
    if content != original:
        script_path.write_text(content, encoding='utf-8')
        return True
    return False

def main():
    SCRIPTS_DIR = Path("scripts/pipeline_steps")
    
    # Get all scripts except 001 (already updated)
    scripts = sorted(SCRIPTS_DIR.glob("0*.py"))
    scripts = [s for s in scripts if s.name != "001_数据读取与基础可视化.py"]
    
    print(f"Updating {len(scripts)} scripts...")
    
    updated_count = 0
    for script in scripts:
        if update_script(script):
            print(f"✅ Updated: {script.name}")
            updated_count += 1
        else:
            print(f"⏭️  Skipped: {script.name} (no changes needed)")
    
    print(f"\n✅ Updated {updated_count} scripts")

if __name__ == "__main__":
    main()
