#!/usr/bin/env python3
"""
Update pipeline scripts to use new path configuration
=====================================================
"""

import re
from pathlib import Path

# Path mappings
PATH_REPLACEMENTS = {
    # Old paths -> New paths
    'PROJECT_ROOT = Path(__file__).resolve().parents[1]': 
        'import sys\nsys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))\nfrom gaokao_recommender.paths import *',
    
    'DATA_DIR = PROJECT_ROOT / "志愿填报辅助系统" / "上海高考录取数据17-23年"':
        'DATA_DIR = DATA_RAW_DIR',
    
    'OUTPUT_DIR = DATA_DIR / "用到的"':
        'OUTPUT_DIR = DATA_PROCESSED_DIR',
}

# Additional path patterns to replace
PATTERN_REPLACEMENTS = [
    # Replace specific file paths
    (r'DATA_DIR / "2023上海专业分数线\.txt"',
     'DATA_DIR / "2023上海专业分数线.txt"'),
    
    (r'DATA_DIR / "2023年考生高考成绩分布表（上海市）\.txt"',
     'DATA_DIR / "2023年考生高考成绩分布表（上海市）.txt"'),
    
    (r'OUTPUT_DIR / "fig_mbti_pie_2023\.png"',
     'FIGURE_DIR / "fig_mbti_pie_2023.png"'),
]

def update_script(script_path: Path):
    """Update a single script with new paths"""
    content = script_path.read_text(encoding='utf-8')
    
    # Apply direct replacements
    for old, new in PATH_REPLACEMENTS.items():
        content = content.replace(old, new)
    
    # Apply pattern replacements
    for old_pattern, new_pattern in PATTERN_REPLACEMENTS:
        content = re.sub(old_pattern, new_pattern, content)
    
    # Write back
    script_path.write_text(content, encoding='utf-8')
    print(f"✅ Updated: {script_path.name}")

def main():
    SCRIPTS_DIR = Path("scripts/pipeline_steps")
    
    # Update all pipeline scripts
    scripts = sorted(SCRIPTS_DIR.glob("0*.py"))
    print(f"Found {len(scripts)} scripts to update")
    
    for script in scripts:
        update_script(script)
    
    print(f"\n✅ Updated {len(scripts)} scripts")

if __name__ == "__main__":
    main()
