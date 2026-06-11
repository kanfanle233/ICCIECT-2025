#!/usr/bin/env python3
"""
Unified Pipeline Entry Point
============================

Run all pipeline stages or specific stages.

Usage:
    python scripts/run_pipeline.py --stage prepare
    python scripts/run_pipeline.py --stage all
    python scripts/run_pipeline.py --stage eda
    python scripts/run_pipeline.py --stage train
    python scripts/run_pipeline.py --stage recommend
    python scripts/run_pipeline.py --stage explain
    python scripts/run_pipeline.py --stage benchmark
"""

import argparse
import sys
import subprocess
from pathlib import Path

# Add src to path
SCRIPTS_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gaokao_recommender.paths import *

# Pipeline stages
STAGES = {
    "prepare": {
        "name": "数据准备",
        "description": "数据读取、清洗、标准化",
        "scripts": [
            "001_数据读取与基础可视化.py",
            "002_清洗标准化与核心表生成.py",
        ]
    },
    "eda": {
        "name": "探索性分析",
        "description": "数据探索与建模前检查",
        "scripts": [
            "003_探索分析与建模前检查.py",
        ]
    },
    "train": {
        "name": "模型训练",
        "description": "LSTM训练与优化",
        "scripts": [
            "004_LSTM基线训练评估与可视化.py",
            "005_优化版LSTM训练_自动GPU_CPU.py",
            "006_交叉验证与时序切分评估.py",
        ]
    },
    "recommend": {
        "name": "推荐系统",
        "description": "SVD推荐系统与向量化优化",
        "scripts": [
            "007_SVD推荐系统_向量化优化.py",
            "009_完整推荐流程与可视化输出.py",
        ]
    },
    "explain": {
        "name": "可解释性分析",
        "description": "模型可解释性分析",
        "scripts": [
            "008_可解释性分析_自动GPU_CPU.py",
        ]
    },
    "benchmark": {
        "name": "基准对比",
        "description": "多模型基准对比",
        "scripts": [
            "010_多模型基准对比.py",
        ]
    },
}

def run_stage(stage_name: str) -> bool:
    """Run a single pipeline stage"""
    if stage_name not in STAGES:
        print(f"❌ Unknown stage: {stage_name}")
        print(f"Available stages: {', '.join(STAGES.keys())}")
        return False
    
    stage = STAGES[stage_name]
    print(f"\n{'='*60}")
    print(f"Running stage: {stage['name']}")
    print(f"Description: {stage['description']}")
    print(f"{'='*60}\n")
    
    for script_name in stage["scripts"]:
        script_path = PIPELINE_STEPS_DIR / script_name
        
        if not script_path.exists():
            print(f"❌ Script not found: {script_path}")
            return False
        
        print(f"\n▶ Running: {script_name}")
        print("-" * 60)
        
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(PROJECT_ROOT),
                check=True,
                capture_output=False
            )
            print(f"✅ Completed: {script_name}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed: {script_name} (exit code {e.returncode})")
            return False
        except KeyboardInterrupt:
            print(f"\n⚠️ Interrupted by user")
            return False
    
    print(f"\n✅ Stage {stage['name']} completed successfully!")
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Run pipeline stages for 高考志愿填报推荐系统"
    )
    parser.add_argument(
        "--stage",
        choices=["prepare", "eda", "train", "recommend", "explain", "benchmark", "all"],
        required=False,
        default=None,
        help="Pipeline stage to run"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available stages"
    )

    args = parser.parse_args()

    if args.list:
        print("\nAvailable pipeline stages:")
        print("-" * 60)
        for stage_name, stage_info in STAGES.items():
            print(f"\n{stage_name}:")
            print(f"  Name: {stage_info['name']}")
            print(f"  Description: {stage_info['description']}")
            print(f"  Scripts: {', '.join(stage_info['scripts'])}")
        return

    if args.stage is None:
        parser.print_help()
        sys.exit(0)

    if args.stage == "all":
        print("\n🚀 Running all pipeline stages")
        print("=" * 60)

        for stage_name in ["prepare", "eda", "train", "recommend", "explain", "benchmark"]:
            if not run_stage(stage_name):
                print(f"\n❌ Pipeline failed at stage: {stage_name}")
                sys.exit(1)

        print("\n" + "=" * 60)
        print("✅ All pipeline stages completed successfully!")
        print("=" * 60)
    else:
        if not run_stage(args.stage):
            sys.exit(1)

if __name__ == "__main__":
    main()
