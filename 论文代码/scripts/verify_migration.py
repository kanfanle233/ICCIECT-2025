#!/usr/bin/env python3
"""
Verify migration completeness
=============================
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gaokao_recommender.paths import *

def check_directory_structure():
    """Check if all required directories exist"""
    print("📋 检查目录结构...")
    
    required_dirs = [
        ("src/gaokao_recommender", "Python 模块"),
        ("scripts/pipeline_steps", "流水线脚本"),
        ("notebooks", "Jupyter Notebooks"),
        ("data/raw", "原始数据"),
        ("data/interim", "中间数据"),
        ("data/processed", "处理后数据"),
        ("data/private", "私有数据"),
        ("models/lstm_rank", "LSTM 排名模型"),
        ("models/lstm_rank_opt", "优化 LSTM 模型"),
        ("models/mlp_rank", "MLP 排名模型"),
        ("models/transformer_rank", "Transformer 模型"),
        ("models/interpretability", "可解释性模型"),
        ("reports/figures", "图表"),
        ("reports/paper", "论文"),
        ("reports/references", "参考资料"),
        ("docs", "文档"),
        ("archive", "归档"),
        ("metadata", "元数据"),
    ]
    
    all_ok = True
    for dir_path, description in required_dirs:
        full_path = PROJECT_ROOT / dir_path
        if full_path.exists():
            print(f"  ✅ {dir_path:30} ({description})")
        else:
            print(f"  ❌ {dir_path:30} ({description}) - 缺失")
            all_ok = False
    
    return all_ok

def check_key_files():
    """Check if all key files exist"""
    print("\n📄 检查关键文件...")
    
    key_files = [
        ("src/gaokao_recommender/paths.py", "路径配置模块"),
        ("src/gaokao_recommender/__init__.py", "Python 包初始化"),
        ("scripts/run_pipeline.py", "统一入口点"),
        ("scripts/pipeline_steps/001_数据读取与基础可视化.py", "脚本 001"),
        ("scripts/pipeline_steps/002_清洗标准化与核心表生成.py", "脚本 002"),
        ("scripts/pipeline_steps/003_探索分析与建模前检查.py", "脚本 003"),
        ("scripts/pipeline_steps/004_LSTM基线训练评估与可视化.py", "脚本 004"),
        ("scripts/pipeline_steps/005_优化版LSTM训练_自动GPU_CPU.py", "脚本 005"),
        ("scripts/pipeline_steps/006_交叉验证与时序切分评估.py", "脚本 006"),
        ("scripts/pipeline_steps/007_SVD推荐系统_向量化优化.py", "脚本 007"),
        ("scripts/pipeline_steps/008_可解释性分析_自动GPU_CPU.py", "脚本 008"),
        ("scripts/pipeline_steps/009_完整推荐流程与可视化输出.py", "脚本 009"),
        ("scripts/pipeline_steps/010_多模型基准对比.py", "脚本 010"),
        ("docs/README.md", "项目说明"),
        ("docs/data_dictionary.md", "数据字典"),
        ("docs/migration_checklist.md", "迁移清单"),
        ("metadata/file_manifest.csv", "文件清单"),
        (".gitignore", "Git 配置"),
    ]
    
    all_ok = True
    for file_path, description in key_files:
        full_path = PROJECT_ROOT / file_path
        if full_path.exists():
            print(f"  ✅ {file_path:50} ({description})")
        else:
            print(f"  ❌ {file_path:50} ({description}) - 缺失")
            all_ok = False
    
    return all_ok

def check_path_configuration():
    """Check if path configuration is working"""
    print("\n🔗 检查路径配置...")
    
    try:
        from gaokao_recommender.paths import (
            PROJECT_ROOT, DATA_RAW_DIR, DATA_PROCESSED_DIR, 
            MODEL_DIR, FIGURE_DIR
        )
        
        print(f"  ✅ PROJECT_ROOT: {PROJECT_ROOT}")
        print(f"  ✅ DATA_RAW_DIR: {DATA_RAW_DIR}")
        print(f"  ✅ DATA_PROCESSED_DIR: {DATA_PROCESSED_DIR}")
        print(f"  ✅ MODEL_DIR: {MODEL_DIR}")
        print(f"  ✅ FIGURE_DIR: {FIGURE_DIR}")
        
        return True
    except Exception as e:
        print(f"  ❌ 路径配置加载失败: {e}")
        return False

def check_script_updates():
    """Check if scripts have been updated with new paths"""
    print("\n📝 检查脚本更新...")
    
    scripts_dir = PROJECT_ROOT / "scripts" / "pipeline_steps"
    scripts = sorted(scripts_dir.glob("0*.py"))
    
    all_ok = True
    for script in scripts:
        content = script.read_text()
        if "from gaokao_recommender.paths import" in content:
            print(f"  ✅ {script.name}: 已更新路径")
        else:
            print(f"  ❌ {script.name}: 路径未更新")
            all_ok = False
    
    return all_ok

def check_data_files():
    """Check if core data files exist"""
    print("\n📊 检查数据文件...")
    
    data_files = [
        ("data/processed/2023上海专业分数线_clean.csv", 3524),
        ("data/processed/recommendation_candidates.csv", 70480),
        ("data/processed/combo_based_recommendations.csv", 200),
    ]
    
    all_ok = True
    for file_path, expected_lines in data_files:
        full_path = PROJECT_ROOT / file_path
        if full_path.exists():
            line_count = len(full_path.read_text().split('\n')) - 1  # Subtract header
            if line_count == expected_lines:
                print(f"  ✅ {file_path:50} ({line_count} 行，符合预期)")
            else:
                print(f"  ⚠️  {file_path:50} ({line_count} 行，预期 {expected_lines} 行)")
                all_ok = False
        else:
            print(f"  ❌ {file_path:50} - 文件缺失")
            all_ok = False
    
    return all_ok

def main():
    print("=" * 80)
    print("论文代码项目迁移验证")
    print("=" * 80)
    print()
    
    checks = [
        ("目录结构", check_directory_structure),
        ("关键文件", check_key_files),
        ("路径配置", check_path_configuration),
        ("脚本更新", check_script_updates),
        ("数据文件", check_data_files),
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"  ❌ {check_name} 检查失败: {e}")
            results.append((check_name, False))
    
    print("\n" + "=" * 80)
    print("验证结果汇总")
    print("=" * 80)
    
    all_passed = True
    for check_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{check_name:15} {status}")
        if not result:
            all_passed = False
    
    print()
    if all_passed:
        print("✅ 所有检查通过！迁移完成。")
        print()
        print("下一步:")
        print("  1. 运行: python scripts/run_pipeline.py --list")
        print("  2. 运行: python scripts/run_pipeline.py --stage prepare")
        print("  3. 查看: docs/README.md")
    else:
        print("❌ 部分检查失败，请查看上方详细信息。")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
