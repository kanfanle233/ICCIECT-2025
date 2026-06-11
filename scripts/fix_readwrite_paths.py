#!/usr/bin/env python3
"""
批量修复脚本的读写路径和 plt.show() 问题
"""

import re
from pathlib import Path

# 定义修复规则
FIXES = {
    # 001 脚本
    "scripts/pipeline_steps/001_数据读取与基础可视化.py": {
        "read_fixes": [],
        "write_fixes": [
            (r'OUTPUT_DIR / "fig_mbti_pie_2023\.png"', 'FIGURE_DIR / "fig_mbti_pie_2023.png"'),
            (r'OUT_DIR = OUTPUT_DIR', 'OUT_DIR = FIGURE_DIR'),
        ],
        "plt_fixes": [
            (r'plt\.show\(\)', 'plt.close()'),
        ],
        "add_matplotlib_agg": True,
    },

    # 003 脚本
    "scripts/pipeline_steps/003_探索分析与建模前检查.py": {
        "read_fixes": [
            (r'DATA_DIR / "subject_combo_to_mbti_clean\.csv"', 'OUTPUT_DIR / "subject_combo_to_mbti_clean.csv"'),
            (r'DATA_DIR / "2023上海专业分数线_clean\.csv"', 'OUTPUT_DIR / "2023上海专业分数线_clean.csv"'),
            (r'DATA_DIR / "2023年考生高考成绩分布表_clean\.csv"', 'OUTPUT_DIR / "2023年考生高考成绩分布表_clean.csv"'),
        ],
        "write_fixes": [
            (r'DATA_DIR / "score_line_2023\.png"', 'FIGURE_DIR / "score_line_2023.png"'),
            (r'DIR / "major_box_2023\.png"', 'FIGURE_DIR / "major_box_2023.png"'),
            (r'DATA_DIR / "combo_mbti_bar_2023\.png"', 'FIGURE_DIR / "combo_mbti_bar_2023.png"'),
        ],
        "plt_fixes": [
            (r'plt\.show\(\)', 'plt.close()'),
        ],
        "add_matplotlib_agg": True,
    },

    # 004 脚本
    "scripts/pipeline_steps/004_LSTM基线训练评估与可视化.py": {
        "read_fixes": [],
        "write_fixes": [],
        "plt_fixes": [
            (r'plt\.show\(\)', 'plt.close()'),
        ],
        "add_matplotlib_agg": True,
    },

    # 006 脚本
    "scripts/pipeline_steps/006_交叉验证与时序切分评估.py": {
        "read_fixes": [
            (r'DATA_DIR / "2023年考生高考成绩分布表_clean\.csv"', 'OUTPUT_DIR / "2023年考生高考成绩分布表_clean.csv"'),
            (r'DATA_DIR / "2023上海专业分数线_clean\.csv"', 'OUTPUT_DIR / "2023上海专业分数线_clean.csv"'),
        ],
        "write_fixes": [
            (r'DATA_DIR / "linear_regression_scatter\.png"', 'FIGURE_DIR / "linear_regression_scatter.png"'),
            (r'DATA_DIR / "linear_regression_residuals\.png"', 'FIGURE_DIR / "linear_regression_residuals.png"'),
            (r'DATA_DIR / "linear_regression_curve\.png"', 'FIGURE_DIR / "linear_regression_curve.png"'),
            (r'DIR / "time_split_scatter\.png"', 'FIGURE_DIR / "time_split_scatter.png"'),
            (r'DIR / "time_split_residuals\.png"', 'FIGURE_DIR / "time_split_residuals.png"'),
            (r'DATA_DIR / "2023上海专业分数线_with_PredictedMBTI\.csv"', 'OUTPUT_DIR / "2023上海专业分数线_with_PredictedMBTI.csv"'),
            (r'DATA_DIR / "未映射MBTI的专业列表\.csv"', 'OUTPUT_DIR / "未映射MBTI的专业列表.csv"'),
        ],
        "plt_fixes": [
            (r'plt\.show\(\)', 'plt.close()'),
        ],
        "add_matplotlib_agg": True,
    },

    # 007 脚本
    "scripts/pipeline_steps/007_SVD推荐系统_向量化优化.py": {
        "read_fixes": [
            (r'DATA_DIR / "subject_combo_to_mbti_clean\.csv"', 'OUTPUT_DIR / "subject_combo_to_mbti_clean.csv"'),
            (r'DATA_DIR / "2023上海专业分数线_with_PredictedMBTI\.csv"', 'OUTPUT_DIR / "2023上海专业分数线_with_PredictedMBTI.csv"'),
        ],
        "write_fixes": [
            (r'DATA_DIR / "combo_based_recommendations_sklearn\.csv"', 'OUTPUT_DIR / "combo_based_recommendations_sklearn.csv"'),
            (r'DATA_DIR / "combo_based_recommendations_bpr\.csv"', 'OUTPUT_DIR / "combo_based_recommendations_bpr.csv"'),
            (r'DATA_DIR / "recommendation_candidates\.csv"', 'OUTPUT_DIR / "recommendation_candidates.csv"'),
            (r'DATA_DIR / "recommendation_analysis_bpr\.png"', 'FIGURE_DIR / "recommendation_analysis_bpr.png"'),
            (r'DATA_DIR / "recommendation_analysis_sklearn\.png"', 'FIGURE_DIR / "recommendation_analysis_sklearn.png"'),
            (r'DATA_DIR / "recommendation_analysis_report_bpr\.txt"', 'PROJECT_ROOT / "reports" / "recommendation_analysis_report_bpr.txt"'),
            (r'DATA_DIR / "recommendation_analysis_report_sklearn\.txt"', 'PROJECT_ROOT / "reports" / "recommendation_analysis_report_sklearn.txt"'),
        ],
        "plt_fixes": [],
        "add_matplotlib_agg": True,
    },

    # 008 脚本
    "scripts/pipeline_steps/008_可解释性分析_自动GPU_CPU.py": {
        "read_fixes": [
            (r'DATA_DIR / "recommendation_candidates\.csv"', 'OUTPUT_DIR / "recommendation_candidates.csv"'),
        ],
        "write_fixes": [
            (r'DATA_DIR / "pytorch_interpretability\.png"', 'FIGURE_DIR / "pytorch_interpretability.png"'),
            (r'DATA_DIR / "pytorch_interpretability_report\.txt"', 'PROJECT_ROOT / "reports" / "pytorch_interpretability_report.txt"'),
            (r'DATA_DIR / "interpretable_model\.pth"', 'MODEL_INTERPRETABILITY_DIR / "interpretable_model.pth"'),
            (r'DATA_DIR / "interpretable_model_complete\.pth"', 'MODEL_INTERPRETABILITY_DIR / "interpretable_model_complete.pth"'),
            (r'DATA_DIR / "shap_analysis\.png"', 'FIGURE_DIR / "shap_analysis.png"'),
            (r'DATA_DIR / "feature_analysis_alternative\.png"', 'FIGURE_DIR / "feature_analysis_alternative.png"'),
            (r'DATA_DIR / "shap_analysis_results\.txt"', 'PROJECT_ROOT / "reports" / "shap_analysis_results.txt"'),
        ],
        "plt_fixes": [
            (r'plt\.show\(\)', 'plt.close()'),
        ],
        "add_matplotlib_agg": True,
    },

    # 009 脚本
    "scripts/pipeline_steps/009_完整推荐流程与可视化输出.py": {
        "read_fixes": [
            (r'DATA_DIR / "subject_combo_to_mbti_clean\.csv"', 'OUTPUT_DIR / "subject_combo_to_mbti_clean.csv"'),
            (r'DATA_DIR / "2023上海专业分数线_with_PredictedMBTI\.csv"', 'OUTPUT_DIR / "2023上海专业分数线_with_PredictedMBTI.csv"'),
            (r'DATA_DIR / "combo_based_recommendations\.csv"', 'OUTPUT_DIR / "combo_based_recommendations.csv"'),
        ],
        "write_fixes": [
            (r'DATA_DIR / "combo_based_recommendations\.csv"', 'OUTPUT_DIR / "combo_based_recommendations.csv"'),
            (r'DATA_DIR / f"{{uid}}_top10_recommendations\.png"', 'FIGURE_DIR / f"{uid}_top10_recommendations.png"'),
            (r'DATA_DIR / "all_users_recommendation_summary\.png"', 'FIGURE_DIR / "all_users_recommendation_summary.png"'),
        ],
        "plt_fixes": [
            (r'plt\.show\(\)', 'plt.close()'),
        ],
        "add_matplotlib_agg": True,
    },

    # 010 脚本
    "scripts/pipeline_steps/010_多模型基准对比.py": {
        "read_fixes": [],
        "write_fixes": [
            (r'OUTPUT_DIR / "lstm_residuals_hist\.png"', 'FIGURE_DIR / "lstm_residuals_hist.png"'),
        ],
        "plt_fixes": [
            (r'plt\.show\(\)', 'plt.close()'),
        ],
        "add_matplotlib_agg": True,
    },
}


def apply_fixes(filepath: str, fixes: dict) -> bool:
    """应用修复到指定文件"""
    path = Path(filepath)
    if not path.exists():
        print(f"❌ 文件不存在: {filepath}")
        return False

    content = path.read_text(encoding='utf-8')
    original_content = content

    # 1. 添加 matplotlib.use('Agg')
    if fixes.get("add_matplotlib_agg") and "matplotlib.use('Agg')" not in content:
        # 在第一个 import matplotlib.pyplot 之前添加
        content = re.sub(
            r'(import matplotlib\.pyplot as plt)',
            r"import matplotlib\nmatplotlib.use('Agg')\n\1",
            content,
            count=1
        )

    # 2. 修复读取路径
    for pattern, replacement in fixes.get("read_fixes", []):
        content = re.sub(pattern, replacement, content)

    # 3. 修复写入路径
    for pattern, replacement in fixes.get("write_fixes", []):
        content = re.sub(pattern, replacement, content)

    # 4. 修复 plt.show()
    for pattern, replacement in fixes.get("plt_fixes", []):
        content = re.sub(pattern, replacement, content)

    # 写回文件
    if content != original_content:
        path.write_text(content, encoding='utf-8')
        print(f"✅ 已修复: {filepath}")
        return True
    else:
        print(f"⏭️  无需修改: {filepath}")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("开始批量修复脚本")
    print("=" * 60)

    fixed_count = 0
    total_count = len(FIXES)

    for filepath, fixes in FIXES.items():
        if apply_fixes(filepath, fixes):
            fixed_count += 1

    print("\n" + "=" * 60)
    print(f"修复完成: {fixed_count}/{total_count} 个文件已修改")
    print("=" * 60)


if __name__ == "__main__":
    main()
