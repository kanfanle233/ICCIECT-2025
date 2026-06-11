"""
兼容入口：调用已完成的 MNIST 条件生成模型进行可视化测试。
保留原脚本想表达的“模型测试/生成展示”用途。
"""

from p11_0_ import get_model, show_reconstruction, show_generation


def test_visualization():
    model = get_model()
    show_reconstruction(model)
    show_generation(model)


if __name__ == "__main__":
    test_visualization()
