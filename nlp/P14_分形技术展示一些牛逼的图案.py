import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def plot_mandelbrot(ax):
    """曼德勃罗集 (Mandelbrot Set)"""
    x, y = np.ogrid[-2:1:1000j, -1.5:1.5:1000j]
    c = x + 1j * y
    z = c
    div_time = np.zeros(z.shape, dtype=int)
    m = np.full(c.shape, True, dtype=bool)

    for i in range(150):
        z[m] = z[m]**2 + c[m]
        diverged = np.abs(z[m]) > 2
        div_now = diverged.copy()
        div_time[m] = np.where(div_now, i, div_time[m])
        m[m] = ~diverged
    
    # 使用magma渐变色，非常绚丽
    ax.imshow(div_time, cmap='magma', extent=[-2, 1, -1.5, 1.5])
    ax.set_title("Mandelbrot Set", color='white', fontsize=16)
    ax.axis('off')

def plot_julia(ax):
    """朱利亚集 (Julia Set)"""
    x, y = np.ogrid[-1.5:1.5:1000j, -1.5:1.5:1000j]
    z = x + 1j * y
    # 调整c的值会生成完全不同的美丽分形
    c = -0.8 + 0.156j 
    div_time = np.zeros(z.shape, dtype=int)
    m = np.full(z.shape, True, dtype=bool)

    for i in range(150):
        z[m] = z[m]**2 + c
        diverged = np.abs(z[m]) > 2
        div_now = diverged.copy()
        div_time[m] = np.where(div_now, i, div_time[m])
        m[m] = ~diverged

    # 使用inferno渐变色
    ax.imshow(div_time, cmap='inferno', extent=[-1.5, 1.5, -1.5, 1.5])
    ax.set_title("Julia Set (c=-0.8+0.156j)", color='white', fontsize=16)
    ax.axis('off')

def plot_barnsley_fern(ax):
    """巴恩斯利蕨 (Barnsley Fern) - 使用迭代函数系统(IFS)"""
    n = 100000
    x, y = np.zeros(n), np.zeros(n)
    
    for i in range(1, n):
        r = np.random.rand()
        if r < 0.01:
            x[i] = 0
            y[i] = 0.16 * y[i-1]
        elif r < 0.86:
            x[i] = 0.85 * x[i-1] + 0.04 * y[i-1]
            y[i] = -0.04 * x[i-1] + 0.85 * y[i-1] + 1.6
        elif r < 0.93:
            x[i] = 0.20 * x[i-1] - 0.26 * y[i-1]
            y[i] = 0.23 * x[i-1] + 0.22 * y[i-1] + 1.6
        else:
            x[i] = -0.15 * x[i-1] + 0.28 * y[i-1]
            y[i] = 0.26 * x[i-1] + 0.24 * y[i-1] + 0.44

    ax.scatter(x, y, s=0.1, c='springgreen', marker='.')
    ax.set_title("Barnsley Fern", color='white', fontsize=16)
    ax.axis('off')

def plot_sierpinski(ax):
    """谢尔宾斯基三角形 (Sierpinski Triangle) - 混沌游戏生成法"""
    n = 50000
    # 定义等边三角形的三个顶点
    vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    point = np.array([0.5, 0.5])
    
    x, y = np.zeros(n), np.zeros(n)
    for i in range(n):
        v = vertices[np.random.randint(0, 3)]
        point = (point + v) / 2
        x[i], y[i] = point[0], point[1]

    ax.scatter(x, y, s=0.2, c='cyan', marker='.')
    ax.set_title("Sierpinski Triangle", color='white', fontsize=16)
    ax.axis('off')

if __name__ == "__main__":
    print("正在生成分形图案（计算量较大，请稍候约几秒钟）...")
    
    # 设置黑色背景风格，更能凸显分形的绚丽
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.patch.set_facecolor('black')
    
    # 绘制四大经典分形
    plot_mandelbrot(axes[0, 0])
    plot_julia(axes[0, 1])
    plot_barnsley_fern(axes[1, 0])
    plot_sierpinski(axes[1, 1])
    
    # 添加主标题
    fig.suptitle("Beautiful Fractals Showcase", color='white', fontsize=24, y=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
    print("生成完毕！即将弹出窗口展示...")
    # 可选：直接将图案保存为图片
    # plt.savefig('Fractals_Showcase.png', dpi=300, bbox_inches='tight')
    plt.show()
