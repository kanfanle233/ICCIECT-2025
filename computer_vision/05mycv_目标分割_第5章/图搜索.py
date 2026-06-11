"""
教学示例：图搜索

- 功能：演示 图搜索与目标区域分析 中与“图搜索”相关的核心流程。
- 主要数据结构：通常使用 numpy 数组保存图像矩阵，必要时再用列表或字典保存统计结果。
- 这样设置的原因：把参数、卷积核和阈值显式写出来，便于零基础同学逐行观察修改前后的效果。
"""

import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def build_graph_from_edges(edges):
    """
    基于二值边缘图像构建NetworkX图结构。
    像素值大于0的作为节点。
    使用8邻域连接建立边。
    """
    G = nx.Graph()
    
    # 获取所有边缘像素的坐标
    y_idx, x_idx = np.where(edges > 0)
    edge_points = set(zip(y_idx, x_idx))
    
    # 添加节点
    for p in edge_points:
        G.add_node(p)
        
    # 8邻域方向
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
                  
    # 添加边与权重
    for p in edge_points:
        y, x = p
        for dy, dx in directions:
            neighbor = (y + dy, x + dx)
            if neighbor in edge_points:
                # 权重可以使用欧氏距离
                dist = np.sqrt(dy**2 + dx**2)
                G.add_edge(p, neighbor, weight=dist)
                
    return G, list(edge_points)

def main():
    """演示从二值图像构建图结构，并使用Dijkstra算法寻找最短路径的完整流程。"""
    # 1. 准备图像 (这里生成一张带有一个C形轮廓的合成测试图)
    img = np.zeros((200, 200), dtype=np.uint8)
    
    # 画一个圆形作为基础边缘
    cv2.circle(img, (100, 100), 50, 255, 10)
    # 擦除左侧一部分，使其形成一个C形状的大致连通域
    img[50:150, 40:100] = 0
    # 在右侧也擦除一部分，制造断点或复杂路径
    # img[90:110, 140:160] = 0
    
    # 2. Canny边缘检测
    edges = cv2.Canny(img, 100, 200)
    
    # 3. 构建图结构
    print("正在基于边缘像素构建图...")
    G, edge_points = build_graph_from_edges(edges)
    print(f"图构建完成，包含 {G.number_of_nodes()} 个节点，{G.number_of_edges()} 条边。")
    
    if G.number_of_nodes() == 0:
        print("未检测到边缘。")
        return
        
    # 4. 获取最大的连通分量，以确保选取的起点和终点可以连通
    components = list(nx.connected_components(G))
    largest_component = max(components, key=len)
    
    lc_nodes = list(largest_component)
    # 按y坐标排序，取顶部和底部两个距离较远的节点作为起点和终点
    lc_nodes.sort(key=lambda p: p[0])
    
    start_node = lc_nodes[0]    # 顶部节点
    end_node = lc_nodes[-1]     # 底部节点
    
    print(f"正在使用 Dijkstra 算法寻找节点 {start_node} 到 {end_node} 的最短路径...")
    
    # 5. 使用 Dijkstra 算法寻找最短路径
    try:
        path = nx.dijkstra_path(G, source=start_node, target=end_node, weight='weight')
        print(f"找到的最短路径长度（节点数）：{len(path)}")
    except nx.NetworkXNoPath:
        print("这两个节点之间没有路径连接。")
        path = []
        
    # 6. 结果可视化
    # 创建彩色图像用于展示
    vis_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # 在图像上绘制路径 (红色)
    for i in range(len(path) - 1):
        p1 = (path[i][1], path[i][0])     # (x, y) 对应 OpenCV 坐标
        p2 = (path[i+1][1], path[i+1][0])
        cv2.line(vis_img, p1, p2, (0, 0, 255), 2)
        
    # 绘制起点和终点
    if len(path) > 0:
        cv2.circle(vis_img, (start_node[1], start_node[0]), 4, (0, 255, 0), -1) # 绿色代表起点
        cv2.circle(vis_img, (end_node[1], end_node[0]), 4, (255, 0, 0), -1)   # 蓝色代表终点
    
    # 使用 matplotlib 展示
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.title("Original Image")
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
    plt.subplot(132)
    plt.title("Canny Edges")
    plt.imshow(edges, cmap='gray')
    plt.axis('off')
    
    plt.subplot(133)
    plt.title("Dijkstra Shortest Path")
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
