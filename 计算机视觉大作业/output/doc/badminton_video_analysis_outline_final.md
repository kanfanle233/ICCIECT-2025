# 基于 TrackNet 与姿态估计的羽毛球比赛视频智能分析系统论文大纲（重写终版）

> 本文件是论文写作大纲，不是正文稿。它的目标是指导后续扩展成约 16 页正文内容：每一节写什么、按什么逻辑展开、插入哪些图片和三线表、公式放在哪里、参考文献如何分布。

## 0 写作定位与参考论文风格

### 0.1 论文题目建议

**基于 TrackNet 与姿态估计的羽毛球比赛视频智能分析系统设计与实验研究**

备选题目：

1. **面向羽毛球比赛视频的球轨迹检测与球员运动分析系统设计**
2. **融合小目标跟踪与姿态估计的羽毛球视频可视分析系统**
3. **基于轨迹质量门控的羽毛球比赛视频智能分析方法与实现**

推荐使用第一个题目，原因是它既覆盖 TrackNet、姿态估计和实验研究，又不会夸大为“提出全新模型”。

### 0.2 与三篇参考论文的风格对应

| 参考论文 | 可借鉴写法 | 本文采用方式 |
|---|---|---|
| 参考1：VAC2 可视分析系统论文 | 先提出任务挑战，再给系统管线、视觉设计、交互与评估；图1直接展示系统界面和主要模块 | 本文采用“挑战 - 任务分析 - 系统管线 - 可视化设计 - 实验验证”的结构 |
| 参考2：数学优化与可视分析综述 | 使用分类体系、阶段划分和表格归纳，强调方法被放入管线中的位置 | 本文在相关工作和系统设计中使用分类表、模块表、指标表，避免只堆技术名词 |
| 参考3：CV/AI 方法论文 | 引言中明确指出已有方法的不足；方法部分给出数学形式；实验部分用定量表格支撑结论 | 本文在关键方法中加入轨迹、单应性、速度、质量评分公式，并在实验中用本地结果表格验证 |

### 0.3 论文写作边界

本文必须明确：本地项目基于上游 `badminton-pipeline-repro` 进行工程复现和本地化改进，贡献集中在工程整合、macOS/MPS 适配、轨迹后处理、质量门控、结果归档和前端可视化展示。不要写成“从零提出新模型”或“重新训练 TrackNet”。

推荐贡献表述：

1. 构建了一条从羽毛球比赛视频输入到球轨迹、球员运动统计和前端展示的完整处理流程。
2. 在 TrackNet 输出之后加入轨迹 refine 与质量门控机制，用于过滤误检、修复短缺口并判断轨迹是否适合展示。
3. 将 YOLOv8-pose、ByteTrack 和单应性变换结合，用于估计球员位置、速度和累计跑动距离。
4. 基于本地 9 段视频实验数据验证系统稳定性，并通过 D3.js 前端仪表盘展示分析结果。

## 1 正文总体结构与页数安排（扩展后约 16 页）

| 正文章节 | 建议页数 | 写作功能 |
|---|---:|---|
| 1 引言 | 2.0 页 | 提出研究背景、核心困难、本文目标和贡献边界 |
| 2 相关工作 | 2.0 页 | 按主题归纳，不按算法流水账堆砌 |
| 3 任务分析与系统总体设计 | 2.0 页 | 借鉴参考1，先定义任务，再给系统管线 |
| 4 关键方法 | 3.0 页 | 借鉴参考3，集中放公式、算法流程和质量评分 |
| 5 系统实现与可视化设计 | 2.0 页 | 结合代码文件、前端数据结构和仪表盘展示 |
| 6 实验设计与结果分析 | 3.0 页 | 用本地数据、三线表和图说明系统有效性 |
| 7 讨论与局限性 | 1.3 页 | 解释为什么质量门控必要，以及当前结果不能等同标注准确率 |
| 8 结论与展望 | 0.7 页 | 收束贡献，提出后续工作 |

摘要、关键词、参考文献、致谢不计入上述 16 页正文。若学校最终要求正文页数更少，可以压缩第 2 节和第 7 节；若要求更长，可以扩展第 4 节公式解释和第 6 节实验分析。

## 2 摘要与关键词大纲

### 摘要写作要点（约 350-450 字）

摘要只写一段，不拆成多段。按照以下顺序写：

1. **问题背景**：羽毛球目标小、飞行速度快、遮挡频繁、透视明显，普通视频中难以人工稳定分析。
2. **方法概括**：系统整合 TrackNet 球检测、轨迹后处理、YOLOv8-pose 姿态估计、ByteTrack 跟踪、单应性运动统计和前端可视化。
3. **工程实现**：以 `run_pipeline.py` 为统一入口，输出 CSV、JSON、分析视频和前端仪表盘数据。
4. **实验数据**：9 段视频、4247 帧、150.68 s；全部 Green；质量分数 88.96-95.96，平均 93.81；总体球可见率 68.92%；球员累计跑动距离 714.72 m，最高速度 9.46 m/s。
5. **结论边界**：系统适合教学演示和初步运动分析，质量分数是工程可信度指标，不等同人工标注准确率。

### 关键词

羽毛球视频分析；小目标跟踪；TrackNet；姿态估计；运动统计；可视分析；质量门控

## 3 第1章 引言

### 1.1 研究背景与意义（约 0.6 页）

写作目的：从体育视频智能分析引出羽毛球场景，不要一上来介绍代码。

展开顺序：

1. 体育视频正在从“人工观赛复盘”走向“数据驱动分析”。
2. 视频分析可以提取球轨迹、球员移动、速度、距离等结构化信息。
3. 对羽毛球训练、教学演示、课程项目和战术理解都有意义。
4. 引出问题：羽毛球视频分析比一般目标检测更难。

可用句式风格：

“与静态图像识别任务相比，体育视频分析不仅要求模型识别目标，还要求系统在连续时间维度上保持轨迹一致性和结果可解释性。”

### 1.2 羽毛球视频分析的主要挑战（约 0.6 页）

按参考1的“challenge”方式写，建议列 4 个挑战：

1. **C1 小目标与运动模糊**：羽毛球像素面积小，速度快。
2. **C2 遮挡与漏检**：球员身体、球拍和网前区域会遮挡羽毛球。
3. **C3 透视与物理坐标映射**：广播视角不是俯视图，需要单应性映射。
4. **C4 结果可信度表达**：如果没有质量门控，错误轨迹会被前端直接展示。

### 1.3 本文研究目标（约 0.4 页）

写作目的：不要说“提出 SOTA 算法”，而是说“构建可复现系统”。

目标写成 3 点：

1. 从普通羽毛球比赛视频中提取羽毛球轨迹。
2. 结合姿态估计与跟踪，计算球员速度、位置和累计跑动距离。
3. 建立质量评分和前端可视化流程，使结果能够被检查和解释。

### 1.4 本文贡献与边界（约 0.4 页）

写成条目式贡献，参考 CV 论文的 contribution 写法：

1. 设计并整理了 TrackNet、轨迹 refine、球员分析和前端导出的完整工程管线。
2. 引入基于可见率、缺失间隔、插值率和误检拒绝率的轨迹质量评分。
3. 基于 9 段本地视频给出系统实验验证，结果全部达到 Green 等级。
4. 明确项目是工程复现与本地化优化，不声称重新训练或提出全新 TrackNet 模型。

图表安排：第1章不建议放三线表。可以在引言末尾引用图1，过渡到第3章。

## 4 第2章 相关工作

本章建议写 2 页，采用“主题归纳”而不是“逐篇介绍”。参考2的综述风格，但不要写成真正综述。

### 2.1 体育小目标轨迹检测（约 0.6 页）

重点文献：MonoTrack、TrackNetV3、TrackNetV4。

写作内容：

1. TrackNet 系列说明连续帧热力图适合羽毛球、网球等高速小目标。
2. MonoTrack 说明单目羽毛球视频可结合球场、姿态和轨迹信息。
3. TrackNetV4 说明运动信息对小目标跟踪有帮助。

本文承接点：本文不重新训练小目标检测模型，而是在已有检测输出后加入工程后处理和质量门控。

### 2.2 人体姿态估计与多目标跟踪（约 0.5 页）

重点文献：YOLO-Pose、ViTPose、RTMPose、ByteTrack、OC-SORT、BoT-SORT。

写作内容：

1. 姿态估计用于定位球员关键点或脚点。
2. 多目标跟踪用于保持 near/far 两名球员身份连续。
3. ByteTrack 的优势是关联低置信度检测框，适合遮挡场景。

本文承接点：本项目使用 YOLOv8-pose 和 ByteTrack，不把姿态估计作为创新模型，而作为运动统计模块基础。

### 2.3 体育视频理解与可视分析（约 0.5 页）

重点文献：SportsMOT、ShuttleSet、ShuttleSet22、VideoMAE、MotionBERT。

写作内容：

1. SportsMOT 说明体育场景跟踪与普通 MOT 有差异。
2. ShuttleSet 说明羽毛球分析正在从轨迹走向击球和战术层面。
3. VideoMAE、MotionBERT 可作为视频/人体运动表示学习背景。

本文承接点：本文暂不做击球类别识别或战术预测，而是先完成可运行、可解释的数据提取和展示系统。

### 2.4 小结（约 0.4 页）

明确现有方法不足：

1. 单一检测模型不能保证轨迹展示可靠。
2. 许多研究关注算法指标，较少展示本地工程复现链路。
3. 羽毛球项目需要同时处理球、球员、球场映射和前端展示。

**三线表1插入位置：第2章末尾。**

表名：**表1 相关方法与本文系统关注点对比**

列设计：

| 方法类别 | 代表文献 | 主要目标 | 与本文关系 |
|---|---|---|---|
| 羽毛球/小球轨迹检测 | MonoTrack、TrackNetV3、TrackNetV4 | 球轨迹检测或重建 | 提供球检测技术背景 |
| 姿态估计 | YOLO-Pose、ViTPose、RTMPose | 人体关键点定位 | 支撑球员位置估计 |
| 多目标跟踪 | ByteTrack、OC-SORT、BoT-SORT | 身份保持与轨迹关联 | 支撑 near/far 球员连续统计 |
| 体育视频理解 | SportsMOT、ShuttleSet、MotionBERT | 数据集、动作/战术理解 | 说明后续扩展方向 |

表后介绍方式：

“如表1所示，现有研究已经分别在球轨迹检测、姿态估计和体育视频理解方面取得进展，但本文更强调这些模块在本地工程流程中的组合、质量控制和可视化交付。”

## 5 第3章 任务分析与系统总体设计

本章借鉴参考1的“Task Analysis and Pipeline”结构，是整篇论文从背景进入系统的关键章节。

### 3.1 任务分析（约 0.7 页）

定义 4 个任务：

1. **T1 球轨迹提取**：从视频帧中得到羽毛球像素坐标和可见状态。
2. **T2 轨迹可信度控制**：识别误检、跳点、静态锁定和大缺口。
3. **T3 球员运动统计**：根据姿态估计和单应性映射计算速度与距离。
4. **T4 交互式展示**：在前端同时展示视频、轨迹、检测时间线和质量面板。

写作时强调：这些任务不是彼此独立的，T2 决定 T4 是否可信，T3 依赖球场标定。

### 3.2 系统输入与输出（约 0.4 页）

输入：

1. 原始视频 `inputs/*.mp4`
2. 球场四角点 `court_points`
3. TrackNet 权重 `weights/TrackNet_best.pt`
4. YOLOv8s-pose 权重 `weights/yolov8s-pose.pt`

输出：

1. `*_ball.csv`
2. `*_players.csv`
3. `*_motion.csv`
4. `*_stats.json`
5. `*_overlay.mp4`
6. `*_final.mp4`
7. `frontend/public/data/manifest.json` 与各视频的 `analysis.json`、`quality.json`

### 3.3 系统管线（约 0.6 页）

按照流程写：

`原始视频 -> TrackNet 球检测 -> 轨迹 refine -> YOLOv8-pose + ByteTrack 球员分析 -> 视频渲染 -> 前端数据导出`

**图1插入位置：3.3 开头。**

图名：**图1 羽毛球视频智能分析系统总体管线**

图内容建议：

1. 左侧：输入视频和球场角点。
2. 中间：TrackNet、轨迹 refine、YOLOv8-pose、ByteTrack、Homography。
3. 右侧：CSV/JSON/MP4 输出和前端仪表盘。

图后介绍方式：

“如图1所示，系统将球检测、球员分析与前端展示拆分为可独立检查的模块。这样的设计既便于在某一阶段出现错误时定位问题，也使实验结果能够从 CSV、JSON 和视频三个层面相互验证。”

### 3.4 模块化设计优势（约 0.3 页）

强调：

1. 可复现：统一入口 `run_pipeline.py`。
2. 可检查：每阶段都有中间文件。
3. 可展示：前端统一加载导出数据。

**三线表2插入位置：3.4 后。**

表名：**表2 系统模块输入输出**

列设计：

| 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|
| TrackNet 球检测 | 原始视频、TrackNet 权重 | 原始球坐标 CSV | 获取候选羽毛球位置 |
| 轨迹 refine | 原始球坐标、球场角点 | 精炼球坐标、质量报告 | 清理误检与短缺口 |
| 球员分析 | 原始视频、球坐标、YOLO 权重 | players/motion/stats | 计算球员位置、速度、距离 |
| 视频渲染 | 分析数据、视频帧 | overlay/final 视频 | 生成展示视频 |
| 前端导出 | output 目录文件 | manifest、analysis、quality | 支撑仪表盘展示 |

表后介绍方式：

“表2展示了系统各模块之间的数据依赖关系。本文后续方法部分主要解释轨迹 refine、球员运动统计和质量评分三个关键环节。”

## 6 第4章 关键方法

本章是论文技术核心，应借鉴参考3的写法：先说明方法目的，再给公式，再解释符号和工程含义。

### 4.1 TrackNet 球检测（约 0.6 页）

写作内容：

1. TrackNet 输入连续帧，输出热力图。
2. 由热力图最大响应得到球位置。
3. 根据阈值判断 Visibility。
4. 说明本项目使用预训练权重，不重新训练。

公式1：热力图坐标估计

$$
\hat{p}_t=(\hat{x}_t,\hat{y}_t)=\arg\max_{(x,y)} H_t(x,y)
$$

公式2：可见性判断

$$
V_t=\mathbf{1}[\max H_t(x,y)>\tau]
$$

公式说明：放在 4.1 末尾，用来解释 `Frame, X, Y, Visibility` 字段来源。

### 4.2 轨迹后处理与短缺口修复（约 0.9 页）

写作内容：

1. ROI 过滤：依据球场区域扩展成飞行区域，剔除明显场外点。
2. static-lock 剔除：剔除长时间固定在背景处的伪球点。
3. jump rejection：剔除相邻帧位移过大的跳点。
4. Kalman 平滑：对连续可信段平滑，不长距离外推。
5. 短缺口插值：只修复长度不超过阈值的小缺口。

**图2插入位置：4.2 开头。**

图名：**图2 羽毛球轨迹 refine 处理流程**

图后介绍方式：

“如图2所示，轨迹 refine 并不替代 TrackNet 模型，而是在模型输出之后对空间合理性、时间连续性和短时缺失进行约束，从而提高轨迹展示的稳定性。”

公式3：线性插值

$$
\tilde{p}_{t+k}=(1-\alpha)p_t+\alpha p_{t+g},\quad \alpha=\frac{k}{g}
$$

公式4：Kalman 状态转移（简写）

$$
\mathbf{x}_t=F\mathbf{x}_{t-1}+\mathbf{w}_t,\quad \mathbf{z}_t=H\mathbf{x}_t+\mathbf{v}_t
$$

说明：只需解释状态向量包含位置和速度，不需要展开完整 Kalman 推导。

### 4.3 球场单应性映射与球员运动统计（约 0.7 页）

写作内容：

1. 四角点按 TL、TR、BR、BL 标定。
2. 标准羽毛球场宽 6.1 m、长 13.4 m。
3. 使用单应性矩阵将脚点映射到球场坐标。
4. 根据相邻帧物理坐标计算速度和累计距离。

公式5：单应性映射

$$
s[u,v,1]^T=H[x,y,1]^T
$$

公式6：速度

$$
v_t=\frac{\|p_t-p_{t-1}\|_2}{\Delta t}
$$

公式7：累计跑动距离

$$
D=\sum_{t=2}^{T}\|p_t-p_{t-1}\|_2
$$

图片插入位置：

**图3插入位置：4.3 或 5.1。**

图名：**图3 球场四角点标定示意图**

使用图片：`docs/images/02_court_corners.jpg`

图后介绍方式：

“图3展示了球场角点标定结果。角点顺序决定单应性矩阵的正确性，若顺序错误，后续球员位置、速度和跑动距离都会产生明显偏差。”

### 4.4 轨迹质量评分与质量门控（约 0.8 页）

写作内容：

1. 为什么需要质量评分：防止低可信轨迹直接进入前端。
2. 指标包括：可见率、最大缺失间隔、插值率、ROI 拒绝率、static-lock 拒绝率、jump 拒绝率。
3. 解释 Green/Yellow/Red 分级。
4. 强调质量分数不是人工标注准确率。

公式8：可见率

$$
R_{vis}=\frac{N_{visible}}{N_{frames}}
$$

公式9：插值率

$$
R_{interp}=\frac{N_{interp}}{N_{visible}}
$$

公式10：质量分数

$$
Q=35S_{vis}+20S_{gap}+10S_{interp}+15S_{roi}+10S_{static}+10S_{jump}
$$

公式11：质量等级

$$
G=
\begin{cases}
\text{Green}, & Q\ge75,\ R_{vis}\ge0.55,\ L_{gap}\le75\\
\text{Yellow}, & Q\ge55,\ R_{vis}\ge0.40,\ L_{gap}\le120\\
\text{Red}, & \text{otherwise}
\end{cases}
$$

**三线表3插入位置：4.4 公式之后。**

表名：**表3 轨迹质量评分指标及权重**

列设计：

| 指标 | 权重 | 度量含义 | 对应工程字段 |
|---|---:|---|---|
| S_vis | 35 | 最终可见帧比例 | final_visible / frames |
| S_gap | 20 | 最大连续缺失间隔 | max_missing_gap |
| S_interp | 10 | 插值依赖程度 | interpolated / final_visible |
| S_roi | 15 | 场外误检拒绝比例 | rejected_roi / raw_visible |
| S_static | 10 | 静态锁定拒绝比例 | rejected_static_lock / raw_visible |
| S_jump | 10 | 跳点拒绝比例 | rejected_jump / raw_visible |

表后介绍方式：

“表3给出了质量评分中各子指标的权重。可见率和最大缺失间隔权重最高，说明本文更关注轨迹在时间维度上的覆盖性和连续性。”

## 7 第5章 系统实现与可视化设计

### 5.1 工程入口与运行流程（约 0.5 页）

围绕 `run_pipeline.py` 写，不要写太多代码细节。

写作内容：

1. 统一入口检查输入视频、脚本和权重。
2. 支持 `--tracknet-device auto/cuda/mps/cpu` 和 `--pose-device auto/cuda/mps/cpu`。
3. 支持 `--refine-ball` 和 `--no-frontend-export` 等参数。
4. 默认输出到 `output/<video_id>/`。

### 5.2 关键脚本与文件组织（约 0.5 页）

重点文件：

1. `scripts/tracknet_runtime/predict.py`
2. `scripts/tools/refine_ball_csv.py`
3. `scripts/overlay/overlay_player_analytics.py`
4. `scripts/tools/export_frontend_data.py`
5. `frontend/public/data/manifest.json`

**三线表4插入位置：5.2 后。**

表名：**表4 关键脚本与功能对应关系**

列设计：

| 文件 | 功能 | 主要输出 |
|---|---|---|
| run_pipeline.py | 统一调度完整流程 | output/<video_id>/ |
| predict.py | TrackNet 推理 | 原始 ball.csv |
| refine_ball_csv.py | 轨迹修正与质量报告 | 精炼 ball.csv、report.json |
| overlay_player_analytics.py | 姿态估计与球员统计 | players.csv、motion.csv、stats.json |
| export_frontend_data.py | 前端数据导出 | manifest.json、analysis.json、quality.json |

### 5.3 前端可视化界面（约 0.7 页）

借鉴参考1的系统界面介绍方式。不要只说“有前端”，要按面板介绍：

1. 视频播放区：Original / Overlay / Final。
2. Court Spatial View：球场俯视轨迹与热力图。
3. Temporal Analytics：速度曲线、距离曲线、置信度曲线。
4. Detection Timeline：逐帧检测状态。
5. Quality Stats：质量分数和缺失段。

**图4插入位置：5.3 开头。**

图名：**图4 前端可视化仪表盘界面**

使用图片：`docs/images/frontend_dashboard.png`

图后介绍方式：

“图4展示了系统前端界面。界面左侧保留视频上下文，中间展示球场空间视图，右侧呈现时间序列统计和质量信息，使用户能够同时观察视频画面、轨迹位置和检测可靠性。”

### 5.4 图像结果展示（约 0.3 页）

**图5插入位置：5.4。**

图名：**图5 原始视频帧与 TrackNet 输出示例**

使用图片：

1. `docs/images/01_input_frame.jpg`
2. `docs/images/03_tracknet_output.jpg`

介绍方式：

“图5对比展示了原始输入帧和 TrackNet 球轨迹输出。可以看到，系统需要在较复杂背景中定位尺寸很小的羽毛球，因此后续轨迹质量控制是必要的。”

## 8 第6章 实验设计与结果分析

这是整篇论文最重要的数据支撑章节，建议写 3 页。

### 6.1 实验数据与环境（约 0.5 页）

写作内容：

1. 数据来自本地 `inputs/` 与 `output/` 目录，共 8 段视频。
2. 总帧数 3617，总时长 120.68 s。
3. 所有视频约 29.97 fps。
4. 使用已有输出文件进行统计，不虚构人工标注数据。

**三线表5插入位置：6.1 后。**

表名：**表5 实验视频基本信息**

表格数据：

| 视频 | 时长/s | FPS | 帧数 |
|---|---:|---:|---:|
| 1_00_01 | 7.50 | 30.00 | 225 |
| pro_match17_1_02_02 | 12.41 | 29.97 | 372 |
| pro_match17_1_15_13 | 17.72 | 29.97 | 531 |
| pro_match17_2_01_01 | 17.72 | 29.97 | 531 |
| pro_match17_2_08_05 | 17.58 | 29.97 | 527 |
| pro_match17_2_15_11 | 15.68 | 29.97 | 470 |
| pro_match17_2_18_11 | 16.72 | 29.97 | 501 |
| pro_match19_1_01_01 | 15.35 | 29.97 | 460 |

表后介绍方式：

“表5列出了实验视频的基本信息。测试片段覆盖不同时长，均以 29.97 fps 录制，可作为观察系统在不同场景下鲁棒性的典型样例。”

### 6.2 评价指标（约 0.4 页）

指标分两类：

1. 球轨迹质量：quality_score、quality_level、visible_rate、max_missing_gap、interp_rate。
2. 球员运动统计：total_distance_m、max_speed_mps、avg_speed_mps。

强调：没有人工标注真值，因此不写 precision/recall/mAP。

### 6.3 球轨迹质量结果（约 0.8 页）

结论先行：

1. 8 段视频全部 Green。
2. 质量分数 93.14-95.96，平均 94.54。
3. 总体球可见率 70.86%。
4. 最大缺失间隔 30-51 帧。
5. 平均插值率 7.94%。

**三线表6插入位置：6.3 开头。**

表名：**表6 羽毛球轨迹质量评估结果**

表格数据：

| 视频 | 等级 | 质量分数 | 可见率/% | 最大缺失间隔/帧 | 插值率/% |
|---|---|---:|---:|---:|---:|
| pro_match19_1_01_01 | Green | 95.96 | 69.13 | 51 | 4.72 |
| pro_match17_2_15_11 | Green | 95.24 | 80.21 | 30 | 7.96 |
| pro_match17_2_01_01 | Green | 94.96 | 71.37 | 48 | 8.71 |
| 1_00_01 | Green | 94.67 | 68.44 | 36 | 5.84 |
| pro_match17_2_18_11 | Green | 94.02 | 70.46 | 35 | 7.93 |
| pro_match17_1_15_13 | Green | 93.84 | 76.27 | 30 | 10.37 |
| pro_match17_2_08_05 | Green | 93.53 | 64.90 | 50 | 7.02 |
| pro_match17_1_02_02 | Green | 93.14 | 65.59 | 40 | 11.89 |

表后介绍方式：

“如表6所示，所有测试视频均达到 Green 等级，说明系统在当前数据范围内能够稳定生成可展示的球轨迹。各视频质量分数均高于 93 分，表明 refine 模块能够有效清理 TrackNet 原始输出中的噪声。”

**图6插入位置：6.3 文字分析之后。**

图名：**图6 球轨迹检测与可视化示例**

使用图片：`docs/images/03_tracknet_output.jpg`

图后介绍方式：

“图6展示了球轨迹检测结果。图中轨迹能够直观呈现羽毛球在回合中的飞行变化，而表6中的质量分数用于说明该轨迹是否适合被前端展示。”

### 6.4 轨迹后处理统计（约 0.5 页）

写作目的：证明 refine 不是装饰，而是真正在清理数据。

本地合计数据：

1. ROI 外误检拒绝：664 个。
2. 静态锁定拒绝：475 个。
3. 跳点拒绝：132 个。
4. 短缺口插值：293 帧。

**三线表7插入位置：6.4。**

表名：**表7 轨迹后处理统计汇总**

列设计：

| 指标 | 数值 | 含义 |
|---|---:|---|
| rejected_roi | 664 | 被判定为场地外或飞行区域外的误检点 |
| rejected_static_lock | 475 | 长时间固定在背景位置的伪检测 |
| rejected_jump | 132 | 与前后帧运动不连续的跳点 |
| interpolated | 293 | 被短缺口插值恢复的帧 |

表后介绍方式：

“表7说明，轨迹 refine 主要解决了场地外误检和静态锁定问题。若这些点直接进入前端，轨迹会出现明显跳变或固定在背景上的异常现象。”

### 6.5 球员运动分析结果（约 0.6 页）

结论先行：

1. 8 段视频双人累计跑动距离合计 591.48 m。
2. 单视频平均 73.94 m。
3. 最高速度 9.46 m/s。
4. `pro_match17_2_08_05` 视频累计跑动距离最长，达 90.12 m。

**三线表8插入位置：6.5 开头。**

表名：**表8 球员运动统计结果**

表格数据：

| 视频 | 双人累计跑动距离/m | 最高速度/(m/s) | 平均速度/(m/s) |
|---|---:|---:|---:|
| 1_00_01 | 16.06 | 8.34 | 2.23 |
| pro_match17_1_02_02 | 63.20 | 9.00 | 2.69 |
| pro_match17_1_15_13 | 81.25 | 9.19 | 2.43 |
| pro_match17_2_01_01 | 89.20 | 9.46 | 2.76 |
| pro_match17_2_08_05 | 90.12 | 9.46 | 2.68 |
| pro_match17_2_15_11 | 80.90 | 9.45 | 2.71 |
| pro_match17_2_18_11 | 89.63 | 9.43 | 2.81 |
| pro_match19_1_01_01 | 81.12 | 9.33 | 2.75 |

表后介绍方式：

“表8表明，系统不仅能够生成羽毛球轨迹，还能进一步形成球员运动负荷指标。累计跑动距离和最高速度可用于描述回合强度，是前端运动分析面板的重要数据来源。”

### 6.6 前端展示效果分析（约 0.2 页）

**图7插入位置：6.6。**

图名：**图7 前端质量面板与多维分析界面**

使用图片：`docs/images/frontend_dashboard.png` 或 `docs/images/05_panel_close.jpg`

介绍方式：

“图7展示了前端对视频、球轨迹、检测时间线和质量信息的联合呈现。与单独播放分析视频相比，仪表盘能够同时保留画面上下文和结构化指标，便于观察异常帧和运动趋势。”

## 9 第7章 讨论与局限性

### 7.1 为什么不能只依赖 TrackNet 原始输出（约 0.4 页）

论点：

1. TrackNet 能定位球，但原始结果仍包含误检。
2. 可视化系统对错误点更敏感，一个跳点会破坏整条轨迹观感。
3. 因此后处理是工程可信度的重要环节。

### 7.2 为什么质量分数不是准确率（约 0.4 页）

必须写清楚：

1. 当前没有人工标注真值。
2. quality_score 来自后处理统计，不是 precision/recall。
3. 它能说明轨迹“是否适合展示”，不能说明模型“检测得多准”。

### 7.3 单应性与球速误差（约 0.3 页）

论点：

1. 单应性把球场近似成平面。
2. 球员脚点适合投影到平面。
3. 羽毛球飞行有高度变化，因此球速投影存在误差。
4. 因此论文中对球员速度更有把握，对羽毛球物理速度要谨慎。

### 7.4 当前系统适用范围（约 0.2 页）

适合：

1. 课程展示。
2. 教学演示。
3. 初步视频分析。

不适合直接声称：

1. 专业比赛战术系统。
2. 有人工真值验证的 SOTA 算法。
3. 实时商业系统。

## 10 第8章 结论与展望

### 8.1 结论（约 0.4 页）

按 3 句话收束：

1. 本文构建了完整羽毛球视频分析系统。
2. 本地 9 段视频实验结果表明系统轨迹质量稳定，全部 Green。
3. 系统能够同时输出球轨迹和球员运动统计，具备教学演示和初步分析价值。

### 8.2 展望（约 0.3 页）

未来工作：

1. 建立人工标注真值，计算精确率、召回率和定位误差。
2. 微调 TrackNet 或引入运动注意力模型。
3. 增加击球事件识别和回合阶段识别。
4. 探索多机位或 3D 轨迹估计。
5. 优化推理速度，接近实时处理。

## 11 图表总清单

### 图片清单

| 编号 | 插入章节 | 图片/示意图 | 文件或制作方式 | 介绍重点 |
|---|---|---|---|---|
| 图1 | 3.3 | 系统总体管线 | 需绘制流程图 | 说明从视频输入到前端输出的完整流程 |
| 图2 | 4.2 | 轨迹 refine 流程 | 需绘制流程图 | 说明 ROI、static-lock、jump、Kalman、插值顺序 |
| 图3 | 4.3 | 球场四角点标注 | `docs/images/02_court_corners.jpg` | 说明单应性映射依赖角点顺序 |
| 图4 | 5.3 | 前端仪表盘 | `docs/images/frontend_dashboard.png` | 说明多面板可视分析界面 |
| 图5 | 5.4 | 原始帧与检测输出对比 | `docs/images/01_input_frame.jpg` + `docs/images/03_tracknet_output.jpg` | 说明小目标检测难点和输出效果 |
| 图6 | 6.3 | 球轨迹检测示例 | `docs/images/03_tracknet_output.jpg` | 配合质量结果表解释轨迹可靠性 |
| 图7 | 6.6 | 前端质量面板 | `docs/images/frontend_dashboard.png` 或 `docs/images/05_panel_close.jpg` | 展示质量门控和检测时间线 |

### 三线表清单

| 编号 | 插入章节 | 表名 | 作用 |
|---|---|---|---|
| 表1 | 2.4 | 相关方法与本文系统关注点对比 | 归纳文献，不堆叠算法 |
| 表2 | 3.4 | 系统模块输入输出 | 说明系统结构 |
| 表3 | 4.4 | 轨迹质量评分指标及权重 | 解释质量公式 |
| 表4 | 5.2 | 关键脚本与功能对应关系 | 连接论文和代码实现 |
| 表5 | 6.1 | 实验视频基本信息 | 说明数据规模 |
| 表6 | 6.3 | 羽毛球轨迹质量评估结果 | 支撑“全部 Green”结论 |
| 表7 | 6.4 | 轨迹后处理统计汇总 | 证明 refine 有实际作用 |
| 表8 | 6.5 | 球员运动统计结果 | 支撑运动分析能力 |

三线表格式要求：

1. 表题放表格上方。
2. 只保留顶线、表头下横线、底线。
3. 表内不要使用竖线。
4. 数字保留两位小数；百分比列不写过多小数。
5. 表后必须有解释，不要让表格孤立出现。

## 12 数学公式清单

| 编号 | 公式 | 放置位置 | 用途 |
|---|---|---|---|
| (1) | `p_hat_t = argmax H_t(x,y)` | 4.1 | 热力图转球坐标 |
| (2) | `V_t = 1[max H_t > tau]` | 4.1 | 可见性判断 |
| (3) | `p_tilde = (1-alpha)p_t + alpha p_{t+g}` | 4.2 | 短缺口线性插值 |
| (4) | `x_t = F x_{t-1} + w_t, z_t = H x_t + v_t` | 4.2 | Kalman 平滑 |
| (5) | `s[u,v,1]^T = H[x,y,1]^T` | 4.3 | 单应性映射 |
| (6) | `v_t = norm(p_t-p_{t-1}) / Δt` | 4.3 | 瞬时速度 |
| (7) | `D = Σ norm(p_t-p_{t-1})` | 4.3 | 累计跑动距离 |
| (8) | `R_vis = N_visible / N_frames` | 4.4 | 可见率 |
| (9) | `R_interp = N_interp / N_visible` | 4.4 | 插值率 |
| (10) | `Q = 35S_vis + 20S_gap + 10S_interp + 15S_roi + 10S_static + 10S_jump` | 4.4 | 质量评分 |
| (11) | Green/Yellow/Red 分段函数 | 4.4 | 质量门控 |

公式写作原则：

1. 公式只服务方法解释，不堆砌无关数学。
2. 每个公式后必须解释符号。
3. 不要把质量评分写成“准确率公式”。
4. 不要把单应性下的羽毛球速度解释成严格三维速度。

## 13 参考文献建议（正式论文约 30 篇）

以下文献主要用于正式论文，不需要把三篇“语言风格参考 PDF”全部写入参考文献，除非正文真的讨论它们的方法。本文主题是羽毛球视频分析，因此正式参考文献应优先放体育视频、小目标跟踪、姿态估计、多目标跟踪和视频理解。

[1] Liu P, Wang J H. MonoTrack: Shuttle trajectory reconstruction from monocular badminton video[C]//CVPR Workshops. 2022: 3513-3522.

[2] Sun C Y, et al. TrackNetV3: Enhancing ShuttleCock Tracking with Augmentations and Trajectory Rectification[C]//ACM Multimedia Asia. 2023.

[3] Raj A, Wang L, Gedeon T. TrackNetV4: Enhancing Fast Sports Object Tracking with Motion Attention Maps[EB/OL]. arXiv:2409.14543, 2024.

[4] Wang W Y, Huang Y C, Ik T U, Peng W C. ShuttleSet: A Human-Annotated Stroke-Level Singles Dataset for Badminton Tactical Analysis[EB/OL]. arXiv:2306.04948, 2023.

[5] Wang W Y, Huang Y C, Lee H Y, et al. Benchmarking Stroke Forecasting with Stroke-Level Badminton Dataset[C]//IJCAI. 2024.

[6] Zhang Y, Han X, Wang L, et al. Future Prediction of Shuttlecock Trajectory in Badminton Using Player Information[J]. Sensors, 2023.

[7] Cowley B, Nguyen A, Miska M, et al. Exploration of Player Behaviours from Broadcast Badminton Videos[J]. Computer Graphics Forum, 2023.

[8] Cui Y, Zeng C, Zhao X, Yang Y, Wu G, Wang L. SportsMOT: A Large Multi-Object Tracking Dataset in Multiple Sports Scenes[C]//ICCV. 2023.

[9] Zhang Y, Sun P, Jiang Y, et al. ByteTrack: Multi-Object Tracking by Associating Every Detection Box[C]//ECCV. 2022.

[10] Cao J, Pang J, Weng X, Khirodkar R, Kitani K. Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking[C]//CVPR. 2023.

[11] Aharon N, Orfaig R, Bobrovsky B Z. BoT-SORT: Robust Associations Multi-Pedestrian Tracking[EB/OL]. arXiv:2206.14651, 2022.

[12] Wang C Y, Bochkovskiy A, Liao H Y M. YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors[C]//CVPR. 2023.

[13] Wang C Y, Yeh I H, Liao H Y M. YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information[EB/OL]. arXiv:2402.13616, 2024.

[14] Zhao Y, Lv W, Xu S, et al. DETRs Beat YOLOs on Real-time Object Detection[C]//CVPR. 2024.

[15] Zhang H, Li F, Liu S, et al. DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection[C]//ICLR. 2023.

[16] Liu S, Zeng Z, Ren T, et al. Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection[C]//ECCV. 2024.

[17] Maji D, Nagori S, Mathew M, Poddar D. YOLO-Pose: Enhancing YOLO for Multi Person Pose Estimation Using Object Keypoint Similarity Loss[C]//CVPR Workshops. 2022.

[18] Xu Y, Zhang J, Zhang Q, Tao D. ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation[C]//NeurIPS. 2022.

[19] Jiang T, Lu P, Zhang L, et al. RTMPose: Real-Time Multi-Person Pose Estimation Based on MMPose[EB/OL]. arXiv:2303.07399, 2023.

[20] Zhu W, Ma X, Liu Z, Liu L, Wu W, Wang Y. MotionBERT: A Unified Perspective on Learning Human Motion Representations[C]//ICCV. 2023.

[21] Chi H G, Ha M H, Chi S, Lee S W, Huang Q, Ramani K. InfoGCN: Representation Learning for Human Skeleton-Based Action Recognition[C]//CVPR. 2022.

[22] Lee J, Lee M, Lee D, Lee S. Hierarchically Decomposed Graph Convolutional Networks for Skeleton-Based Action Recognition[C]//ICCV. 2023.

[23] Tong Z, Song Y, Wang J, Wang L. VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training[C]//NeurIPS. 2022.

[24] Li Y, Wu C Y, Fan H, et al. MViTv2: Improved Multiscale Vision Transformers for Classification and Detection[C]//CVPR. 2022.

[25] Li K, Wang Y, Gao P, et al. UniFormer: Unified Transformer for Efficient Spatial-Temporal Representation Learning[C]//ICLR. 2022.

[26] Cui Y, Jiang C, Wang L, Wu G. MixFormer: End-to-End Tracking with Iterative Mixed Attention[C]//CVPR. 2022.

[27] Chen X, Peng H, Wang D, Lu H, Hu H. SeqTrack: Sequence to Sequence Learning for Visual Object Tracking[C]//CVPR. 2023.

[28] Yan B, Jiang Y, Sun P, et al. Towards Grand Unification of Object Tracking[C]//ECCV. 2022.

[29] Feichtenhofer C, Fan H, Li Y, He K. Masked Autoencoders as Spatiotemporal Learners[C]//NeurIPS. 2022.

[30] Yang C, Wu Y, Zhu Z, et al. BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers[EB/OL]. arXiv:2208.06366, 2022.

[31] ychenfen. badminton-pipeline-repro[EB/OL]. GitHub repository.

## 14 写作时应避免的问题

1. 不要写“本文提出了一种全新的 TrackNet 模型”。
2. 不要把 quality_score 写成 accuracy。
3. 不要把没有人工标注的数据写成监督评测。
4. 不要只罗列代码文件，要解释每个模块解决什么问题。
5. 不要在实验部分只贴表格，必须先给结论，再用表格支撑。
6. 不要把参考文献堆在相关工作里逐篇复述，要按问题归类。
7. 不要插入图表后不解释，图表前后都要有承接文字。
