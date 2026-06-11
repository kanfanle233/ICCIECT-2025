"""
Gradio 界面模块
构建交互式演示平台的前端界面
"""

import cv2
import gradio as gr
from pathlib import Path

from config import IMAGES_DIR, VIDEOS_DIR, TASK_INFO
from session import stop_session
from utils import list_samples
from inference import predict_image, predict_video_stream, predict_webcam_stream


def build_interface():
    """构建并返回 Gradio Blocks 界面"""
    imgs, vids = list_samples()
    # choices 是一个顺序列表，界面会按这个顺序渲染样本下拉框。
    # 把“图片”和“视频”的标签直接写进字符串里，能减少前端再做额外判断。
    choices = [f"[图片] {n}" for n in imgs] + [f"[视频] {n}" for n in vids]
    css = (
        ".task-card{border-left:4px solid #C87E4F;padding-left:12px;margin-bottom:8px}"
        ".task-title{font-weight:bold;color:#1A4D4E;font-size:16px}"
        ".task-desc{color:#64748B;font-size:13px}"
    )

    with gr.Blocks(title="计算机视觉多任务演示平台") as demo:
        gr.Markdown(
            "# 计算机视觉：从感知到理解\n### 交互式演示平台 | 基于 Ultralytics YOLO 框架"
        )

        current_tab = gr.Textbox(value="file", visible=False, interactive=False)

        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("## 任务选择")
                task = gr.Radio(
                    choices=[
                        ("目标检测", "detection"),
                        ("图像分割", "segmentation"),
                        ("图像分类", "classification"),
                        ("姿态估计", "pose"),
                        ("定向检测", "obb"),
                    ],
                    value="detection",
                    label="选择计算机视觉任务",
                )

                @gr.render(inputs=task)
                def info(t):
                    """根据所选任务类型，渲染对应的任务说明卡片。"""
                    i = TASK_INFO[t]
                    gr.HTML(
                        f'<div class="task-card"><div class="task-title">{i["name"]}</div>'
                        f'<div class="task-desc">{i["desc"]}</div>'
                        f'<div class="task-desc">模型: {i["model"]}</div></div>'
                    )

                gr.Markdown("## 输入源")
                with gr.Tabs() as tabs:
                    with gr.TabItem("📁 上传文件") as tab_file:
                        file_in = gr.File(
                            label="上传图片或视频", file_types=["image", "video"]
                        )
                    with gr.TabItem("📷 实时摄像头") as tab_cam:
                        gr.Markdown(
                            "**拍照推理**：点击摄像头拍照后点【开始推理】\n"
                            "**实时推理**：切到实时模式后直接点【开始推理】"
                        )
                        cam_in = gr.Image(
                            label="摄像头画面", sources=["webcam"], height=320
                        )
                        cam_mode = gr.Radio(
                            choices=[("📸 拍照", "snapshot"), ("🎥 实时", "stream")],
                            value="snapshot",
                            label="摄像头模式",
                        )
                    with gr.TabItem("🎬 预置样本") as tab_sample:
                        sample = gr.Dropdown(
                            choices=choices,
                            label="选择样本",
                            value=choices[0] if choices else None,
                        )
                        preview = gr.Image(label="预览", height=200)

                        def load_sample(c):
                            """加载选中的预置样本，返回图片或视频首帧用于预览。"""
                            if not c:
                                return None
                            if c.startswith("[图片]"):
                                return str(IMAGES_DIR / c.replace("[图片] ", ""))
                            cap = cv2.VideoCapture(
                                str(VIDEOS_DIR / c.replace("[视频] ", ""))
                            )
                            ret, f = cap.read()
                            cap.release()
                            return cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if ret else None

                        sample.change(load_sample, sample, preview)

                with gr.Row():
                    run = gr.Button("▶ 开始推理", variant="primary", size="lg")
                    stop = gr.Button("⏹ 停止推理", variant="stop", size="lg")
                    clear = gr.Button("🗑 清空", variant="secondary")

            with gr.Column(scale=2):
                gr.Markdown("## 推理结果")
                out_img = gr.Image(label="可视化结果", height=480)
                out_txt = gr.Textbox(
                    label="检测详情", lines=10, max_lines=20, interactive=False
                )
                gr.Markdown(
                    '<div style="margin-top:16px;padding:12px;background:#EAE8E1;border-radius:8px;'
                    'font-size:12px;color:#64748B"><strong>数据说明</strong><br>'
                    "样本数据来源于 COCO / ImageNet / Open Images。仅供教学使用。</div>"
                )

        # Tab 切换
        tab_file.select(lambda: "file", outputs=current_tab).then(
            lambda: (None, ""), outputs=[out_img, out_txt]
        )
        tab_cam.select(lambda: "webcam", outputs=current_tab).then(
            lambda: (None, ""), outputs=[out_img, out_txt]
        )
        tab_sample.select(lambda: "sample", outputs=current_tab).then(
            lambda: (None, ""), outputs=[out_img, out_txt]
        )

        # 输入/任务变更时停止并清空
        def reset():
            """停止当前推理会话并清空输出结果。"""
            stop_session()
            return None, ""

        task.change(reset, outputs=[out_img, out_txt])
        file_in.change(reset, outputs=[out_img, out_txt])
        sample.change(reset, outputs=[out_img, out_txt])
        cam_mode.change(reset, outputs=[out_img, out_txt])

        # 推理调度
        def do_run(task, tab, file_obj, sample_choice, cam_img, mode):
            """根据输入源类型和任务选择，调度对应的推理流程并返回结果。"""
            if tab == "file":
                if file_obj is None:
                    yield None, "⚠️ 请先上传文件"
                    return
                p = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
                if Path(p).suffix.lower() in (
                    ".mp4",
                    ".avi",
                    ".mov",
                    ".mkv",
                    ".wmv",
                    ".flv",
                    ".webm",
                ):
                    yield from predict_video_stream(p, task)
                else:
                    img, txt = predict_image(p, task)
                    yield img, txt
            elif tab == "sample":
                if not sample_choice:
                    yield None, "⚠️ 请选择样本"
                    return
                if sample_choice.startswith("[图片]"):
                    p = str(IMAGES_DIR / sample_choice.replace("[图片] ", ""))
                    if not Path(p).exists():
                        yield None, f"❌ 文件不存在: {p}"
                        return
                    img, txt = predict_image(p, task)
                    yield img, txt
                else:
                    p = str(VIDEOS_DIR / sample_choice.replace("[视频] ", ""))
                    if not Path(p).exists():
                        yield None, f"❌ 文件不存在: {p}"
                        return
                    yield from predict_video_stream(p, task)
            elif tab == "webcam":
                if mode == "stream":
                    yield from predict_webcam_stream(task)
                else:
                    if cam_img is None:
                        yield None, "⚠️ 请先拍照或切到实时模式"
                        return
                    img, txt = predict_image(cam_img, task)
                    yield img, txt

        run.click(
            do_run,
            inputs=[task, current_tab, file_in, sample, cam_in, cam_mode],
            outputs=[out_img, out_txt],
        )
        stop.click(stop_session)
        clear.click(
            lambda: (
                stop_session(),
                None,
                "",
                None,
                choices[0] if choices else None,
                None,
            )[1:],
            outputs=[out_img, out_txt, file_in, sample, cam_in],
        )

    return demo, css
