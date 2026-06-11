"""
会话管理模块
负责控制推理任务的生命周期：创建会话、停止信号、运行锁
"""

import threading
import uuid

# 这几个全局变量本质上就是“运行中的共享状态表”。
# 新手可以把它理解成一个极简版的任务调度器：谁在跑、能不能停、有没有别的任务抢占。
_session_lock = threading.Lock()
_current_session_id = None
_run_lock = threading.Lock()
_stop_event = threading.Event()


def new_session():
    """创建新会话，返回会话 ID"""
    global _current_session_id
    sid = uuid.uuid4().hex[:12]
    with _session_lock:
        _current_session_id = sid
    _stop_event.clear()
    return sid


def stop_session():
    """停止当前会话，设置停止信号并释放运行锁"""
    _stop_event.set()
    with _session_lock:
        _current_session_id = None
    release_run()


def is_alive(sid):
    """检查指定会话是否仍然活跃"""
    if _stop_event.is_set():
        return False
    with _session_lock:
        return _current_session_id == sid


def try_run():
    """尝试获取运行锁，非阻塞"""
    return _run_lock.acquire(blocking=False)


def release_run():
    """释放运行锁"""
    try:
        _run_lock.release()
    except RuntimeError:
        pass
