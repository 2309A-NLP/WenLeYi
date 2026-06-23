# -*- coding: utf-8 -*-
"""
工单编号：人工智能NLP-Agent 数字人项目-日程提醒智能体任务
精确提醒模块：每条日程单独定时，到点精准触发（不再定时扫描）
"""
import random
import threading
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger
import database
import config

# 提醒话术池
REMIND_PHRASES = [
    "温馨提醒：{content}的时间到啦，主人！",
    "主人！是时候{content}了喔~",
    "亲爱的主人，现在是{content}的时候啦！",
    "嘿，主人，该{content}了哦~",
]

# 新提醒暂存队列，前端通过轮询获取
_remind_queue = []
_queue_lock = threading.Lock()

# 已调度的job记录 {schedule_id: job_id}
_scheduled_jobs = {}
_jobs_lock = threading.Lock()


def get_new_reminders():
    """获取新的提醒并清空队列"""
    with _queue_lock:
        reminders = list(_remind_queue)
        _remind_queue.clear()
    return reminders


def _push_reminder(remind_text):
    """将提醒推入队列"""
    with _queue_lock:
        _remind_queue.append({
            "text": remind_text,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })


def _fire_reminder(schedule_id):
    """触发单条日程提醒"""
    try:
        record = database.get_schedule_by_id(schedule_id)
    except Exception as e:
        print("[精确提醒] 查询日程失败 id={}: {}".format(schedule_id, e))
        return

    # 已删除或已提醒则跳过
    if not record or record.get("status") == 0 or record.get("remind_status") == 0:
        return

    content = record["content"]
    repeat_rule = record.get("repeat_rule")

    # 随机话术
    phrase = random.choice(REMIND_PHRASES).format(content=content)

    # 标记已提醒
    try:
        database.mark_reminded(schedule_id)
    except Exception as e:
        print("[精确提醒] 标记失败 id={}: {}".format(schedule_id, e))
        return

    # 记录日志
    try:
        database.add_remind_log(schedule_id, phrase)
    except Exception as e:
        print("[精确提醒] 日志失败 id={}: {}".format(schedule_id, e))

    # 循环日程创建下一次
    if repeat_rule:
        try:
            new_id = database.create_next_cycle(record)
            if new_id:
                # 为下一次日程也调度精确提醒
                schedule_reminder(new_id, record["schedule_date"], record["schedule_time"], repeat_rule)
                print("[精确提醒] 循环日程已创建下次 id={}".format(new_id))
        except Exception as e:
            print("[精确提醒] 循环创建失败: {}".format(e))

    # 推入前端队列
    _push_reminder(phrase)
    with _jobs_lock:
        _scheduled_jobs.pop(schedule_id, None)
    print("[精确提醒] {}".format(phrase))


def schedule_reminder(schedule_id, schedule_date, schedule_time, repeat_rule=None):
    """为单条日程注册精确提醒"""
    if isinstance(schedule_time, str):
        # 字符串转time
        parts = schedule_time.replace(":", "-").split("-")
        h, m = int(parts[0]), int(parts[1])
        s = int(parts[2]) if len(parts) > 2 else 0
    else:
        # timedelta
        total = int(schedule_time.total_seconds()) if hasattr(schedule_time, 'total_seconds') else 0
        h, m, s = total // 3600, (total % 3600) // 60, total % 60

    if isinstance(schedule_date, str):
        y, mo, d = int(schedule_date[:4]), int(schedule_date[5:7]), int(schedule_date[8:10])
    else:
        y, mo, d = schedule_date.year, schedule_date.month, schedule_date.day

    fire_time = datetime(y, mo, d, h, m, s)

    # 如果时间已过，不调度
    if fire_time <= datetime.now():
        return

    job_id = "remind_{}".format(schedule_id)

    with _jobs_lock:
        # 先移除旧的
        if schedule_id in _scheduled_jobs:
            try:
                scheduler.remove_job(_scheduled_jobs[schedule_id])
            except Exception:
                pass

    try:
        scheduler.add_job(
            _fire_reminder,
            trigger=DateTrigger(run_date=fire_time),
            args=[schedule_id],
            id=job_id,
            replace_existing=True,
        )
        with _jobs_lock:
            _scheduled_jobs[schedule_id] = job_id
        print("[精确提醒] 已注册: id={} 内容={} 触发时间={}".format(schedule_id, "日程", fire_time.strftime("%H:%M:%S")))
    except Exception as e:
        print("[精确提醒] 注册失败 id={}: {}".format(schedule_id, e))


def cancel_reminder(schedule_id):
    """取消某条日程的提醒"""
    with _jobs_lock:
        job_id = _scheduled_jobs.pop(schedule_id, None)
    if job_id:
        try:
            scheduler.remove_job(job_id)
        except Exception:
            pass


def reschedule_pending():
    """启动时扫描所有待提醒日程并注册精确提醒"""
    try:
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")
        now_time = now.strftime("%H:%M:%S")
        pending = database.get_pending_reminders(today_str, "23:59:59")
        count = 0
        for record in pending:
            schedule_reminder(record["id"], record["schedule_date"], record["schedule_time"], record.get("repeat_rule"))
            count += 1
        print("[精确提醒] 启动时注册了 {} 条提醒".format(count))
    except Exception as e:
        print("[精确提醒] 启动扫描失败: {}".format(e))


# 全局调度器实例
scheduler = BackgroundScheduler()


def start_scheduler():
    """启动调度器"""
    scheduler.start()
    print("[精确提醒] 调度器已启动")
    # 启动后扫描所有待提醒日程
    reschedule_pending()
    # 每天凌晨2点清除过期回收站
    from apscheduler.triggers.cron import CronTrigger
    scheduler.add_job(
        _cleanup_recycle,
        trigger=CronTrigger(hour=2, minute=0),
        id="daily_cleanup_recycle",
        replace_existing=True,
    )
    print("[定时任务] 每日清除过期回收站已注册")


def _cleanup_recycle():
    """每日清除超过30天的回收站数据"""
    try:
        count = database.cleanup_old_recycle(30)
        if count > 0:
            print("[定时任务] 已清除{}条过期回收站数据".format(count))
    except Exception as e:
        print("[定时任务] 清除回收站失败: {}".format(e))


def stop_scheduler():
    """停止调度器"""
    if scheduler.running:
        scheduler.shutdown()
        print("[精确提醒] 调度器已停止")
