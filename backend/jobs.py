"""백그라운드 작업 실행기: 진행률, 취소, 실패 메시지."""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

_pool = ThreadPoolExecutor(max_workers=3)
_jobs: dict[str, "Job"] = {}
_guard = threading.Lock()


class Cancelled(Exception):
    pass


class Job:
    def __init__(self, pid: str, action: str, params: dict):
        self.id = uuid.uuid4().hex[:12]
        self.pid, self.action, self.params = pid, action, params
        self.status = "queued"
        self.pct, self.message = 0.0, "대기 중"
        self.key = None
        self.blocks = None
        self.error = None
        self.created = time.time()
        self.started = self.ended = None
        self._cancel = False

    def progress(self, pct: float, message: str = ""):
        self.pct = max(0.0, min(1.0, float(pct)))
        if message:
            self.message = message
        if self._cancel:
            raise Cancelled()

    def is_cancelled(self) -> bool:
        return self._cancel

    def cancel(self):
        self._cancel = True
        if self.status == "queued":
            self.status = "cancelled"

    def as_dict(self, with_blocks=True) -> dict:
        d = {"id": self.id, "pid": self.pid, "action": self.action, "status": self.status, "pct": self.pct,
             "message": self.message, "key": self.key, "error": self.error, "created": self.created,
             "started": self.started, "ended": self.ended}
        if with_blocks and self.status == "done":
            d["blocks"] = self.blocks
        return d


def submit(pid: str, action: str, params: dict, runner) -> Job:
    job = Job(pid, action, params)
    with _guard:
        _jobs[job.id] = job
        # 오래된 작업 정리
        if len(_jobs) > 300:
            for k in sorted(_jobs, key=lambda k: _jobs[k].created)[:100]:
                if _jobs[k].status in ("done", "failed", "cancelled"):
                    _jobs.pop(k, None)

    def work():
        if job._cancel:
            job.status = "cancelled"
            return
        job.status, job.started, job.message = "running", time.time(), "실행 중"
        try:
            key, blocks = runner(pid, action, params, job)
            if job._cancel:
                job.status, job.message = "cancelled", "취소됨"
            else:
                job.key, job.blocks, job.status, job.pct, job.message = key, blocks, "done", 1.0, "완료"
        except Cancelled:
            job.status, job.message = "cancelled", "취소됨"
        except Exception as e:  # 사용자에게 보여줄 메시지
            job.status = "failed"
            job.error = str(e) or type(e).__name__
            job.trace = traceback.format_exc()
            print(job.trace)
        finally:
            job.ended = time.time()
    _pool.submit(work)
    return job


def get(job_id: str) -> Job | None:
    return _jobs.get(job_id)


def for_project(pid: str) -> list[dict]:
    return [j.as_dict(False) for j in sorted(_jobs.values(), key=lambda j: -j.created) if j.pid == pid][:30]
