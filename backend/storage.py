"""프로젝트 저장소: 원본, 작업 데이터, 상태, 결과, 모델을 디스크에 보관한다."""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HOME = Path(os.environ.get("ANALYSIS_HOME") or Path(__file__).resolve().parent.parent / "instance")
PROJECTS = HOME / "projects"
DATA_DIR = Path(os.environ.get("ANALYSIS_DATA") or Path(__file__).resolve().parent.parent / "data")
PROJECTS.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

_locks: dict[str, threading.RLock] = {}
_locks_guard = threading.Lock()


def lock(pid: str) -> threading.RLock:
    with _locks_guard:
        return _locks.setdefault(pid, threading.RLock())


def to_jsonable(value):
    """numpy/pandas 값을 JSON으로 바꿀 수 있는 파이썬 값으로 바꾼다."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (pd.Timestamp,)):
        return None if pd.isna(value) else value.isoformat()
    if value is pd.NaT:
        return None
    if isinstance(value, pd.Interval):
        return str(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(to_jsonable(data), ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def safe_name(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name).strip() or "file"
    return name[:120]


class Project:
    """하나의 분석 프로젝트. 상태는 state.json, 큰 객체는 별도 파일에 둔다."""

    def __init__(self, pid: str):
        self.pid = pid
        self.dir = PROJECTS / pid
        if not self.dir.exists():
            raise KeyError(pid)

    # ---------- 생성/목록 ----------
    @staticmethod
    def create(name: str) -> "Project":
        pid = uuid.uuid4().hex[:12]
        d = PROJECTS / pid
        for sub in ("sources", "data", "results", "models", "reports"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        write_json(d / "meta.json", {"id": pid, "name": name, "created": time.time(), "updated": time.time()})
        write_json(d / "state.json", default_state())
        return Project(pid)

    @staticmethod
    def list() -> list[dict]:
        out = []
        for d in PROJECTS.iterdir():
            meta = read_json(d / "meta.json")
            if meta:
                st = read_json(d / "state.json", {}) or {}
                meta["steps_done"] = len(st.get("results_meta", {}))
                meta["shape"] = st.get("working_shape")
                meta["target"] = st.get("target")
                out.append(meta)
        return sorted(out, key=lambda m: m.get("updated", 0), reverse=True)

    def delete(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    # ---------- 메타/상태 ----------
    @property
    def meta(self) -> dict:
        return read_json(self.dir / "meta.json", {})

    def touch(self) -> None:
        m = self.meta
        m["updated"] = time.time()
        write_json(self.dir / "meta.json", m)

    def rename(self, name: str) -> None:
        m = self.meta
        m["name"] = name
        write_json(self.dir / "meta.json", m)

    @property
    def state(self) -> dict:
        st = read_json(self.dir / "state.json") or default_state()
        for k, v in default_state().items():
            st.setdefault(k, v)
        return st

    def save_state(self, st: dict) -> None:
        write_json(self.dir / "state.json", st)
        self.touch()

    def update_state(self, **changes) -> dict:
        with lock(self.pid):
            st = self.state
            st.update(changes)
            self.save_state(st)
            return st

    # ---------- 데이터프레임 ----------
    def df_path(self, name: str) -> Path:
        return self.dir / "data" / f"{name}.parquet"

    def save_df(self, name: str, df: pd.DataFrame) -> None:
        out = df.copy()
        out.columns = [str(c) for c in out.columns]
        for c in out.columns:
            if out[c].dtype == object:
                # 혼합 타입 열은 문자열로 저장 (결측은 유지)
                vals = out[c]
                if not vals.map(lambda v: isinstance(v, str) or v is None or (isinstance(v, float) and np.isnan(v))).all():
                    out[c] = vals.map(lambda v: v if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
        out.to_parquet(self.df_path(name), index=False)

    def load_df(self, name: str) -> pd.DataFrame:
        p = self.df_path(name)
        if not p.exists():
            raise FileNotFoundError(name)
        return pd.read_parquet(p)

    def has_df(self, name: str) -> bool:
        return self.df_path(name).exists()

    def drop_df(self, name: str) -> None:
        p = self.df_path(name)
        if p.exists():
            p.unlink()

    # ---------- 객체(모델 등) ----------
    def save_obj(self, name: str, obj) -> None:
        joblib.dump(obj, self.dir / "models" / f"{name}.joblib")

    def load_obj(self, name: str):
        p = self.dir / "models" / f"{name}.joblib"
        if not p.exists():
            raise FileNotFoundError(name)
        return joblib.load(p)

    def has_obj(self, name: str) -> bool:
        return (self.dir / "models" / f"{name}.joblib").exists()

    # ---------- 단계 결과 ----------
    def save_result(self, key: str, blocks: list, params: dict | None = None) -> None:
        with lock(self.pid):
            st = self.state
            write_json(self.dir / "results" / f"{key}.json", {"blocks": blocks, "params": params or {}, "time": time.time()})
            st["results_meta"][key] = {"time": time.time(), "data_version": st["data_version"], "model_version": st["model_version"]}
            self.save_state(st)

    def load_result(self, key: str):
        return read_json(self.dir / "results" / f"{key}.json")

    def all_results(self) -> dict:
        out = {}
        for p in sorted((self.dir / "results").glob("*.json")):
            out[p.stem] = read_json(p)
        return out


def default_state() -> dict:
    return {
        "sources": [],            # [{id, name, path, encoding, sheet, rows, cols, hash, origin}]
        "merge": None,            # 결합 설정
        "working_shape": None,
        "roles": {},              # 열 -> role(feature/id/date/group/exclude/target)
        "kinds": {},              # 열 -> 추론 타입
        "recipe": [],             # 결정적 연산(파생변수·행 삭제) 순서 기록. 결합 데이터에서 다시 계산
        "learned_ops": [],        # 학습형 연산(결측 대체·이상치 클리핑): 분할 후 학습 데이터로 적합
        "target": None,
        "target_type": None,      # regression | binary | multiclass
        "positive_label": None,
        "split": None,
        "preprocess": None,
        "models": [],             # 학습된 모델 요약
        "best_model": None,
        "integrity_mode": "improved",  # improved | legacy
        "data_version": 0,
        "model_version": 0,
        "results_meta": {},
        "report_items": [],
        "history": [],            # 사람이 읽는 처리 이력
    }
