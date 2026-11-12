"""1단계: 파일 읽기, 인코딩 판별, 자동 프로파일, 열 타입·역할 추론."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd

from . import blocks as B

ENCODINGS = ["utf-8-sig", "cp949", "euc-kr", "utf-8", "latin1"]
LIMIT_BYTES = 10**9


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def excel_sheets(path: Path) -> list[str]:
    try:
        return pd.ExcelFile(path, engine="openpyxl").sheet_names
    except Exception:
        return []


def sniff_sep(sample: str) -> str:
    counts = {s: sample.count(s) for s in [",", "\t", ";", "|"]}
    return max(counts, key=counts.get) if max(counts.values()) > 0 else ","


def read_file(path: Path, encoding: str | None = None, sheet=None, sep: str | None = None) -> tuple[pd.DataFrame, dict]:
    suffix = path.suffix.lower()
    info: dict = {}
    if suffix in (".xlsx", ".xlsm", ".xls"):
        sheets = excel_sheets(path)
        sh = sheet if sheet not in (None, "") else (sheets[0] if sheets else 0)
        df = pd.read_excel(path, sheet_name=sh, engine="openpyxl" if suffix != ".xls" else None)
        info.update({"format": "xlsx", "sheet": sh, "sheets": sheets, "encoding": None})
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
        info.update({"format": "parquet", "encoding": None})
    elif suffix in (".csv", ".txt", ".tsv"):
        raw = path.read_bytes()[:200_000]
        tried = [encoding] if encoding else ENCODINGS
        last = None
        df = None
        for enc in tried:
            try:
                sample = raw.decode(enc)
            except (UnicodeDecodeError, LookupError) as e:
                last = e
                continue
            s = sep or ("\t" if suffix == ".tsv" else sniff_sep(sample.split("\n", 5)[0] if sample else ""))
            try:
                df = pd.read_csv(path, encoding=enc, sep=s, low_memory=False)
                info.update({"format": "csv", "encoding": enc, "sep": s})
                break
            except UnicodeDecodeError as e:
                last = e
                continue
        if df is None:
            raise ValueError(f"인코딩을 판별하지 못했습니다: {last}")
    else:
        raise ValueError("CSV, XLSX, Parquet 파일만 지원합니다.")
    df.columns = dedupe_columns([str(c).strip() for c in df.columns])
    df = df.loc[:, [c for c in df.columns if not (c.startswith("Unnamed:") and df[c].isna().all())]]
    return df, info


def dedupe_columns(cols: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for c in cols:
        if c in seen:
            seen[c] += 1
            out.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            out.append(c)
    return out


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """문자로 읽힌 숫자·날짜 열을 변환한다."""
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if s.dtype == object or str(s.dtype).startswith("string"):
            nonnull = s.dropna().astype(str).str.strip()
            if nonnull.empty:
                continue
            cleaned = nonnull.str.replace(",", "", regex=False)
            num = pd.to_numeric(cleaned, errors="coerce")
            if num.notna().mean() >= 0.98:
                out[c] = pd.to_numeric(s.astype(str).str.strip().str.replace(",", "", regex=False).replace({"": None, "nan": None, "None": None}), errors="coerce")
                continue
            if nonnull.str.match(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}").mean() >= 0.9:
                dt = pd.to_datetime(s, errors="coerce")
                if dt.notna().mean() >= 0.9 * s.notna().mean():
                    out[c] = dt
    return out


def infer_kind(s: pd.Series, n: int) -> str:
    """연속·이진·범주·날짜·ID·상수·텍스트 추론."""
    nun = s.nunique(dropna=True)
    if nun <= 1:
        return "constant"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_bool_dtype(s):
        return "binary"
    if pd.api.types.is_numeric_dtype(s):
        if nun == 2:
            return "binary"
        nonnull = s.dropna()
        is_int = np.all(np.mod(nonnull.values.astype(float), 1) == 0)
        if is_int and nun == len(nonnull) and n > 50:
            # 연속 정수이고 모두 고유 → ID 가능성
            vals = np.sort(nonnull.values)
            if (np.diff(vals) == 1).mean() > 0.9:
                return "id"
        if is_int and nun <= min(10, max(3, n // 50)):
            return "categorical"
        return "numeric"
    nonnull = s.dropna().astype(str)
    if nun == 2:
        return "binary"
    if nun == len(nonnull) and n > 20:
        return "id"
    if nonnull.str.len().mean() > 40 and nun > 0.5 * len(nonnull):
        return "text"
    if nun > max(50, 0.5 * len(nonnull)):
        return "high_cardinality"
    return "categorical"


KIND_LABEL = {
    "numeric": "연속형", "categorical": "범주형", "binary": "이진", "datetime": "날짜",
    "id": "ID 의심", "constant": "상수", "text": "텍스트", "high_cardinality": "고카디널리티 범주",
}

ROLE_LABEL = {"feature": "특징", "target": "타깃", "id": "식별자", "date": "날짜", "group": "그룹", "exclude": "제외"}


def default_role(kind: str) -> str:
    return {"id": "id", "datetime": "date", "constant": "exclude", "text": "exclude"}.get(kind, "feature")


def profile(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    n = len(df)
    rows = []
    kinds = {}
    for c in df.columns:
        s = df[c]
        k = infer_kind(s, n)
        kinds[c] = k
        num = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        rows.append({
            "변수명": c,
            "저장 타입": str(s.dtype),
            "추론 타입": KIND_LABEL[k],
            "결측수": int(s.isna().sum()),
            "결측률(%)": round(float(s.isna().mean() * 100), 2),
            "고유값 수": int(s.nunique(dropna=True)),
            "평균": float(s.mean()) if num else None,
            "표준편차": float(s.std()) if num else None,
            "최소": float(s.min()) if num else None,
            "최대": float(s.max()) if num else None,
            "왜도": float(s.skew()) if num and s.nunique() > 2 else None,
            "0 비율(%)": round(float((s == 0).mean() * 100), 2) if num else None,
            "최빈값": str(s.mode(dropna=True).iloc[0]) if s.notna().any() else None,
        })
    return pd.DataFrame(rows), kinds


def profile_blocks(df: pd.DataFrame, name: str, info: dict) -> tuple[list, dict]:
    prof, kinds = profile(df)
    counts = pd.Series([KIND_LABEL[k] for k in kinds.values()]).value_counts()
    dup = int(df.duplicated().sum())
    out = [
        B.heading(name, " · ".join(x for x in [
            f"형식 {info.get('format')}", f"인코딩 {info['encoding']}" if info.get("encoding") else None,
            f"시트 {info['sheet']}" if info.get("sheet") is not None and info.get("format") == "xlsx" else None]
            if x)),
        B.metrics([("행 수", len(df)), ("열 수", df.shape[1]), ("전체 결측 셀", int(df.isna().sum().sum())),
                   ("완전 중복 행", dup), ("메모리(MB)", round(df.memory_usage(deep=True).sum() / 1e6, 2))]),
        B.table(df.head(10), "미리보기 (상위 10행)"),
        B.table(prof, "변수 정보와 자동 프로파일", download=f"{name}_profile.csv"),
        B.table(df.describe(include="all").T.reset_index().rename(columns={"index": "변수"}), "기술통계량"),
        B.kv([(k, int(v)) for k, v in counts.items()], "추론 타입 분포"),
    ]
    warns = []
    for c, k in kinds.items():
        if k == "id":
            warns.append(f"`{c}`는 모든 값이 고유해 식별자로 보입니다. 모델 특징에서 제외됩니다.")
        elif k == "constant":
            warns.append(f"`{c}`는 값이 하나뿐이라 분석에 쓸모가 없습니다.")
        elif k == "high_cardinality":
            warns.append(f"`{c}`는 범주가 매우 많습니다. 상위 범주만 쓰거나 타깃 인코딩을 고려하세요.")
    if dup:
        warns.append(f"완전히 같은 행이 {dup:,}개 있습니다. 중복 수집인지 확인하세요.")
    if warns:
        out.append(B.note("\n".join("• " + w for w in warns), "warn"))
    return out, kinds
