"""선택 기능: OpenAI API 키가 있을 때만 결과 해설 초안을 만든다."""
from __future__ import annotations

import json
import os

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def api_key() -> str | None:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    if os.name == "nt":  # Windows 사용자/시스템 환경변수
        try:
            import winreg
            for root, path in [(winreg.HKEY_CURRENT_USER, "Environment"),
                               (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")]:
                try:
                    with winreg.OpenKey(root, path) as k:
                        return winreg.QueryValueEx(k, "OPENAI_API_KEY")[0]
                except OSError:
                    continue
        except Exception:
            return None
    return None


def available() -> bool:
    return bool(api_key())


def condense(blocks: list, limit: int = 12000) -> str:
    """차트는 빼고 표·지표·해설만 텍스트로 압축한다."""
    parts = []
    for b in blocks:
        t = b.get("type")
        if t == "metrics":
            parts.append("지표: " + ", ".join(f"{i['label']}={i['value']}" for i in b["items"]))
        elif t == "table":
            rows = b["rows"][:15]
            parts.append(f"표[{b.get('title')}] 열={b['columns']} 행={json.dumps(rows, ensure_ascii=False, default=str)}")
        elif t in ("note", "interpret"):
            parts.append(b.get("text", ""))
        elif t == "heading":
            parts.append(f"## {b['title']} {b.get('sub') or ''}")
        elif t == "code":
            parts.append(b["text"][:1500])
        elif t == "tabs":
            for tab in b["tabs"]:
                parts.append(f"[{tab['label']}]\n" + condense(tab["blocks"], limit // 3))
    s = "\n".join(parts)
    return s[:limit]


def explain(step_title: str, blocks: list, context: dict) -> str:
    key = api_key()
    if not key:
        raise ValueError("OPENAI_API_KEY가 설정되지 않아 AI 해설을 사용할 수 없습니다.")
    from openai import OpenAI
    client = OpenAI(api_key=key)
    prompt = (f"당신은 제조 데이터 분석 컨설턴트입니다. 아래는 '{step_title}' 단계의 분석 결과 요약입니다.\n"
              f"분석 맥락: {json.dumps(context, ensure_ascii=False)}\n\n{condense(blocks)}\n\n"
              "경영층과 현장 엔지니어가 이해할 수 있게 한국어로 해설하세요. 형식: 1) 핵심 발견 3가지 2) 주의할 점 3) 다음 행동 제안. "
              "수치는 결과에 있는 값만 인용하고, 상관과 인과를 구분하세요. 500자 내외.")
    resp = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.3)
    return resp.choices[0].message.content.strip()
