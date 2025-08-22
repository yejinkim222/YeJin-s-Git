# pybo/python/llm_adapters.py
from __future__ import annotations
import os
from typing import List, Dict, Optional


class BaseChatBackend:
    """
    하는 일: 백엔드별 채팅 호출의 공통 인터페이스 제공.
    입력: messages(list[dict]) - {'role': 'system'|'user'|'assistant', 'content': str}
    출력: str - 생성된 텍스트
    """
    def chat(self, messages: List[Dict], **gen_kwargs) -> str:
        raise NotImplementedError



# ---------- Gemini ----------
class GeminiBackend(BaseChatBackend):
    """
    Google Gemini API 어댑터 (google-generativeai 사용).
    • system 메시지는 모두 합쳐 system_instruction 으로 전달
    • user/assistant는 Gemini의 role 규칙(user/model)에 맞춰 변환
    """
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """
        하는 일: 라이브러리 초기화 및 API 키 구성.
        입력: model_name(str), api_key(Optional[str])
        출력: 없음(예외 발생 가능)
        """
        import google.generativeai as genai  # 지연 import
        self.genai = genai
        self.model_name = model_name

        api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY 가 설정되어 있지 않습니다.")
        genai.configure(api_key=api_key)


    @staticmethod
    def _split_system(messages: List[Dict]):
        """
        하는 일: system 메시지들을 합치고, user/assistant를 Gemini 포맷으로 변환.
        입력: messages(list[dict])
        출력: (system_instruction:str|None, contents:list[dict])
        """
        system_texts, turns = [], []
        for m in messages:
            role = (m.get("role") or "").lower()
            content = (m.get("content") or "").strip()
            if not content:
                continue
            if role == "system":
                system_texts.append(content)
            else:
                # Gemini는 assistant 역할을 'model'로 표기
                gemini_role = "user" if role == "user" else "model"
                turns.append({"role": gemini_role, "parts": [content]})
        system_instruction = "\n\n".join(system_texts) if system_texts else None
        return system_instruction, turns


    def chat(self, messages: List[Dict], **gen_kwargs) -> str:
        """
        하는 일: Gemini API에 messages를 전달해 텍스트를 생성.
        입력: messages(list[dict]), **gen_kwargs(temperature, top_p, max_new_tokens)
        출력: str(생성 텍스트, 안전성 필터로 빈 문자열일 수 있음)
        """
        # 타입 헬퍼(선택)
        GenerationConfig = self.genai.types.GenerationConfig

        system_instruction, contents = self._split_system(messages)
        model = self.genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_instruction
        )

        gcfg = GenerationConfig(
            temperature=gen_kwargs.get("temperature", 0.2),
            top_p=gen_kwargs.get("top_p", 0.85),
            max_output_tokens=gen_kwargs.get("max_new_tokens", 200),
        )

        try:
            resp = model.generate_content(contents, generation_config=gcfg)
        except Exception as e:
            # 호출 측에서 사용자에게 보여줄 수 있도록 예외 메시지를 그대로 전달
            raise RuntimeError(f"Gemini 호출 오류: {type(e).__name__}: {e}") from e

        # 안전성 필터에 막히면 resp.text가 비어 있을 수 있음
        text = (getattr(resp, "text", None) or "").strip()
        if text:
            return text

        # 후보를 직접 훑는 보강(일부 버전에서 필요)
        try:
            for cand in getattr(resp, "candidates", []) or []:
                parts = getattr(getattr(cand, "content", None), "parts", None)
                if not parts:
                    continue
                buf = []
                for p in parts:
                    if hasattr(p, "text") and p.text:
                        buf.append(p.text)
                if buf:
                    return "\n".join(buf).strip()
        except Exception:
            pass

        return ""



# ---------- EXAONE (템플릿) ----------
class ExaoneBackend(BaseChatBackend):
    """
    하는 일: 엔드포인트/인증 정보 저장.
    입력: model_name(str), api_key(Optional[str]), base_url(Optional[str])
    출력: 없음(예외 발생 가능)
    """
    def __init__(self, model_name: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        import httpx  # 지연 import
        self.httpx = httpx
        self.model_name = model_name
        self.base_url = base_url or os.getenv(
            "EXAONE_API_BASE",
            "https://api.exaone.ai/v1/chat/completions"
        )
        self.api_key = api_key or os.getenv("EXAONE_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("EXAONE_API_KEY 가 설정되어 있지 않습니다.")


    def chat(self, messages: List[Dict], **gen_kwargs) -> str:
        """
        하는 일: EXAONE API에 messages를 전달해 텍스트를 생성.
        입력: messages(list[dict]), **gen_kwargs(temperature, top_p, max_new_tokens)
        출력: str(생성 텍스트)
        """
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": (m.get("role") or "user"), "content": (m.get("content") or "")}
                for m in messages if m.get("content")
            ],
            "temperature": gen_kwargs.get("temperature", 0.2),
            "top_p": gen_kwargs.get("top_p", 0.85),
            "max_tokens": gen_kwargs.get("max_new_tokens", 200),
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            with self.httpx.Client(timeout=60) as cli:
                r = cli.post(self.base_url, json=payload, headers=headers)
                r.raise_for_status()
                data = r.json()
        except Exception as e:
            raise RuntimeError(f"EXAONE 호출 오류: {type(e).__name__}: {e}") from e

        # 예시 응답 파싱(OpenAI 포맷과 유사하다고 가정)
        try:
            return (data["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            # 다른 포맷일 수도 있으니, 사전 전체를 문자열로 반환
            return str(data)

