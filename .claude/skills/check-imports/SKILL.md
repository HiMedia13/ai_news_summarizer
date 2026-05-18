---
name: check-imports
description: app.py / discord_bot.py / evaluate_retrieval.py 세 모듈을 import해 syntax / NameError / 순환 import를 즉시 검출. 코드 변경 후 빠르게 검증할 때 사용.
---

# /check-imports — 모듈 import 검증

## 절차

1. **세 모듈 일괄 import.** Bash 도구로:
   ```powershell
   python -c "import app; import discord_bot; import evaluate_retrieval; print('OK')"
   ```

2. **결과 해석.**
   - stdout이 `OK` → 모든 모듈 정상. 사용자에게 한 줄로 보고.
   - traceback이 떴다면:
     - **`SyntaxError`**: 어느 파일·라인인지 알려주고, 그 위치를 Read로 열어 보여줌.
     - **`ImportError` / `ModuleNotFoundError`**: 누락된 패키지면 `requirements.txt`와 비교, `pip install <pkg>` 안내.
     - **`AttributeError` / `NameError`**: 최근 리팩토링에서 이름이 바뀌었거나 export 누락 — 해당 위치 찾아 보여줌.
     - **그 외**: traceback 전문 보여주고 사용자와 함께 진단.

3. **회귀 방지.** 만약 사용자가 방금 직접 코드를 편집했고 결과가 OK라면, "변경된 파일을 import 검증 통과했지만, 동작 검증은 별도 (e.g. `/runweb`)" 라고 한 줄 부언.
