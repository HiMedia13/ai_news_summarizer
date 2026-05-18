"""
Discord 봇 - 웹과 동일한 LangGraph 파이프라인을 슬래시 커맨드로 노출.

실행:
  1. https://discord.com/developers/applications 에서 봇 생성 → Token 복사
  2. .env에 DISCORD_BOT_TOKEN=... 추가
  3. python discord_bot.py
  4. 봇을 서버에 초대 (OAuth2 URL Generator → scope: bot + applications.commands,
     Permissions: Send Messages, Use Slash Commands)

명령어:
  /news <질문>       — 검색·분석·브리핑 풀 파이프라인 (웹 / 와 동일)
  /headlines        — GeekNews 헤드라인 빠른 조회
"""
import asyncio
import logging
import os

import discord
from discord import app_commands
from dotenv import load_dotenv

load_dotenv()

from app import (  # noqa: E402
    DEFAULT_SOURCES, build_news_state, client, fetch_geeknews,
    news_agent, sort_for_display,
)

DISCORD_MSG_LIMIT = 2000  # Discord 단일 메시지 글자 수 제한
log = logging.getLogger("discord_bot")


def _escape_title(title: str) -> str:
    """마크다운 링크 안의 `[`/`]`가 파싱을 깨지 않도록 치환."""
    return (title or "").replace("[", "(").replace("]", ")")


def _format_news_result(result: dict) -> str:
    """news_agent 결과를 디스코드용 마크다운 텍스트로 변환.
    웹 화면(top_picks 강조 + 전체 목록 + brief)과 동일한 정보 구성."""
    brief = (result.get("brief") or "").strip()
    top_picks = result.get("top_picks") or []
    real = sort_for_display(result.get("analyzed") or [], top_picks)
    top_links = {t.get("link") for t in top_picks}

    sections = []
    if brief:
        sections.append(f"**📰 종합 브리핑**\n{brief}")
    if top_picks:
        lines = []
        for t in top_picks:
            imp = t.get("importance", 0)
            evaln = (t.get("evaluation") or "").strip()
            lines.append(
                f"- [**{_escape_title(t.get('title', ''))}**]"
                f"(<{t.get('link', '')}>) · 중요도 {imp}/5 — {evaln}"
            )
        sections.append("**⭐ 주목할 기사**\n" + "\n".join(lines))

    other = [it for it in real if it.get("link") not in top_links]
    if other:
        lines = []
        for it in other:
            imp = it.get("importance", 0)
            evaln = (it.get("evaluation") or "").strip()
            tail = f" — {evaln}" if evaln else ""
            lines.append(
                f"- [{_escape_title(it.get('title', ''))}]"
                f"(<{it.get('link', '')}>) · 중요도 {imp}/5{tail}"
            )
        sections.append("**📋 그 외 검색 결과**\n" + "\n".join(lines))

    return "\n\n".join(sections) if sections else "(검색 결과 없음)"


def _split_chunks(text: str, limit: int = DISCORD_MSG_LIMIT - 100) -> list[str]:
    """긴 답변을 Discord 제한에 맞게 분할.
    빈 줄 → 줄 단위 → (최후의 수단으로) 문자 단위 순서로 쪼개,
    마크다운 링크 `[title](url)` 중간을 자르지 않도록 시도."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current = ""
    for paragraph in text.split("\n\n"):
        sep = "\n\n" if current else ""
        if len(current) + len(sep) + len(paragraph) <= limit:
            current = f"{current}{sep}{paragraph}"
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(paragraph) <= limit:
            current = paragraph
            continue
        # 긴 단락은 줄 단위로 쪼개기 (목록 항목 한 줄은 보통 limit 미만)
        for line in paragraph.split("\n"):
            line_sep = "\n" if current else ""
            if len(current) + len(line_sep) + len(line) <= limit:
                current = f"{current}{line_sep}{line}"
                continue
            if current:
                chunks.append(current)
                current = ""
            # 한 줄이 limit를 넘는 극단 케이스만 강제 슬라이스
            while len(line) > limit:
                chunks.append(line[:limit])
                line = line[limit:]
            current = line
    if current:
        chunks.append(current)
    return chunks


intents = discord.Intents.default()


class NewsBot(discord.Client):
    def __init__(self) -> None:
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self) -> None:
        # 전역 동기화 — 봇이 초대된 모든 서버에 슬래시 커맨드 등록(최대 1시간 캐싱)
        await self.tree.sync()


bot = NewsBot()


@bot.event
async def on_ready() -> None:
    print(f"[discord] {bot.user} 로그인 완료 · 등록된 슬래시 커맨드: "
          f"{[c.name for c in bot.tree.get_commands()]}")


@bot.tree.command(name="news",
                  description="AI 뉴스 검색·분석·브리핑 (웹과 동일한 LangGraph 파이프라인)")
@app_commands.describe(query="검색어 (예: 'AI 반도체')")
async def news_cmd(interaction: discord.Interaction, query: str) -> None:
    if client is None:
        await interaction.response.send_message(
            "OPENAI_API_KEY가 설정되지 않았습니다.", ephemeral=True)
        return
    await interaction.response.defer(thinking=True)
    try:
        result = await asyncio.to_thread(
            news_agent.invoke,
            build_news_state(query, list(DEFAULT_SOURCES), do_summarize=True),
        )
        answer = _format_news_result(result)
    except Exception:
        # exception 메시지를 디스코드에 노출하지 않음 (API 키·경로 누출 방지).
        log.exception("news_cmd 실패 — query=%r", query)
        await interaction.followup.send(
            "❌ 에이전트 실행 실패. 봇 콘솔 로그를 확인하세요.")
        return

    chunks = _split_chunks(answer)
    for chunk in chunks:
        await interaction.followup.send(chunk)


@bot.tree.command(name="headlines",
                  description="GeekNews 최신 헤드라인 10건 (검색·평가 없음)")
@app_commands.describe(query="(선택) 키워드 필터 — 예: 'AI'")
async def headlines_cmd(interaction: discord.Interaction,
                        query: str | None = None) -> None:
    await interaction.response.defer(thinking=True)
    try:
        items = await asyncio.to_thread(fetch_geeknews, query or "", 10)
    except Exception:
        log.exception("headlines_cmd 실패 — query=%r", query)
        await interaction.followup.send(
            "❌ GeekNews 조회 실패. 봇 콘솔 로그를 확인하세요.")
        return
    if not items:
        await interaction.followup.send("(GeekNews 결과 없음)")
        return
    # `<url>`로 감싸 Discord 임베드 미리보기를 막아 줄 정리
    lines = [
        f"{i}. [{_escape_title(it.get('title', ''))}](<{it.get('link', '')}>)"
        for i, it in enumerate(items, 1)
    ]
    await interaction.followup.send("\n".join(lines))


if __name__ == "__main__":
    # exception 로그가 콘솔에 출력되도록 root logger 구성.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        raise SystemExit(
            "DISCORD_BOT_TOKEN 미설정 — .env에 추가하세요.\n"
            "발급: https://discord.com/developers/applications → Bot → Reset Token")
    bot.run(token)
