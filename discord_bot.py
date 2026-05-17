"""
Discord 봇 - ReAct 뉴스 에이전트를 슬래시 커맨드로 노출.

실행:
  1. https://discord.com/developers/applications 에서 봇 생성 → Token 복사
  2. .env에 DISCORD_BOT_TOKEN=... 추가
  3. python discord_bot.py
  4. 봇을 서버에 초대 (OAuth2 URL Generator → scope: bot + applications.commands,
     Permissions: Send Messages, Use Slash Commands)

명령어:
  /news <질문>       — ReAct 에이전트로 검색·평가·브리핑
  /headlines        — GeekNews 헤드라인 빠른 조회
"""
import asyncio
import os

import discord
from discord import app_commands
from dotenv import load_dotenv

load_dotenv()

from app import react_agent, fetch_geeknews  # noqa: E402

DISCORD_MSG_LIMIT = 2000  # Discord 단일 메시지 글자 수 제한
EMBED_DESC_LIMIT = 4096   # Embed description 제한


def _split_chunks(text: str, limit: int = DISCORD_MSG_LIMIT - 100) -> list[str]:
    """긴 답변을 Discord 제한에 맞게 분할. 가능하면 빈 줄 경계로 나눔."""
    if len(text) <= limit:
        return [text]
    chunks, current = [], ""
    for paragraph in text.split("\n\n"):
        if len(current) + len(paragraph) + 2 <= limit:
            current = f"{current}\n\n{paragraph}" if current else paragraph
        else:
            if current:
                chunks.append(current)
            # 한 단락이 limit를 넘으면 강제로 잘라야 함
            while len(paragraph) > limit:
                chunks.append(paragraph[:limit])
                paragraph = paragraph[limit:]
            current = paragraph
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


@bot.tree.command(name="news", description="AI 뉴스 검색·평가·브리핑 (ReAct 에이전트)")
@app_commands.describe(query="검색어 또는 자연어 요청 (예: 'AI 반도체 3건 평가해줘')")
async def news_cmd(interaction: discord.Interaction, query: str) -> None:
    if react_agent is None:
        await interaction.response.send_message(
            "OPENAI_API_KEY가 설정되지 않았습니다.", ephemeral=True)
        return
    # 응답 deferred — Discord는 deferred 후 최대 15분 응답 가능
    await interaction.response.defer(thinking=True)
    try:
        result = await asyncio.to_thread(
            react_agent.invoke,
            {"messages": [("user", query)]},
            {"recursion_limit": 25},
        )
        answer = result["messages"][-1].content or "(빈 응답)"
    except Exception as e:
        await interaction.followup.send(f"❌ 에이전트 실행 실패: `{e}`")
        return

    chunks = _split_chunks(answer)
    await interaction.followup.send(chunks[0])
    for chunk in chunks[1:]:
        await interaction.followup.send(chunk)


@bot.tree.command(name="headlines",
                  description="GeekNews 최신 헤드라인 10건 (검색·평가 없음)")
@app_commands.describe(query="(선택) 키워드 필터 — 예: 'AI'")
async def headlines_cmd(interaction: discord.Interaction,
                        query: str | None = None) -> None:
    await interaction.response.defer(thinking=True)
    items = await asyncio.to_thread(fetch_geeknews, query or "", 10)
    if not items:
        await interaction.followup.send("(GeekNews 결과 없음)")
        return
    lines = []
    for i, it in enumerate(items, 1):
        title = it.get("title", "").replace("[", "(").replace("]", ")")
        link = it.get("link", "")
        lines.append(f"{i}. [{title}](<{link}>)")
    # `<url>` 형식으로 감싸면 Discord가 임베드 미리보기를 막아 줄 정리됨
    await interaction.followup.send("\n".join(lines))


if __name__ == "__main__":
    token = os.getenv("DISCORD_BOT_TOKEN")
    if not token:
        raise SystemExit(
            "DISCORD_BOT_TOKEN 미설정 — .env에 추가하세요.\n"
            "발급: https://discord.com/developers/applications → Bot → Reset Token")
    bot.run(token)
