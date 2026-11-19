"""Command-line interface for the Scam Detection & Awareness chatbot.

Run:  python chatbot.py

  <question>        ask about scams, red flags, what to do, where to report
  check <message>   analyse a suspicious message (or just type 'check' and paste it)
  sources           toggle showing which dataset rows were used
  reset             clear chat history
  quit              exit
"""

import sys

from rag import SAFETY_NOTE, RagChatbot


def main():
    try:  # load GEMINI_API_KEY from a local .env file if python-dotenv is installed
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    bot = RagChatbot()
    show_sources = True
    print(f"Scam Detection bot [{bot.mode}] - {len(bot.df)} knowledge entries indexed.")
    print("Ask a question, or use 'check <suspicious message>'. Commands: sources | reset | quit\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        cmd = query.lower()
        if cmd in {"quit", "exit", "q"}:
            break
        if cmd == "reset":
            bot.reset()
            print("Bot: Conversation cleared.\n")
            continue
        if cmd == "sources":
            show_sources = not show_sources
            print(f"Bot: Source display {'on' if show_sources else 'off'}.\n")
            continue

        if cmd == "check" or cmd.startswith("check "):
            message = query[5:].strip()
            if not message:
                try:
                    message = input("Paste the suspicious message: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
            if not message:
                continue
            reply = bot.check(message)
            print(f"Bot: {reply.text}")
            if bot.client:
                print(f"     ({SAFETY_NOTE})")
        else:
            reply = bot.ask(query)
            print(f"Bot: {reply.text}")

        if show_sources and reply.sources:
            refs = "; ".join(
                f"#{i + 2} \"{bot.df.loc[i, 'Question']}\" ({s:.2f})" for i, s in reply.sources
            )
            print(f"     [sources: scam_faq.csv row {refs}]")
        print()


if __name__ == "__main__":
    sys.exit(main())
