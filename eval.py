"""Evaluate the chatbot.

1. Retrieval: does the right dataset row appear in the top-k for a reworded question?
2. Off-topic rejection: are unrelated questions refused?
3. Message check: does the red-flag scanner rate scam messages >= SUSPICIOUS and genuine ones clean?

Run:  python eval.py     (uses Gemini embeddings if GEMINI_API_KEY is set, else TF-IDF)
"""

import os

import redflags
from rag import EmbeddingRetriever, TfidfRetriever, load_dataset

# (reworded student question, exact Question text of the row that should be found)
CASES = [
    ("what does phishing mean", "What is phishing?"),
    ("fake SMS from my bank asking me to click", "How do I spot a fake bank SMS or email?"),
    ("I got a message saying my parcel is stuck and I must pay a fee", "How do I recognise a fake parcel delivery message?"),
    ("someone calling from the police says I am involved in money laundering", "What is a government or police impersonation scam?"),
    ("a caller from my bank's fraud team wants me to move money", "What is a bank officer impersonation scam?"),
    ("online boyfriend I never met asks for money", "What is a romance scam?"),
    ("crypto platform promising guaranteed returns", "What is an investment scam?"),
    ("part-time job liking videos for commission then asked to deposit", "What is a job or task scam?"),
    ("seller wants bank transfer for a cheap phone", "What is an online shopping or marketplace scam?"),
    ("fake QR code on a parking meter", "What is a QR code scam?"),
    ("I won a lucky draw I never entered", "What is a lottery or prize scam?"),
    ("popup says my computer has a virus and to call a number", "What is a tech support scam?"),
    ("loan company wants a processing fee first", "What is a fake loan scam?"),
    ("landlord wants a deposit before I see the flat", "What is a rental scam?"),
    ("hi mum this is my new number please send money", "What is the 'Hi Mum' or friend-in-need WhatsApp scam?"),
    ("scammer used AI to copy my son's voice", "What are deepfake and voice-cloning scams?"),
    ("someone is blackmailing me with my private photos", "What is a sextortion scam?"),
    ("supplier emailed new bank details on an invoice", "What is a fake invoice or business email scam?"),
    ("what are the warning signs of a scam", "What are the most common red flags of a scam?"),
    ("caller is rushing me to act now", "Why do scammers create a sense of urgency?"),
    ("the bank texted me an OTP and a caller asked me to read it out", "Should I ever share my OTP or TAC?"),
    ("is this link dangerous", "How can I check if a link is safe?"),
    ("how to tell if a shopping website is fake", "How do I check if a website is fake?"),
    ("how do I know the caller is really from my bank", "How can I verify a caller who says they are from my bank or the government?"),
    ("why does the call show my bank's real number", "What is caller ID spoofing?"),
    ("someone sent me an APK to install", "Is it safe to install an APK file sent to me?"),
    ("I tapped a link in a scam text", "What should I do if I clicked a suspicious link?"),
    ("I gave my OTP to a stranger", "What if I already shared my OTP, password or banking details?"),
    ("I just transferred money to a scammer", "What should I do if I already transferred money to a scammer?"),
    ("where do I report a scam in Malaysia", "How do I report a scam in Malaysia?"),
    ("who do I call about a scam in Singapore", "How do I report a scam in Singapore?"),
    ("how to check if a bank account is used by scammers", "What is Semak Mule?"),
    ("what proof should I save after being cheated", "What evidence should I keep after a scam?"),
    ("will I recover the money I lost", "Can I get my money back after being scammed?"),
    ("someone asked to rent my bank account for commission", "What is a money mule?"),
    ("how do I keep my grandparents safe from scammers", "How can I protect elderly family members from scams?"),
    ("will police ever tell me to use a safe account", "Will the police or my bank ever ask me to transfer money to a safe account?"),
    ("is this company licensed to take investments", "How do I check if an investment company is licensed?"),
]

OFF_TOPIC = ["what's the weather today", "best pizza near me", "who won the football match", "tell me a joke"]

# (message, minimum expected rating) - generic example texts written for testing
SCAM_MESSAGES = [
    "[BANK] Your account has been suspended. Verify now at http://bit.ly/x9Kd2 within 24 hours or it will be closed.",
    "This is Inspector Lim from the police. Your IC is linked to a money laundering case. Do not tell anyone and transfer your savings to a safe account.",
    "Congratulations! You won a lucky draw prize. Pay a RM50 processing fee to claim your prize.",
    "Your parcel is on hold due to unpaid customs fee. Confirm your address and pay at www.pos-delivery-my.top",
    "Earn RM300 per day with simple part-time tasks. Like and subscribe to videos, commission per task. Send me your TAC to register.",
    "Hi Mum, this is my new number. I lost my phone, urgently need you to transfer money, please help",
    "Please install this APK and allow AnyDesk remote access so we can fix your account.",
]
GENUINE_MESSAGES = [
    "Your OTP is 482913. Do not share this code with anyone. Valid for 5 minutes.",
    "Hi, your dentist appointment is tomorrow at 3pm. Reply YES to confirm.",
    "Your Grab driver is arriving in 3 minutes.",
    "Meeting moved to 4pm in room 2. See you there!",
    "Transaction alert: RM25.00 spent at STARBUCKS using card ending 1234. If this was not you, call the number on the back of your card.",
]


def main():
    df = load_dataset()
    if os.getenv("GEMINI_API_KEY"):
        from google import genai

        retriever = EmbeddingRetriever(df, genai.Client())
    else:
        retriever = TfidfRetriever(df)
    row_of = {q: i for i, q in enumerate(df["Question"])}

    hit1 = hit3 = 0
    for query, expected in CASES:
        ranked = [i for i, _ in retriever.search(query, k=3)]
        hit1 += ranked[0] == row_of[expected]
        hit3 += row_of[expected] in ranked
        if ranked[0] != row_of[expected]:
            print(f"  miss: {query!r} -> got {df.loc[ranked[0], 'Question']!r}")

    n = len(CASES)
    rejected = sum(retriever.search(q, k=1)[0][1] < retriever.min_score for q in OFF_TOPIC)
    print(f"\n[Retrieval] {retriever.name}")
    print(f"Hit@1: {hit1}/{n} = {hit1 / n:.0%}   Hit@3: {hit3}/{n} = {hit3 / n:.0%}")
    print(f"Off-topic correctly rejected: {rejected}/{len(OFF_TOPIC)}")

    caught = 0
    for m in SCAM_MESSAGES:
        score, _ = redflags.scan(m)
        caught += score >= redflags.MEDIUM
        if score < redflags.MEDIUM:
            print(f"  scam not flagged (score {score}): {m[:60]!r}")
    clean = 0
    for m in GENUINE_MESSAGES:
        score, flags = redflags.scan(m)
        clean += score < redflags.MEDIUM
        if score >= redflags.MEDIUM:
            print(f"  false alarm (score {score}, {[f.name for f in flags]}): {m[:60]!r}")
    print("\n[Message check - rule scanner]")
    print(f"Scam messages flagged: {caught}/{len(SCAM_MESSAGES)}   Genuine messages left clean: {clean}/{len(GENUINE_MESSAGES)}")


if __name__ == "__main__":
    main()
