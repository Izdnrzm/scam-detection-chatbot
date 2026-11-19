# Scam Detection & Awareness Chatbot (RAG)

A domain-specific chatbot that helps people **recognise scams, check suspicious messages, and know what to do if they have been scammed**. It uses Retrieval-Augmented Generation (RAG) over our own dataset, `scam_faq.csv` (49 Question/Answer pairs).

**LLM provider:** Google Gemini API (free tier) - `gemini-embedding-001` for retrieval, `gemini-2.5-flash` for answers.

## Two features

| Feature | How to use | What happens |
|---------|-----------|--------------|
| **Ask** | `how do I spot a fake parcel SMS?` | Retrieves matching dataset rows; Gemini answers using only those rows |
| **Check** | `check <pasted message>` | A rule-based red-flag scanner scores the message; Gemini explains the verdict using the findings and retrieved scam knowledge |

Urgent situations ("I just got scammed", "I gave my OTP to a stranger") are routed straight to the emergency steps.

## Architecture

```
 scam_faq.csv (Question, Answer)
        | one document per row
        v
 Gemini embeddings --> cached vectors (.cache/)
                              |
 ASK:   question --> embed --> cosine similarity --> top-3 rows above threshold
                                                          |
                                   none relevant          v
              safety message  <--------------  Gemini answers ONLY from retrieved rows
              (bank, NSRC 997 / 1799)           (+ last 3 chat turns)

 CHECK: message --> redflags.py rules (score + findings) --+
                --> retrieve related scam entries ---------+--> Gemini: Verdict / Why / What to do
                    (message is passed as untrusted data between markers)
```

- **Grounded:** the system prompt forbids outside knowledge, so numbers and contacts come from the dataset, not the model's guesses.
- **Prompt-injection guard:** pasted messages are treated as untrusted data; instructions inside them are ignored.
- **Never says "safe":** even with no red flags, the bot advises verifying through an official channel.
- **Fallbacks:** no API key -> TF-IDF retrieval + the rule-based scanner (fully offline); LLM error -> stored answer or rule report; nothing relevant -> safety message.

## Setup

```bash
git clone <your-repo-url>
cd <your-repo>
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Get a free key at https://aistudio.google.com/apikey, then set it in the terminal:

```bash
export GEMINI_API_KEY=your-key-here                    # PowerShell: $env:GEMINI_API_KEY="your-key-here"
```

or copy `.env.example` to `.env` and put the key there (`.env` is git-ignored).

## Run

```bash
python chatbot.py
```

```
You: what is phishing
You: check Your account is suspended. Verify now at http://bit.ly/x9Kd2 within 24 hours
You: check                      (then paste a longer message when prompted)
You: I just got scammed
```

Commands: `check`, `sources` (toggle citations), `reset`, `quit`.

On GitHub: open the repo -> **Code** -> **Codespaces** -> **Create codespace**, then run the commands above in its terminal (add `GEMINI_API_KEY` under repo Settings -> Secrets and variables -> Codespaces).

## Evaluate

```bash
python eval.py
```

Reports retrieval Hit@1 / Hit@3 on 38 reworded questions, off-topic rejection, and how the rule scanner rates 7 scam and 5 genuine messages. Run it with and without `GEMINI_API_KEY` to compare semantic embeddings against the keyword baseline.

Offline baseline (TF-IDF, no API): Hit@1 89%, Hit@3 95%, off-topic rejected 4/4, scam messages flagged 7/7, genuine messages left clean 5/5. These test questions were written by the authors, so treat the numbers as indicative, not as a benchmark. Record your Gemini numbers in your report.

## Files

| File | Purpose |
|------|---------|
| `scam_faq.csv` | Own dataset: Question and Answer columns |
| `rag.py` | RAG core: indexing, retrieval, grounded answers, message check, fallbacks |
| `redflags.py` | Rule-based scam red-flag scanner (works offline) |
| `chatbot.py` | Command-line chat interface |
| `eval.py` | Retrieval and scanner evaluation |
| `requirements.txt` | Dependencies |
| `.env.example` | Template for the API key (never commit the real key) |

## Limitations

- Heuristics and a small dataset cannot catch every scam; a clean result is **not** a guarantee. This is awareness guidance, not legal or financial advice.
- Reporting contacts cover Malaysia (NSRC 997, Semak Mule) and Singapore (ScamShield 1799, Anti-Scam Helpline 1800-722-6688, Police 1800-255-0000, i-Witness). They were checked against police and news sources in November 2026 (the Singapore details come from a 2023 police article) - confirm them on official sites before relying on them, and add your own country's contacts to the CSV.
- Similarity thresholds (`MIN_SCORE_EMBED`, `MIN_SCORE_TFIDF` in `rag.py`) should be tuned on your own test questions.
- Gemini free-tier limits apply; the bot degrades to stored answers if they are hit.
