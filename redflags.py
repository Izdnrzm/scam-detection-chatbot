"""Rule-based red-flag scanner for suspicious messages.

Works fully offline and is used in two ways:
  * on its own, when no Gemini API key is available;
  * as "automated checks" handed to the LLM, so its verdict is grounded in concrete findings.

It is a heuristic, not a guarantee: a clean result never means a message is safe.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Negative look-behinds stop "do NOT share your OTP" (a genuine bank warning) from being flagged.
_NOT = r"(?<!not )(?<!never )(?<!n't )"
_SECRET_WORDS = r"(?:otp|tac|pin|cvv|password|passcode|verification code|security code)"
_GIVE_VERBS = r"(?:send|share|give|tell|reply|provide|enter|confirm|read out|forward)"


@dataclass(frozen=True)
class Rule:
    name: str
    weight: int
    pattern: re.Pattern
    hint: str


def _rule(name: str, weight: int, pattern: str, hint: str) -> Rule:
    return Rule(name, weight, re.compile(pattern, re.IGNORECASE | re.DOTALL), hint)


RULES: list[Rule] = [
    _rule("Urgency or threat", 2,
          r"\b(urgent(ly)?|immediately|right now|within \d+ ?(hours?|hrs?)|last (warning|chance|notice)|"
          r"final notice|act now|expires? (today|soon)|will be (blocked|suspended|closed|frozen|terminated)|"
          r"arrest(ed)?|warrant|legal action)\b",
          "Scammers rush you so you act before checking."),
    _rule("Asks for an OTP, PIN or password", 3,
          rf"\b{_SECRET_WORDS}\b.{{0,40}}{_NOT}\b{_GIVE_VERBS}\b|{_NOT}\b{_GIVE_VERBS}\b.{{0,40}}\b{_SECRET_WORDS}\b",
          "No genuine bank, agency or company asks you to share these."),
    _rule("Contains a link", 1,
          r"(https?://|www\.)\S+",
          "Never tap links in unexpected messages; open the official app or type the address yourself."),
    _rule("Shortened or disguised link", 2,
          r"\b(bit\.ly|tinyurl\.com|t\.co|cutt\.ly|s\.id|is\.gd|rb\.gy|shorturl\.\w+)/\S+|"
          r"https?://\S+\.(top|xyz|vip|cc|icu|click|cyou|shop)\b",
          "Short or unusual domains hide where the link really goes."),
    _rule("Asks you to install an APK or app", 3,
          r"\.apk\b|\b(install|download)\b.{0,20}\b(apk|app)\b",
          "Apps sent outside the official app stores can take over your phone and banking."),
    _rule("Asks for remote access", 3,
          r"\b(anydesk|teamviewer|ultraviewer|remote access|remote control)\b",
          "Remote access lets a stranger operate your device and bank."),
    _rule("Unusual payment method", 3,
          r"\b(gift ?cards?|itunes card|google play card|bitcoin|btc|usdt|crypto(currency)?|western union)\b",
          "Gift cards and crypto are hard to trace or reverse, which is why scammers ask for them."),
    _rule("'Safe account' or move-your-money request", 3,
          r"\b(safe|secure|holding|temporary) account\b|\bmove (your )?(money|funds|savings)\b|"
          r"\btransfer (your )?(money|funds|savings) to\b",
          "No police, court or bank asks you to move savings to a 'safe' account."),
    _rule("Claims to be police, court or government", 2,
          r"\b(police|polis|court|mahkamah|lhdn|irb|iras|immigration|bank negara|bnm|mcmc|"
          r"prosecutor|money laundering|case (no|number|id))\b",
          "Authority impersonation is used to frighten victims into paying."),
    _rule("Pressure to keep it secret", 3,
          r"\b(do not|don't|dont) (tell|inform|disclose)\b|keep (this|it) (a )?(secret|confidential)|"
          r"do not hang up|stay on the line",
          "Secrecy stops family or the bank from warning you."),
    _rule("Too-good-to-be-true offer or prize", 2,
          r"\b(guaranteed (profit|returns?|income)|risk[- ]free|daily (profit|returns?|income)|double your|"
          r"passive income|congratulations|you('ve| have)? won|winner|lucky draw|claim your (prize|reward|gift))\b",
          "Unexpected prizes and guaranteed returns are classic bait."),
    _rule("Easy-money task or job offer", 3,
          r"\b(like (and|&) (subscribe|share)|complete (simple |easy )?tasks?|commission per task|"
          r"part[- ]time (job|work).{0,60}(earn|commission|daily|per day))\b",
          "Task scams pay small amounts first, then demand deposits."),
    _rule("Parcel or delivery problem", 2,
          r"\b(parcel|package|delivery|shipment|courier)\b.{0,80}\b(fee|fine|customs|on hold|held|failed|"
          r"reschedule|redeliver|confirm your address|unpaid)\b",
          "Fake courier notices ask for a small fee or your details."),
    _rule("Asks for an upfront fee or deposit", 2,
          r"\b(pay|transfer|deposit|send)\b.{0,40}\b(processing fee|admin fee|deposit|insurance|release fee|clearance)\b",
          "Genuine prizes, loans and jobs do not require upfront payment."),
    _rule("Friend or family 'new number' story", 2,
          r"\b(hi (mum|mom|dad)|new number|changed my number|lost my phone)\b.{0,100}\b(money|pay|transfer|help)\b",
          "Call the person on their old, known number before sending anything."),
    _rule("Account-problem wording", 1,
          r"\b(unusual|suspicious) (activity|login|transaction)|verify your (account|identity|details)|"
          r"update your (account|details|information)|account (has been )?(locked|compromised|suspended)\b",
          "Fake alerts say your account has a problem so you click."),
]

HIGH = 5
MEDIUM = 2


def scan(text: str) -> tuple[int, list[Rule]]:
    """Return (risk score, matched rules) for a message."""
    hits = [r for r in RULES if r.pattern.search(text)]
    return sum(r.weight for r in hits), hits


def verdict(score: int) -> str:
    if score >= HIGH:
        return "HIGH RISK - very likely a scam"
    if score >= MEDIUM:
        return "SUSPICIOUS - treat as a possible scam"
    return "NO OBVIOUS RED FLAGS - but scammers adapt, so still verify through an official channel"
