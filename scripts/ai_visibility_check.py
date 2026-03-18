#!/usr/bin/env python3
"""
AI Visibility Check v2 — audyt widoczności marki w modelach AI
Odpytuje wiele modeli równolegle przez OpenRouter API.
Wersja 2: pełne odpowiedzi, kontekst cytatów, sentiment, pozycja wzmianki.

Użycie:
    python3 ai_visibility_check.py --brand "nazwa" --domain "domena.pl"
    python3 ai_visibility_check.py --brand "nazwa" --domain "domena.pl" --lang pl
    python3 ai_visibility_check.py --brand "nazwa" --domain "domena.pl" \
        --prompts "pytanie 1" "pytanie 2"
"""

import os
import sys
import json
import asyncio
import argparse
import re
from datetime import datetime
from typing import Optional

try:
    from openai import AsyncOpenAI
except ImportError:
    print("ERROR: Brak biblioteki openai. Zainstaluj: pip install openai")
    sys.exit(1)


# ─────────────────────────────────────────────
# KONFIGURACJA MODELI
# ─────────────────────────────────────────────

MODELS = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "perplexity/sonar-pro",
    "perplexity/sonar",
    "anthropic/claude-sonnet-4",
    "google/gemini-2.0-flash-001",
    "google/gemini-2.5-pro-preview",
    "mistralai/mistral-large",
    "x-ai/grok-3-mini-beta",
    "meta-llama/llama-4-maverick",
]

NO_ONLINE_SUFFIX = {
    "perplexity/sonar-pro",
    "perplexity/sonar",
}


# ─────────────────────────────────────────────
# DOMYŚLNE PYTANIA
# ─────────────────────────────────────────────

DEFAULT_PROMPTS_PL = [
    "Jakie są najlepsze agencje marketingu cyfrowego w Polsce?",
    "Polecane agencje Google Ads dla firm B2B",
    "Najlepsze agencje SEO w Polsce 2025",
    "Jak wybrać agencję marketingową dla firmy B2B?",
    "Agencje specjalizujące się w Google Ads i SEO w Polsce",
]

DEFAULT_PROMPTS_EN = [
    "Best digital marketing agencies in Poland",
    "Top Google Ads agencies for B2B companies in Poland",
    "Best SEO agencies in Poland 2025",
    "How to choose a marketing agency for B2B?",
    "Recommended PPC agencies Poland",
]


# ─────────────────────────────────────────────
# SENTIMENT (rule-based, zero kosztów)
# ─────────────────────────────────────────────

POSITIVE_WORDS = [
    "poleca", "polecam", "najlepsz", "świetn", "doskonał", "wyróżnia", "lider",
    "ekspert", "zaufan", "skuteczn", "profesjonaln", "renomowa", "award",
    "top", "best", "leading", "excellent", "recommended", "trusted", "proven",
    "specialist", "premier", "award-winning", "outstanding", "innovative",
]

NEGATIVE_WORDS = [
    "problem", "słab", "zły", "zła", "niski", "nisk", "słaba", "niepoleca",
    "unika", "ostrzeżen", "scam", "fake", "bad", "poor", "avoid", "negative",
    "complaint", "issue", "failure", "unreliable", "disappointing",
]

def analyze_sentiment(text: str, keywords: list[str]) -> str:
    """Prosty sentiment: szukamy pozytywnych/negatywnych słów w otoczeniu keyword."""
    text_lower = text.lower()

    # Znajdź zdania zawierające keyword
    sentences = re.split(r'(?<=[.!?])\s+', text)
    relevant_sentences = []
    for sent in sentences:
        sent_lower = sent.lower()
        if any(kw.lower() in sent_lower for kw in keywords):
            relevant_sentences.append(sent_lower)

    if not relevant_sentences:
        return "neutralny"

    context = " ".join(relevant_sentences)
    pos = sum(1 for w in POSITIVE_WORDS if w in context)
    neg = sum(1 for w in NEGATIVE_WORDS if w in context)

    if pos > neg:
        return "pozytywny"
    elif neg > pos:
        return "negatywny"
    else:
        return "neutralny"


def sentiment_emoji(sentiment: str) -> str:
    return {"pozytywny": "😊", "negatywny": "😟", "neutralny": "😐"}.get(sentiment, "😐")


# ─────────────────────────────────────────────
# WYCIĄGANIE KONTEKSTU WZMIANKI
# ─────────────────────────────────────────────

def extract_mention_contexts(content: str, keywords: list[str], context_sentences: int = 2) -> list[dict]:
    """
    Dla każdego keyword zwróć wszystkie wystąpienia z otaczającymi zdaniami.
    Zwraca listę dict: {keyword, excerpt, sentence_index, total_sentences}
    """
    sentences = re.split(r'(?<=[.!?\n])\s+', content.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    total = len(sentences)
    results = []

    for kw in keywords:
        kw_lower = kw.lower()
        for i, sent in enumerate(sentences):
            if kw_lower in sent.lower():
                start = max(0, i - context_sentences)
                end = min(total, i + context_sentences + 1)
                excerpt = " ".join(sentences[start:end])
                results.append({
                    "keyword": kw,
                    "excerpt": excerpt,
                    "sentence_index": i,
                    "total_sentences": total,
                    "position_pct": round((i / max(total - 1, 1)) * 100),
                })

    return results


def check_mentions(content: str, keywords: list[str]) -> dict:
    """Sprawdź ile razy i gdzie pojawiają się keywords + konteksty."""
    content_lower = content.lower()
    results = {}
    for kw in keywords:
        kw_lower = kw.lower()
        count = content_lower.count(kw_lower)
        if count > 0:
            pos = content_lower.find(kw_lower)
            position_ratio = pos / max(len(content_lower), 1)
            contexts = extract_mention_contexts(content, [kw])
            results[kw] = {
                "count": count,
                "first_position": position_ratio,
                "contexts": contexts,
            }
    return results


def check_citations(annotations: list, domains: list[str]) -> list[dict]:
    found = []
    for annotation in annotations:
        try:
            url = ""
            if hasattr(annotation, "url_citation"):
                url = annotation.url_citation.url.lower()
            elif isinstance(annotation, dict):
                url = annotation.get("url", "").lower()
            if not url:
                continue
            for domain in domains:
                if domain.lower() in url:
                    found.append({"domain": domain, "url": url})
        except Exception:
            pass
    return found


# ─────────────────────────────────────────────
# ZAPYTANIE DO MODELU
# ─────────────────────────────────────────────

async def query_model(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    keywords: list[str],
    domains: list[str],
    semaphore: asyncio.Semaphore,
    location: Optional[str] = None,
) -> dict:
    model_id = model if model in NO_ONLINE_SUFFIX else f"{model}:online"
    result = {
        "model": model,
        "prompt": prompt,
        "location": location,
        "success": False,
        "content": "",
        "mentions": {},
        "mention_contexts": [],
        "sentiment": "neutralny",
        "citations": [],
        "error": None,
        "tokens": 0,
    }

    # Buduj messages z opcjonalną lokalizacją
    messages = []
    if location:
        messages.append({
            "role": "system",
            "content": (
                f"You are a helpful assistant. "
                f"The user is located in {location}. "
                f"Provide answers relevant to this location and local context."
            )
        })
    # Dodaj lokalizację też do pytania dla pewności
    user_prompt = f"{prompt} (location: {location})" if location else prompt
    messages.append({"role": "user", "content": user_prompt})

    async with semaphore:
        try:
            completion = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model_id,
                    stream=False,
                    max_tokens=1500,
                    messages=messages,
                ),
                timeout=45.0,
            )

            content = completion.choices[0].message.content or ""
            result["content"] = content  # pełny tekst, bez obcinania
            result["success"] = True
            result["tokens"] = (
                completion.usage.total_tokens if completion.usage else 0
            )

            result["mentions"] = check_mentions(content, keywords)

            # Konteksty wszystkich wzmianek (płasko)
            all_contexts = []
            for kw_data in result["mentions"].values():
                all_contexts.extend(kw_data.get("contexts", []))
            result["mention_contexts"] = all_contexts

            # Sentiment
            if result["mentions"]:
                result["sentiment"] = analyze_sentiment(content, keywords)

            annotations = getattr(completion.choices[0].message, "annotations", []) or []
            result["citations"] = check_citations(annotations, domains)

        except asyncio.TimeoutError:
            result["error"] = "timeout (45s)"
        except Exception as e:
            result["error"] = str(e)[:200]

    return result


# ─────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────

def compute_model_score(results_for_model: list[dict], keywords: list[str]) -> dict:
    total = len(results_for_model)
    if total == 0:
        return {"score": 0, "mention_rate": 0, "citation_rate": 0, "avg_mentions": 0}

    successful = [r for r in results_for_model if r["success"]]
    if not successful:
        return {"score": 0, "mention_rate": 0, "citation_rate": 0, "avg_mentions": 0, "errors": total}

    with_mention = sum(1 for r in successful if r["mentions"])
    with_citation = sum(1 for r in successful if r["citations"])
    total_mentions = sum(
        sum(v["count"] for v in r["mentions"].values()) for r in successful
    )

    # Sentiment summary
    sentiments = [r["sentiment"] for r in successful if r["mentions"]]
    pos_count = sentiments.count("pozytywny")
    neg_count = sentiments.count("negatywny")

    mention_rate = with_mention / len(successful)
    citation_rate = with_citation / len(successful)
    avg_mentions = total_mentions / len(successful)

    position_scores = []
    for r in successful:
        for kw_data in r["mentions"].values():
            position_scores.append(1.0 - kw_data["first_position"])
    avg_position = sum(position_scores) / len(position_scores) if position_scores else 0

    score = (mention_rate * 50) + (citation_rate * 30) + (avg_position * 20)
    score = round(min(100, score), 1)

    return {
        "score": score,
        "mention_rate": round(mention_rate * 100, 1),
        "citation_rate": round(citation_rate * 100, 1),
        "avg_mentions": round(avg_mentions, 2),
        "avg_position": round(avg_position * 100, 1),
        "successful_queries": len(successful),
        "failed_queries": total - len(successful),
        "sentiment_positive": pos_count,
        "sentiment_negative": neg_count,
        "sentiment_neutral": len(sentiments) - pos_count - neg_count,
    }


def score_to_status(score: float) -> str:
    if score >= 80:
        return "🟢 Silna"
    elif score >= 50:
        return "🟡 Umiarkowana"
    elif score >= 20:
        return "🟠 Słaba"
    else:
        return "🔴 Niewidoczna"


# ─────────────────────────────────────────────
# GENEROWANIE RAPORTU MARKDOWN
# ─────────────────────────────────────────────

def generate_report(
    brand: str,
    domain: str,
    keywords: list[str],
    domains: list[str],
    prompts: list[str],
    all_results: list[dict],
    model_scores: dict,
    run_date: str,
    locations: list[str] = None,
) -> str:
    lines = []
    locations = locations or [None]
    multi_location = len(locations) > 1 or (len(locations) == 1 and locations[0] is not None)

    # Remap scoring keys for single-location mode (keys are "model||None" → "model")
    if not multi_location:
        remapped = {}
        for key, val in model_scores.items():
            clean_key = key.split("||")[0]
            remapped[clean_key] = val
        model_scores = remapped

    lines += [
        f"# Raport AI Visibility — {brand}",
        f"",
        f"**Data:** {run_date}  ",
        f"**Domena:** {domain}  ",
        f"**Keywords:** {', '.join(keywords)}  ",
        f"**Modele:** {len(set(r['model'] for r in all_results))}  ",
        f"**Pytania:** {len(prompts)}  ",
    ]
    if multi_location:
        lines.append(f"**Lokalizacje:** {', '.join(str(l) for l in locations)}  ")
    lines.append("")

    # ── EXECUTIVE SUMMARY ────────────────────────────────────────────────
    all_scores = [v["score"] for v in model_scores.values()]
    overall = round(sum(all_scores) / len(all_scores), 1) if all_scores else 0

    all_sentiments = [r["sentiment"] for r in all_results if r["success"] and r["mentions"]]
    total_s = len(all_sentiments)
    pos_pct = round(all_sentiments.count("pozytywny") / total_s * 100) if total_s else 0
    neg_pct = round(all_sentiments.count("negatywny") / total_s * 100) if total_s else 0
    neu_pct = 100 - pos_pct - neg_pct

    lines += [f"## Executive Summary", f"",
              f"**Overall AI Visibility Score: {overall}/100** — {score_to_status(overall)}", f""]

    if overall >= 80:
        lines.append(f"Marka **{brand}** jest dobrze widoczna w ekosystemie AI.")
    elif overall >= 50:
        lines.append(f"Marka **{brand}** ma umiarkowaną widoczność — pojawia się w części odpowiedzi.")
    elif overall >= 20:
        lines.append(f"Marka **{brand}** jest słabo widoczna w AI. Wymagana strategia GEO.")
    else:
        lines.append(f"Marka **{brand}** praktycznie nie istnieje w odpowiedziach modeli AI.")

    if total_s > 0:
        lines += [f"",
                  f"**Sentiment wzmianek:** 😊 {pos_pct}% pozytywny · 😐 {neu_pct}% neutralny · 😟 {neg_pct}% negatywny  ",
                  f"*(na podstawie {total_s} odpowiedzi z wzmianką)*"]
    lines.append("")

    # ── SCORE PER MODEL (z kolumnami lokalizacji jeśli multi) ────────────
    lines += [f"## Wyniki per model", f""]

    if multi_location:
        loc_headers = " | ".join(f"Score {l}" for l in locations)
        loc_sep = " | ".join("---" for _ in locations)
        lines += [
            f"| Model | {loc_headers} | Avg Score | Sentiment |",
            f"|---| {loc_sep} |---|---|",
        ]
        for model in MODELS:
            model_short = model.split("/")[-1]
            scores_per_loc = []
            for loc in locations:
                key = f"{model}||{loc}"
                s = model_scores.get(key, {}).get("score", "—")
                scores_per_loc.append(f"**{s}**" if s != "—" else "—")
            # avg across locations
            avg_vals = [model_scores.get(f"{model}||{loc}", {}).get("score", 0) for loc in locations]
            avg_score = round(sum(avg_vals) / len(avg_vals), 1)
            # sentiment sum
            s_pos = sum(model_scores.get(f"{model}||{loc}", {}).get("sentiment_positive", 0) for loc in locations)
            s_neg = sum(model_scores.get(f"{model}||{loc}", {}).get("sentiment_negative", 0) for loc in locations)
            s_neu = sum(model_scores.get(f"{model}||{loc}", {}).get("sentiment_neutral", 0) for loc in locations)
            loc_cells = " | ".join(scores_per_loc)
            lines.append(f"| `{model_short}` | {loc_cells} | {avg_score} | 😊{s_pos} 😐{s_neu} 😟{s_neg} |")
    else:
        lines += [
            f"| Model | Score | Status | Mention Rate | Citation Rate | Avg Mentions | Sentiment |",
            f"|---|---|---|---|---|---|---|",
        ]
        for model, stats in sorted(model_scores.items(), key=lambda x: -x[1]["score"]):
            model_short = model.split("/")[-1]
            s_pos = stats.get("sentiment_positive", 0)
            s_neg = stats.get("sentiment_negative", 0)
            s_neu = stats.get("sentiment_neutral", 0)
            lines.append(
                f"| `{model_short}` | **{stats['score']}** | {score_to_status(stats['score'])} "
                f"| {stats['mention_rate']}% | {stats['citation_rate']}% | {stats['avg_mentions']} | 😊{s_pos} 😐{s_neu} 😟{s_neg} |"
            )
    lines.append("")

    # ── SCORE PER PYTANIE ────────────────────────────────────────────────
    lines += [f"## Wyniki per pytanie", f""]
    if multi_location:
        loc_headers = " | ".join(f"Wzmianki {l}" for l in locations)
        loc_sep = " | ".join("---" for _ in locations)
        lines += [
            f"| Pytanie | {loc_headers} |",
            f"|---| {loc_sep} |",
        ]
        for prompt in prompts:
            cells = []
            for loc in locations:
                pr = [r for r in all_results if r["prompt"] == prompt and r.get("location") == loc and r["success"]]
                with_m = sum(1 for r in pr if r["mentions"])
                total_m = len(pr)
                cells.append(f"{with_m}/{total_m}")
            prompt_short = prompt[:55] + ("..." if len(prompt) > 55 else "")
            lines.append(f"| {prompt_short} | {' | '.join(cells)} |")
    else:
        lines += [
            f"| Pytanie | Modele z wzmianką | Modele z cytowaniem | Sentiment |",
            f"|---|---|---|---|",
        ]
        for prompt in prompts:
            pr = [r for r in all_results if r["prompt"] == prompt and r["success"]]
            with_m = sum(1 for r in pr if r["mentions"])
            with_c = sum(1 for r in pr if r["citations"])
            total_m = len(pr)
            sents = [r["sentiment"] for r in pr if r["mentions"]]
            if sents:
                dominant = max(set(sents), key=sents.count)
                sent_str = f"{sentiment_emoji(dominant)} {dominant}"
            else:
                sent_str = "—"
            prompt_short = prompt[:60] + ("..." if len(prompt) > 60 else "")
            lines.append(f"| {prompt_short} | {with_m}/{total_m} | {with_c}/{total_m} | {sent_str} |")
    lines.append("")

    # ── GŁÓWNA SEKCJA: per model → per pytanie → per lokalizacja ─────────
    lines += [
        f"## Pełne odpowiedzi modeli",
        f"",
        f"> Struktura: Model → Pytanie → odpowiedzi per lokalizacja.",
        f"",
    ]

    for model in MODELS:
        model_results = [r for r in all_results if r["model"] == model and r["success"]]
        if not model_results:
            continue

        model_short = model.split("/")[-1]
        if multi_location:
            avg_score = round(
                sum(model_scores.get(f"{model}||{loc}", {}).get("score", 0) for loc in locations) / len(locations), 1
            )
            lines += [f"### `{model_short}` — avg score {avg_score}/100", f""]
        else:
            stats = model_scores.get(model, {})
            lines += [f"### `{model_short}` — score {stats.get('score', 0)}/100", f""]

        for prompt in prompts:
            prompt_short = prompt[:80]
            lines += [f"#### 💬 {prompt_short}", f""]

            if multi_location:
                # Tabela porównawcza lokalizacji
                lines += [
                    f"| Lokalizacja | Wzmianka | Cytat URL | Sentiment | Fragment |",
                    f"|---|---|---|---|---|",
                ]
                for loc in locations:
                    r = next(
                        (x for x in all_results if x["model"] == model
                         and x["prompt"] == prompt
                         and x.get("location") == loc
                         and x["success"]),
                        None
                    )
                    if not r:
                        lines.append(f"| {loc} | ❓ | ❓ | — | *brak danych* |")
                        continue
                    has_m = bool(r["mentions"])
                    has_c = bool(r["citations"])
                    sent = sentiment_emoji(r.get("sentiment", "neutralny"))
                    # Pierwszy cytat (skrócony)
                    fragment = "—"
                    if r["mention_contexts"]:
                        raw = r["mention_contexts"][0]["excerpt"]
                        fragment = raw[:120].replace("|", "\\|") + ("..." if len(raw) > 120 else "")
                    lines.append(
                        f"| **{loc}** | {'✅' if has_m else '❌'} | {'✅' if has_c else '❌'} | {sent} | {fragment} |"
                    )
                lines.append("")

                # Pełne odpowiedzi per lokalizacja (zwinięte)
                for loc in locations:
                    r = next(
                        (x for x in all_results if x["model"] == model
                         and x["prompt"] == prompt
                         and x.get("location") == loc
                         and x["success"]),
                        None
                    )
                    if not r:
                        continue
                    lines += [
                        f"<details>",
                        f"<summary>📄 Pełna odpowiedź — {loc} ({len(r['content'])} znaków)</summary>",
                        f"",
                        f"```",
                        r["content"],
                        f"```",
                        f"",
                        f"</details>",
                        f"",
                    ]
            else:
                # Tryb bez lokalizacji — stary układ
                r = next(
                    (x for x in all_results if x["model"] == model
                     and x["prompt"] == prompt
                     and x["success"]),
                    None
                )
                if not r:
                    continue
                has_mention = bool(r["mentions"])
                sentiment = r.get("sentiment", "neutralny")
                lines += [
                    f"**Wzmianka:** {'✅ TAK' if has_mention else '❌ NIE'} | "
                    f"**Cytat URL:** {'✅ TAK' if r['citations'] else '❌ NIE'} | "
                    f"**Sentiment:** {sentiment_emoji(sentiment)} {sentiment}",
                    f"",
                ]
                if r["mention_contexts"]:
                    lines.append(f"**Cytaty z odpowiedzi** *(kontekst ±2 zdania)*:")
                    lines.append("")
                    for ctx in r["mention_contexts"]:
                        lines += [
                            f"> {ctx['excerpt']}",
                            f"  *(keyword: `{ctx['keyword']}` · pozycja ~{ctx['position_pct']}% tekstu)*",
                            f"",
                        ]
                lines += [
                    f"<details>",
                    f"<summary>📄 Pełna odpowiedź ({len(r['content'])} znaków)</summary>",
                    f"",
                    f"```",
                    r["content"],
                    f"```",
                    f"",
                    f"</details>",
                    f"",
                ]
                if r["citations"]:
                    lines.append(f"**Cytowane URL-e:**")
                    for c in r["citations"]:
                        lines.append(f"- {c['url']}")
                    lines.append("")

            lines.append("---")
            lines.append("")

    # ── NAJCZĘSTSZE CYTOWANIA ─────────────────────────────────────────────
    from collections import Counter
    all_citations = [c["url"] for r in all_results for c in r.get("citations", [])]
    if all_citations:
        lines += [f"## Najczęściej cytowane URL-e", f""]
        for url, count in Counter(all_citations).most_common(10):
            lines.append(f"- `{url}` ({count}x)")
        lines.append("")

    # ── BŁĘDY ─────────────────────────────────────────────────────────────
    failed = [r for r in all_results if not r["success"]]
    if failed:
        lines += [f"## Błędy (modele niedostępne)", f""]
        errors_by_model: dict = {}
        for r in failed:
            errors_by_model.setdefault(r["model"], r["error"])
        for model, err in errors_by_model.items():
            lines.append(f"- `{model}`: {err}")
        lines.append("")

    # ── REKOMENDACJE GEO ──────────────────────────────────────────────────
    lines += [f"## Rekomendacje GEO", f""]
    if overall < 50:
        lines += [
            f"### 🔴 Pilne działania", f"",
            f"1. **Treści odpowiadające na pytania** — content bezpośrednio pod pytania które modele AI dostają",
            f"2. **Wikipedia / Wikidata** — jedno z najczęściej cytowanych źródeł przez LLM",
            f"3. **LinkedIn / firmowe profile** — wiarygodne źródła indeksowane przez modele",
            f"4. **Recenzje i case studies** — G2, Clutch, Trustpilot",
            f"5. **Schema markup** — FAQPage, Organization, breadcrumbs",
            f"",
        ]
    else:
        lines += [
            f"### 🟡 Optymalizacja", f"",
            f"1. **Zwiększ pokrycie tematyczne** — odpowiadaj na pytania gdzie nie ma jeszcze wzmianki",
            f"2. **Buduj cytowania** — publikuj dane, statystyki, raporty branżowe",
            f"3. **Reddit / Quora / fora** — autentyczne zaangażowanie w dyskusje",
            f"4. **robots.txt** — sprawdź czy nie blokujesz GPTBot/ClaudeBot/PerplexityBot",
            f"5. **Monitoring miesięczny** — uruchamiaj audyt regularnie",
            f"",
        ]

    # ── METODOLOGIA ───────────────────────────────────────────────────────
    total_queries = len(prompts) * len(set(r["model"] for r in all_results)) * len(locations)
    lines += [
        f"## Metodologia", f"",
        f"- Zapytania: {len(prompts)} pytań × {len(set(r['model'] for r in all_results))} modeli"
        + (f" × {len(locations)} lokalizacje = {total_queries} kombinacji" if multi_location else f" = {len(prompts) * len(set(r['model'] for r in all_results))} kombinacji"),
        f"- Lokalizacja symulowana przez system prompt + sufiks pytania",
        f"- Modele odpytywane z web searchem (`:online` suffix OpenRouter)",
        f"- Mention detection: case-insensitive, z kontekstem ±2 zdania",
        f"- Sentiment: rule-based (słownik PL+EN), per odpowiedź",
        f"- Citation detection: dopasowanie domeny w URL-ach anotacji",
        f"- Scoring: 50% mention_rate + 30% citation_rate + 20% position_score",
        f"- API: OpenRouter (https://openrouter.ai)",
        f"",
        f"---",
        f"*Wygenerowano przez AI Visibility Skill v2.1 dla Claude Code — {run_date}*",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────
# MAIN (bez zmian w logice)
# ─────────────────────────────────────────────

async def run_audit(
    brand: str,
    domain: str,
    keywords: list[str],
    domains: list[str],
    prompts: list[str],
    models: list[str],
    api_key: str,
    locations: list[str] = None,
    out_json: Optional[str] = None,
    out_md: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    client = get_client(api_key)
    run_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    semaphore = asyncio.Semaphore(8)
    locations = locations or [None]

    total = len(models) * len(prompts) * len(locations)
    if verbose:
        print(f"\n🔍 AI Visibility Audit v2.1 — {brand}")
        print(f"   Domena: {domain}")
        if any(l for l in locations):
            print(f"   Lokalizacje: {', '.join(str(l) for l in locations)}")
        print(f"   Modele: {len(models)} | Pytania: {len(prompts)} | Lokalizacje: {len(locations)} | Razem: {total} zapytań")
        print(f"   Keywords: {', '.join(keywords)}\n")

    tasks = [
        query_model(client, model, prompt, keywords, domains, semaphore, location)
        for model in models
        for prompt in prompts
        for location in locations
    ]

    if verbose:
        print(f"⏳ Uruchamianie {total} zapytań równolegle...")

    all_results = await asyncio.gather(*tasks)
    all_results = list(all_results)

    if verbose:
        successful = sum(1 for r in all_results if r["success"])
        print(f"✅ Zakończono: {successful}/{total} zapytań udanych\n")

    # Scoring per model+location (klucz: "model||location")
    model_scores = {}
    for model in models:
        for loc in locations:
            key = f"{model}||{loc}"
            loc_results = [r for r in all_results if r["model"] == model and r.get("location") == loc]
            model_scores[key] = compute_model_score(loc_results, keywords)

    if verbose:
        print("📊 Wyniki per model:")
        for model in models:
            model_short = model.split("/")[-1]
            scores = [model_scores.get(f"{model}||{loc}", {}).get("score", 0) for loc in locations]
            avg = round(sum(scores) / len(scores), 1)
            loc_str = "  ".join(f"{str(loc)[:12]}={s}" for loc, s in zip(locations, scores)) if len(locations) > 1 else f"score={scores[0]}"
            print(f"   {model_short:30s} avg={avg:5.1f}  {loc_str}")
        print()

    report_md = generate_report(
        brand, domain, keywords, domains, prompts,
        all_results, model_scores, run_date,
        locations=locations,
    )

    if out_md is None:
        safe_domain = re.sub(r"[^\w.-]", "_", domain)
        date_str = datetime.now().strftime("%Y-%m-%d")
        out_md = f"{safe_domain}-ai-visibility-{date_str}.md"

    with open(out_md, "w", encoding="utf-8") as f:
        f.write(report_md)

    if verbose:
        print(f"📄 Raport markdown: {out_md}")

    output_data = {
        "meta": {
            "brand": brand,
            "domain": domain,
            "keywords": keywords,
            "domains": domains,
            "prompts": prompts,
            "models": models,
            "locations": locations,
            "run_date": run_date,
            "version": "2.1",
        },
        "model_scores": model_scores,
        "results": all_results,
    }

    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        if verbose:
            print(f"📦 Surowe dane JSON: {out_json}")

    return output_data


def get_client(api_key: str) -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "https://7p.marketing",
            "X-Title": "AI Visibility Checker",
        },
    )


def main():
    parser = argparse.ArgumentParser(
        description="AI Visibility Check v2 — audyt widoczności marki w modelach AI"
    )
    parser.add_argument("--brand", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--keywords", nargs="+")
    parser.add_argument("--domains", nargs="+")
    parser.add_argument("--prompts", nargs="+")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--lang", default="pl", choices=["pl", "en"])
    parser.add_argument("--out")
    parser.add_argument("--out-json")
    parser.add_argument("--api-key")
    parser.add_argument("--locations", nargs="+", help="Lokalizacje, np. 'Wroclaw, Poland' 'Warsaw, Poland' 'Berlin, Germany'")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Brak OPENROUTER_API_KEY.")
        sys.exit(1)

    keywords = [args.brand]
    if args.keywords:
        keywords.extend(args.keywords)
    clean_domain = args.domain.replace("www.", "").replace("https://", "").replace("http://", "").rstrip("/")
    keywords.append(clean_domain)
    keywords = list(dict.fromkeys(keywords))

    domains = [args.domain, clean_domain]
    if args.domains:
        domains.extend(args.domains)
    domains = list(dict.fromkeys(domains))

    prompts = args.prompts if args.prompts else (DEFAULT_PROMPTS_PL if args.lang == "pl" else DEFAULT_PROMPTS_EN)
    models = args.models if args.models else MODELS

    asyncio.run(
        run_audit(
            brand=args.brand,
            domain=args.domain,
            keywords=keywords,
            domains=domains,
            prompts=prompts,
            models=models,
            api_key=api_key,
            locations=args.locations,
            out_json=args.out_json,
            out_md=args.out,
            verbose=not args.quiet,
        )
    )


if __name__ == "__main__":
    main()
