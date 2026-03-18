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
) -> dict:
    model_id = model if model in NO_ONLINE_SUFFIX else f"{model}:online"
    result = {
        "model": model,
        "prompt": prompt,
        "success": False,
        "content": "",
        "mentions": {},
        "mention_contexts": [],
        "sentiment": "neutralny",
        "citations": [],
        "error": None,
        "tokens": 0,
    }

    async with semaphore:
        try:
            completion = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model_id,
                    stream=False,
                    max_tokens=1500,
                    messages=[{"role": "user", "content": prompt}],
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
) -> str:
    lines = []

    lines += [
        f"# Raport AI Visibility — {brand}",
        f"",
        f"**Data:** {run_date}  ",
        f"**Domena:** {domain}  ",
        f"**Keywords:** {', '.join(keywords)}  ",
        f"**Modele:** {len(model_scores)}  ",
        f"**Pytania:** {len(prompts)}  ",
        f"",
    ]

    all_scores = [v["score"] for v in model_scores.values()]
    overall = round(sum(all_scores) / len(all_scores), 1) if all_scores else 0

    # Globalny sentiment
    all_sentiments = [r["sentiment"] for r in all_results if r["success"] and r["mentions"]]
    total_s = len(all_sentiments)
    pos_pct = round(all_sentiments.count("pozytywny") / total_s * 100) if total_s else 0
    neg_pct = round(all_sentiments.count("negatywny") / total_s * 100) if total_s else 0
    neu_pct = 100 - pos_pct - neg_pct

    lines += [
        f"## Executive Summary",
        f"",
        f"**Overall AI Visibility Score: {overall}/100** — {score_to_status(overall)}",
        f"",
    ]

    if overall >= 80:
        lines.append(f"Marka **{brand}** jest dobrze widoczna w ekosystemie AI.")
    elif overall >= 50:
        lines.append(f"Marka **{brand}** ma umiarkowaną widoczność — pojawia się w części odpowiedzi.")
    elif overall >= 20:
        lines.append(f"Marka **{brand}** jest słabo widoczna w AI. Wymagana strategia GEO.")
    else:
        lines.append(f"Marka **{brand}** praktycznie nie istnieje w odpowiedziach modeli AI.")

    if total_s > 0:
        lines += [
            f"",
            f"**Sentiment wzmianek:** 😊 {pos_pct}% pozytywny · 😐 {neu_pct}% neutralny · 😟 {neg_pct}% negatywny  ",
            f"*(na podstawie {total_s} odpowiedzi z wzmianką)*",
        ]

    lines.append("")

    # Tabela wyników per model
    lines += [
        f"## Wyniki per model",
        f"",
        f"| Model | Score | Status | Mention Rate | Citation Rate | Avg Mentions | Sentiment |",
        f"|---|---|---|---|---|---|---|",
    ]
    for model, stats in sorted(model_scores.items(), key=lambda x: -x[1]["score"]):
        model_short = model.split("/")[-1]
        s_pos = stats.get("sentiment_positive", 0)
        s_neg = stats.get("sentiment_negative", 0)
        s_neu = stats.get("sentiment_neutral", 0)
        sentiment_str = f"😊{s_pos} 😐{s_neu} 😟{s_neg}"
        lines.append(
            f"| `{model_short}` | **{stats['score']}** | {score_to_status(stats['score'])} "
            f"| {stats['mention_rate']}% | {stats['citation_rate']}% | {stats['avg_mentions']} | {sentiment_str} |"
        )
    lines.append("")

    # Tabela wyników per prompt
    lines += [
        f"## Wyniki per pytanie",
        f"",
        f"| Pytanie | Modele z wzmianką | Modele z cytowaniem | Sentiment |",
        f"|---|---|---|---|",
    ]
    for prompt in prompts:
        prompt_results = [r for r in all_results if r["prompt"] == prompt and r["success"]]
        with_m = sum(1 for r in prompt_results if r["mentions"])
        with_c = sum(1 for r in prompt_results if r["citations"])
        total_m = len(prompt_results)
        sentiments_p = [r["sentiment"] for r in prompt_results if r["mentions"]]
        if sentiments_p:
            dominant = max(set(sentiments_p), key=sentiments_p.count)
            sent_str = f"{sentiment_emoji(dominant)} {dominant}"
        else:
            sent_str = "—"
        prompt_short = prompt[:60] + ("..." if len(prompt) > 60 else "")
        lines.append(f"| {prompt_short} | {with_m}/{total_m} | {with_c}/{total_m} | {sent_str} |")
    lines.append("")

    # ── SEKCJA GŁÓWNA: pełne odpowiedzi per model ──────────────────────────
    lines += [
        f"## Pełne odpowiedzi modeli",
        f"",
        f"> Poniżej wszystkie odpowiedzi z wzmiankami. Konteksty cytatów wyróżnione.",
        f"",
    ]

    for model in MODELS:
        model_results = [r for r in all_results if r["model"] == model and r["success"]]
        if not model_results:
            continue

        model_short = model.split("/")[-1]
        stats = model_scores.get(model, {})
        lines += [
            f"### `{model_short}` — score {stats.get('score', 0)}/100",
            f"",
        ]

        for r in model_results:
            has_mention = bool(r["mentions"])
            sentiment = r.get("sentiment", "neutralny")
            prompt_short = r["prompt"][:80]

            lines += [
                f"**Pytanie:** {prompt_short}  ",
                f"**Wzmianka:** {'✅ TAK' if has_mention else '❌ NIE'} | "
                f"**Cytat URL:** {'✅ TAK' if r['citations'] else '❌ NIE'} | "
                f"**Sentiment:** {sentiment_emoji(sentiment)} {sentiment}",
                f"",
            ]

            # Konteksty wzmianek (wszystkie wystąpienia)
            if r["mention_contexts"]:
                lines.append(f"**Cytaty z odpowiedzi** *(kontekst {2} zdań przed/po)*:")
                lines.append("")
                for ctx in r["mention_contexts"]:
                    pos_label = f"pozycja ~{ctx['position_pct']}% tekstu"
                    lines += [
                        f"> {ctx['excerpt']}",
                        f"  *(keyword: `{ctx['keyword']}` · {pos_label})*",
                        f"",
                    ]

            # Pełna odpowiedź (zwinięta przez HTML details)
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

            # Cytowane URL-e
            if r["citations"]:
                lines.append(f"**Cytowane URL-e:**")
                for c in r["citations"]:
                    lines.append(f"- {c['url']}")
                lines.append("")

            lines.append("---")
            lines.append("")

    # Najczęstsze cytowania
    all_citations = []
    for r in all_results:
        for c in r.get("citations", []):
            all_citations.append(c["url"])

    if all_citations:
        from collections import Counter
        top_urls = Counter(all_citations).most_common(10)
        lines += [f"## Najczęściej cytowane URL-e", f""]
        for url, count in top_urls:
            lines.append(f"- `{url}` ({count}x)")
        lines.append("")

    # Błędy
    failed = [r for r in all_results if not r["success"]]
    if failed:
        lines += [f"## Błędy (modele niedostępne)", f""]
        errors_by_model: dict = {}
        for r in failed:
            errors_by_model.setdefault(r["model"], r["error"])
        for model, err in errors_by_model.items():
            lines.append(f"- `{model}`: {err}")
        lines.append("")

    # Rekomendacje GEO
    lines += [f"## Rekomendacje GEO", f""]

    if overall < 50:
        lines += [
            f"### 🔴 Pilne działania",
            f"",
            f"1. **Treści odpowiadające na pytania** — content bezpośrednio pod pytania, które modele AI dostają",
            f"2. **Wikipedia / Wikidata** — jedno z najczęściej cytowanych źródeł przez LLM",
            f"3. **LinkedIn / firmowe profile** — wiarygodne źródła indeksowane przez modele",
            f"4. **Recenzje i case studies** — G2, Clutch, Trustpilot",
            f"5. **Schema markup** — FAQPage, Organization, breadcrumbs",
            f"",
        ]
    else:
        lines += [
            f"### 🟡 Optymalizacja",
            f"",
            f"1. **Zwiększ pokrycie tematyczne** — odpowiadaj na pytania gdzie jeszcze nie ma wzmianki",
            f"2. **Buduj cytowania** — publikuj dane, statystyki, raporty branżowe",
            f"3. **Reddit / Quora / fora** — autentyczne zaangażowanie w dyskusje",
            f"4. **robots.txt** — sprawdź czy nie blokujesz GPTBot/ClaudeBot/PerplexityBot",
            f"5. **Monitoring miesięczny** — uruchamiaj audyt regularnie",
            f"",
        ]

    best_model_entry = max(model_scores.items(), key=lambda x: x[1]["score"]) if model_scores else None
    worst_model_entry = min(model_scores.items(), key=lambda x: x[1]["score"]) if model_scores else None

    if best_model_entry and worst_model_entry:
        lines += [
            f"### Priorytety per model",
            f"",
            f"- **Najlepsza widoczność:** `{best_model_entry[0].split('/')[-1]}` ({best_model_entry[1]['score']}/100)",
            f"- **Najgorsza widoczność:** `{worst_model_entry[0].split('/')[-1]}` ({worst_model_entry[1]['score']}/100)",
            f"",
        ]

    lines += [
        f"## Metodologia",
        f"",
        f"- Zapytania: {len(prompts)} pytań × {len(model_scores)} modeli = {len(prompts) * len(model_scores)} kombinacji",
        f"- Modele odpytywane z web searchem (`:online` suffix OpenRouter)",
        f"- Mention detection: case-insensitive, z kontekstem ±2 zdania",
        f"- Sentiment: rule-based (słownik PL+EN), per odpowiedź",
        f"- Citation detection: dopasowanie domeny w URL-ach anotacji",
        f"- Scoring: 50% mention_rate + 30% citation_rate + 20% position_score",
        f"- API: OpenRouter (https://openrouter.ai)",
        f"",
        f"---",
        f"*Wygenerowano przez AI Visibility Skill v2 dla Claude Code — {run_date}*",
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
    out_json: Optional[str] = None,
    out_md: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    client = get_client(api_key)
    run_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    semaphore = asyncio.Semaphore(8)

    total = len(models) * len(prompts)
    if verbose:
        print(f"\n🔍 AI Visibility Audit v2 — {brand}")
        print(f"   Domena: {domain}")
        print(f"   Modele: {len(models)} | Pytania: {len(prompts)} | Razem: {total} zapytań")
        print(f"   Keywords: {', '.join(keywords)}\n")

    tasks = [
        query_model(client, model, prompt, keywords, domains, semaphore)
        for model in models
        for prompt in prompts
    ]

    if verbose:
        print(f"⏳ Uruchamianie {total} zapytań równolegle...")

    all_results = await asyncio.gather(*tasks)
    all_results = list(all_results)

    if verbose:
        successful = sum(1 for r in all_results if r["success"])
        print(f"✅ Zakończono: {successful}/{total} zapytań udanych\n")

    model_scores = {}
    for model in models:
        model_results = [r for r in all_results if r["model"] == model]
        model_scores[model] = compute_model_score(model_results, keywords)

    if verbose:
        print("📊 Wyniki per model:")
        for model, stats in sorted(model_scores.items(), key=lambda x: -x[1]["score"]):
            s_pos = stats.get("sentiment_positive", 0)
            s_neg = stats.get("sentiment_negative", 0)
            print(
                f"   {model.split('/')[-1]:30s} "
                f"score={stats['score']:5.1f}  "
                f"mentions={stats['mention_rate']:5.1f}%  "
                f"citations={stats['citation_rate']:5.1f}%  "
                f"sentiment=😊{s_pos}😟{s_neg}"
            )
        print()

    report_md = generate_report(
        brand, domain, keywords, domains, prompts,
        all_results, model_scores, run_date
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
            "run_date": run_date,
            "version": "2.0",
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
            out_json=args.out_json,
            out_md=args.out,
            verbose=not args.quiet,
        )
    )


if __name__ == "__main__":
    main()
