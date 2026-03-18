#!/usr/bin/env python3
"""
AI Visibility Check v2 — audit brand visibility across AI models (GEO/AEO)
Queries multiple models in parallel via OpenRouter API.
Version 2: full responses, mention context, sentiment analysis, position scoring.

Usage:
    python3 ai_visibility_check.py --brand "name" --domain "domain.com"
    python3 ai_visibility_check.py --brand "name" --domain "domain.com" --lang en
    python3 ai_visibility_check.py --brand "name" --domain "domain.com" \
        --prompts "question 1" "question 2"
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
    print("ERROR: Missing library openai. Install: pip install openai")
    sys.exit(1)


# ─────────────────────────────────────────────
# MODEL CONFIGURATION
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
# DEFAULT PROMPTS
# ─────────────────────────────────────────────

DEFAULT_PROMPTS_PL = [  # Polish default prompts (used with --lang pl)
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
# SENTIMENT (rule-based, zero cost)
# ─────────────────────────────────────────────

POSITIVE_WORDS = [
    "poleca", "polecam", "najlepsz", "świetn", "doskonał", "wyróżnia", "lider",  # Polish
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
    """Simple sentiment: find positive/negative words around keyword mentions."""
    text_lower = text.lower()

    # Find sentences containing keyword
    sentences = re.split(r'(?<=[.!?])\s+', text)
    relevant_sentences = []
    for sent in sentences:
        sent_lower = sent.lower()
        if any(kw.lower() in sent_lower for kw in keywords):
            relevant_sentences.append(sent_lower)

    if not relevant_sentences:
        return "neutral"

    context = " ".join(relevant_sentences)
    pos = sum(1 for w in POSITIVE_WORDS if w in context)
    neg = sum(1 for w in NEGATIVE_WORDS if w in context)

    if pos > neg:
        return "positive"
    elif neg > pos:
        return "negative"
    else:
        return "neutral"


def sentiment_emoji(sentiment: str) -> str:
    return {"positive": "😊", "negative": "😟", "neutral": "😐"}.get(sentiment, "😐")


# ─────────────────────────────────────────────
# MENTION CONTEXT EXTRACTION
# ─────────────────────────────────────────────

def extract_mention_contexts(content: str, keywords: list[str], context_sentences: int = 2) -> list[dict]:
    """
    For each keyword return all occurrences with surrounding sentences.
    Returns list of dict: {keyword, excerpt, sentence_index, total_sentences}
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
    """Check how many times and where keywords appear, including context."""
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
# MODEL QUERY
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
        "sentiment": "neutral",
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
            result["content"] = content  # full text, no truncation
            result["success"] = True
            result["tokens"] = (
                completion.usage.total_tokens if completion.usage else 0
            )

            result["mentions"] = check_mentions(content, keywords)

            # All mention contexts (flat list)
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
    pos_count = sentiments.count("positive")
    neg_count = sentiments.count("negative")

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
        return "🟢 Strong"
    elif score >= 50:
        return "🟡 Moderate"
    elif score >= 20:
        return "🟠 Weak"
    else:
        return "🔴 Invisible"


# ─────────────────────────────────────────────
# MARKDOWN REPORT GENERATION
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
        f"**Date:** {run_date}  ",
        f"**Domain:** {domain}  ",
        f"**Keywords:** {', '.join(keywords)}  ",
        f"**Models:** {len(model_scores)}  ",
        f"**Prompts:** {len(prompts)}  ",
        f"",
    ]

    all_scores = [v["score"] for v in model_scores.values()]
    overall = round(sum(all_scores) / len(all_scores), 1) if all_scores else 0

    # Global sentiment
    all_sentiments = [r["sentiment"] for r in all_results if r["success"] and r["mentions"]]
    total_s = len(all_sentiments)
    pos_pct = round(all_sentiments.count("positive") / total_s * 100) if total_s else 0
    neg_pct = round(all_sentiments.count("negative") / total_s * 100) if total_s else 0
    neu_pct = 100 - pos_pct - neg_pct

    lines += [
        f"## Executive Summary",
        f"",
        f"**Overall AI Visibility Score: {overall}/100** — {score_to_status(overall)}",
        f"",
    ]

    if overall >= 80:
        lines.append(f"Brand **{brand}** has strong visibility across the AI ecosystem.")
    elif overall >= 50:
        lines.append(f"Brand **{brand}** has moderate visibility — appears in some responses but lacks dominance.")
    elif overall >= 20:
        lines.append(f"Brand **{brand}** has weak AI visibility. An active GEO strategy is required.")
    else:
        lines.append(f"Brand **{brand}** is virtually absent from AI model responses. Critical GEO intervention needed.")

    if total_s > 0:
        lines += [
            f"",
            f"**Mention sentiment:** 😊 {pos_pct}% positive · 😐 {neu_pct}% neutral · 😟 {neg_pct}% negative  ",
            f"*(based on {total_s} responses with mentions)*",
        ]

    lines.append("")

    # Results table per model
    lines += [
        f"## Results per model",
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

    # Results table per prompt
    lines += [
        f"## Results per prompt",
        f"",
        f"| Prompt | Models with mention | Models with citation | Sentiment |",
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

    # ── MAIN SECTION: full responses per model ─────────────────────────
    lines += [
        f"## Full model responses",
        f"",
        f"> All responses with brand mentions. Excerpt context highlighted.",
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
            sentiment = r.get("sentiment", "neutral")
            prompt_short = r["prompt"][:80]

            lines += [
                f"**Prompt:** {prompt_short}  ",
                f"**Mention:** {'✅ YES' if has_mention else '❌ NO'} | "
                f"**URL citation:** {'✅ YES' if r['citations'] else '❌ NO'} | "
                f"**Sentiment:** {sentiment_emoji(sentiment)} {sentiment}",
                f"",
            ]

            # Mention contexts (all occurrences)
            if r["mention_contexts"]:
                lines.append(f"**Mention excerpts** *(±2 sentence context)*:")
                lines.append("")
                for ctx in r["mention_contexts"]:
                    pos_label = f"pozycja ~{ctx['position_pct']}% tekstu"
                    lines += [
                        f"> {ctx['excerpt']}",
                        f"  *(keyword: `{ctx['keyword']}` · {pos_label})*",
                        f"",
                    ]

            # Full response (collapsed via HTML details)
            lines += [
                f"<details>",
                f"<summary>📄 Full response ({len(r['content'])} characters)</summary>",
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
                lines.append(f"**Cited URLs:**")
                for c in r["citations"]:
                    lines.append(f"- {c['url']}")
                lines.append("")

            lines.append("---")
            lines.append("")

    # Most cited URLs
    all_citations = []
    for r in all_results:
        for c in r.get("citations", []):
            all_citations.append(c["url"])

    if all_citations:
        from collections import Counter
        top_urls = Counter(all_citations).most_common(10)
        lines += [f"## Most cited URLs", f""]
        for url, count in top_urls:
            lines.append(f"- `{url}` ({count}x)")
        lines.append("")

    # Errors
    failed = [r for r in all_results if not r["success"]]
    if failed:
        lines += [f"## Errors (unavailable models)", f""]
        errors_by_model: dict = {}
        for r in failed:
            errors_by_model.setdefault(r["model"], r["error"])
        for model, err in errors_by_model.items():
            lines.append(f"- `{model}`: {err}")
        lines.append("")

    # GEO Recommendations
    lines += [f"## GEO Recommendations", f""]

    if overall < 50:
        lines += [
            f"### 🔴 Urgent actions",
            f"",
            f"1. **Answer-focused content** — create content that directly answers questions AI models receive from users",
            f"2. **Wikipedia / Wikidata** — one of the most cited sources by LLMs",
            f"3. **LinkedIn / company profiles** — trusted sources indexed by AI models",
            f"4. **Reviews and case studies** — G2, Clutch, Trustpilot",
            f"5. **Schema markup** — FAQPage, Organization, breadcrumbs",
            f"",
        ]
    else:
        lines += [
            f"### 🟡 Optimization",
            f"",
            f"1. **Expand topic coverage** — target queries where your brand has no mentions yet",
            f"2. **Build citations** — publish data, statistics, and industry reports",
            f"3. **Reddit / Quora / forums** — authentic engagement in niche discussions",
            f"4. **robots.txt** — ensure you are not blocking GPTBot/ClaudeBot/PerplexityBot",
            f"5. **Monthly monitoring** — run audits regularly to track progress",
            f"",
        ]

    best_model_entry = max(model_scores.items(), key=lambda x: x[1]["score"]) if model_scores else None
    worst_model_entry = min(model_scores.items(), key=lambda x: x[1]["score"]) if model_scores else None

    if best_model_entry and worst_model_entry:
        lines += [
            f"### Model priorities",
            f"",
            f"- **Best visibility:** `{best_model_entry[0].split('/')[-1]}` ({best_model_entry[1]['score']}/100)",
            f"- **Worst visibility:** `{worst_model_entry[0].split('/')[-1]}` ({worst_model_entry[1]['score']}/100)",
            f"",
        ]

    lines += [
        f"## Methodology",
        f"",
        f"- Queries: {len(prompts)} prompts × {len(model_scores)} models = {len(prompts) * len(model_scores)} combinations",
        f"- All models queried with web search enabled (`:online` suffix via OpenRouter)",
        f"- Mention detection: case-insensitive, with ±2 sentence context",
        f"- Sentiment: rule-based (PL+EN dictionary), per response",
        f"- Citation detection: domain match in annotation URLs",
        f"- Scoring: 50% mention_rate + 30% citation_rate + 20% position_score",
        f"- API: OpenRouter (https://openrouter.ai)",
        f"",
        f"---",
        f"*Generated by AI Visibility Skill v2 for Claude Code — {run_date}*",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────
# MAIN
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
        print(f"   Domain: {domain}")
        print(f"   Models: {len(models)} | Prompts: {len(prompts)} | Total: {total} queries")
        print(f"   Keywords: {', '.join(keywords)}\n")

    tasks = [
        query_model(client, model, prompt, keywords, domains, semaphore)
        for model in models
        for prompt in prompts
    ]

    if verbose:
        print(f"⏳ Running {total} queries in parallel...")

    all_results = await asyncio.gather(*tasks)
    all_results = list(all_results)

    if verbose:
        successful = sum(1 for r in all_results if r["success"])
        print(f"✅ Done: {successful}/{total} queries successful\n")

    model_scores = {}
    for model in models:
        model_results = [r for r in all_results if r["model"] == model]
        model_scores[model] = compute_model_score(model_results, keywords)

    if verbose:
        print("📊 Results per model:")
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
        print(f"📄 Markdown report: {out_md}")

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
            print(f"📦 Raw JSON data: {out_json}")

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
        description="AI Visibility Check v2 — audit brand visibility across AI models (GEO/AEO)"
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
        print("ERROR: Missing OPENROUTER_API_KEY.")
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
