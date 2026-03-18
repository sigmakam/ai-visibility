---
name: ai-visibility
description: >
  AI brand visibility audit (GEO/AEO). Queries multiple AI models in parallel
  via OpenRouter API and checks if the brand/domain is mentioned in responses
  and cited in sources. Generates a markdown report with scoring, tables and GEO recommendations.
---

# AI Visibility Skill

Checks brand/domain visibility in AI model responses, querying them
in parallel via OpenRouter with web search enabled (:online).

## IMPORTANT: Workflow before running

**ALWAYS ask the user for missing information before running the audit:**

1. If --locations not provided → ask: "Which locations do you want to check visibility from? (e.g. New York, London, Tokyo) Provide 1-3 locations or type 'no location'."
2. If --prompts not provided → ask: "What questions should I send to the models? Provide 3-5 questions or I'll use the defaults for the selected language."
3. DO NOT run the script until you have answers to the above.

## Commands

/ai-visibility <brand> <domain> [--locations "loc1" "loc2"] [--prompts "question1" "question2"]

## Parameters

| Parameter     | Description                            | Example                        |
|---------------|----------------------------------------|--------------------------------|
| brand         | Brand name (used as keyword)           | McKinsey                       |
| domain        | Domain to track in citations           | mckinsey.com                   |
| --locations   | Locations to compare (1-3)             | "New York, USA" "London, UK"   |
| --prompts     | Custom questions (optional)            | "best strategy consulting firm"|
| --models      | Override model list (optional)         | "openai/gpt-4o"                |
| --out         | Output report file path                | report-2026-03-18.md           |
| --lang        | Question language (pl/en, default: pl) | en                             |

## Models covered

10 models with web search (:online): GPT-4o, GPT-4o-mini, Perplexity Sonar Pro/Sonar,
Claude Sonnet 4, Gemini 2.0 Flash/2.5 Pro, Mistral Large, Grok 3 Mini, Llama 4 Maverick.

## Report format (multi-location)

For each model and question, a comparative location table is generated:

### gpt-4o — score X/100

#### Question text

| Location        | Mention | URL Citation | Sentiment | Excerpt                     |
|-----------------|---------|--------------|-----------|-----------------------------|
| New York, USA   | YES     | YES          | positive  | "McKinsey leads in..."      |
| London, UK      | NO      | NO           | neutral   | —                           |
| Tokyo, Japan    | YES     | NO           | neutral   | "McKinsey is a top firm..." |

## Visibility Score: 0-100 per model per location

| Score  | Status   |
|--------|----------|
| 80-100 | Strong   |
| 50-79  | Moderate |
| 20-49  | Weak     |
| 0-19   | Invisible|
