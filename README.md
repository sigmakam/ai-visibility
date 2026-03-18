# ai-visibility
Claude Code skill — audit your brand's visibility across AI models (GEO/AEO) via OpenRouter

# AI Visibility Skill for Claude Code
Check if your brand is mentioned in AI model responses.
Queries 10 models in parallel via OpenRouter and generates a markdown report.

## Models
GPT-4o, Perplexity Sonar Pro, Gemini 2.5, Claude Sonnet, Grok, Llama, Mistral and more.

## Installation
```bash
mkdir -p ~/.claude/skills/ai-visibility/scripts
cp SKILL.md ~/.claude/skills/ai-visibility/SKILL.md
cp scripts/ai_visibility_check.py ~/.claude/skills/ai-visibility/scripts/
pip3 install --break-system-packages openai
` ` `

## Usage
` ` `bash
export OPENROUTER_API_KEY="sk-or-..."
python3 ~/.claude/skills/ai-visibility/scripts/ai_visibility_check.py \
  --brand "Your Brand" --domain "yourdomain.com" --lang en
` ` `

## Requirements
- Python 3.8+
- OpenRouter account with credits (openrouter.ai)

## Output
Generates `{domain}-ai-visibility-YYYY-MM-DD.md` with per-model scoring (0–100),
mention rate, citation rate, and GEO recommendations.

## Example Output
See an example output report for McKinsey
https://docs.google.com/document/d/1Kpc2qq__TjnTUKnth_WfHEztsstuzrcPm9w11Q3OToI/edit?tab=t.0#heading=h.qb3bvtt2ri0f
---

## Credits
Built by [@sigmakam](https://github.com/sigmakam) with [Claude Sonnet](https://claude.ai) (Anthropic).  
Prompts, architecture & requirements: sigmakam · Code generation: Claude AI
```

