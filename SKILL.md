---
name: ai-visibility
description: >
  Audyt widoczności marki w modelach AI (GEO/AEO). Odpytuje równolegle
  wiele modeli przez OpenRouter API i sprawdza czy marka/domena jest
  wymieniana w odpowiedziach oraz cytowana w źródłach. Generuje raport
  markdown z tabelą wyników, scoringiem i rekomendacjami.
---

# AI Visibility Skill

Sprawdza widoczność marki/domeny w odpowiedziach modeli AI, odpytując
je równolegle przez OpenRouter z włączonym web searchem (`:online`).

## Komendy

```
/ai-visibility <brand> <domain> [--prompts "pytanie1" "pytanie2"]
/ai-visibility check <brand> <domain>
/ai-visibility report <brand> <domain> --out raport.md
```

## Parametry

| Parametr | Opis | Przykład |
|---|---|---|
| `brand` | Nazwa marki (keywords do szukania) | `7p.marketing` |
| `domain` | Domena do szukania w cytowaniach | `7p.marketing` |
| `--prompts` | Własne pytania (opcjonalne) | `"agencja Google Ads Wrocław"` |
| `--models` | Nadpisz listę modeli (opcjonalne) | `"openai/gpt-4o"` |
| `--out` | Ścieżka do zapisu raportu | `raport-2025-03-18.md` |
| `--lang` | Język pytań (pl/en, default: pl) | `pl` |

## Jak używać

### Szybki check

```
/ai-visibility check endocare.wroclaw.pl endocare.wroclaw.pl
```

### Pełny audyt z raportem

```
/ai-visibility report "Endocare Wrocław" endocare.wroclaw.pl --out endocare-ai-visibility-2025-03-18.md
```

### Z własnymi pytaniami

```
/ai-visibility check "Qubus Hotel" qubushotel.com \
  --prompts "najlepszy hotel w Krakowie" "hotel konferencyjny Polska" "hotel business Kraków"
```

## Workflow dla Claude Code

Kiedy skill zostaje wywołany, Claude Code powinien:

1. **Sprawdzić zależności** — upewnić się że Python 3.8+ i `openai` są dostępne
2. **Pobrać OPENROUTER_API_KEY** z env lub zapytać użytkownika
3. **Uruchomić skrypt** `scripts/ai_visibility_check.py` z parametrami
4. **Odczytać wyniki** z pliku JSON i wygenerować raport markdown
5. **Zapisać raport** do pliku z datą w nazwie

### Szczegółowy workflow

```bash
# 1. Sprawdź Python
python3 --version

# 2. Zainstaluj zależności
pip install openai --quiet

# 3. Ustaw API key (jeśli nie w env)
export OPENROUTER_API_KEY="sk-or-..."

# 4. Uruchom audyt
python3 ~/.claude/skills/ai-visibility/scripts/ai_visibility_check.py \
  --brand "nazwa marki" \
  --domain "domena.pl" \
  --lang pl \
  --out /tmp/ai_visibility_results.json

# 5. Raport jest generowany automatycznie jako {domain}-ai-visibility-YYYY-MM-DD.md
```

## Interpretacja wyników

### Visibility Score

Każda marka dostaje score 0–100 na każdym modelu:

| Score | Status | Znaczenie |
|---|---|---|
| 80–100 | 🟢 Silna | Marka dominuje w odpowiedziach |
| 50–79 | 🟡 Umiarkowana | Pojawia się, ale nie na pierwszym miejscu |
| 20–49 | 🟠 Słaba | Sporadyczne wzmianki |
| 0–19 | 🔴 Niewidoczna | Brak wzmianek |

### Metryki per model

- **mention_rate** — % pytań, w których marka jest wymieniona
- **citation_rate** — % pytań, w których domena jest w cytowaniach
- **avg_mentions** — średnia liczba wzmianek per odpowiedź
- **position_score** — czy marka jest na początku odpowiedzi (wyżej = lepiej)

## Modele objęte audytem

Skrypt odpytuje te modele domyślnie (wszystkie z web searchem `:online`):

| Model | Provider | Specjalizacja |
|---|---|---|
| `openai/gpt-4o` | OpenAI | Ogólny, szeroko używany |
| `openai/gpt-4o-mini` | OpenAI | Szybki, tani |
| `perplexity/sonar-pro` | Perplexity | Wyszukiwarka AI, bogate cytowania |
| `perplexity/sonar` | Perplexity | Szybsza wersja |
| `anthropic/claude-sonnet-4` | Anthropic | Precyzyjny, ostrożny |
| `google/gemini-2.0-flash-001` | Google | Świeże dane Google |
| `google/gemini-2.5-pro-preview` | Google | Zaawansowana analiza |
| `mistralai/mistral-large` | Mistral | Europejski model |
| `x-ai/grok-3-mini-beta` | xAI | Twitter/X data |
| `meta-llama/llama-4-maverick` | Meta | Open source |

## Output — struktura raportu

```markdown
# Raport AI Visibility — {brand}
## Executive Summary
## Wyniki per model (tabela)
## Najczęściej cytowane źródła
## Przykładowe odpowiedzi (top/bottom)
## Analiza pozycji vs konkurencja
## Rekomendacje GEO
## Metodologia
```

## Wskazówki

- Uruchamiaj regularnie (co tydzień/miesiąc) żeby śledzić trendy
- Używaj tych samych pytań przy każdym audycie dla porównywalności
- Dodaj pytania w języku klientów, nie tylko po polsku
- Testuj zarówno nazwę marki jak i domenę osobno
- Perplexity ma najlepsze cytowania — priorytetyzuj optymalizację pod niego

## Wymagania

- Python 3.8+
- `openai` library (`pip install openai`)
- `OPENROUTER_API_KEY` w zmiennych środowiskowych
