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

## WAŻNE: Workflow przed uruchomieniem

**ZAWSZE przed uruchomieniem audytu zapytaj użytkownika o brakujące dane:**

1. Jeśli nie podano `--locations` → zapytaj: "Z jakich lokalizacji chcesz sprawdzić widoczność? (np. Wrocław, Warszawa, Berlin) Podaj 1–3 lokalizacje lub wpisz 'bez lokalizacji'."
2. Jeśli nie podano `--prompts` → zapytaj: "Jakie pytania mam zadać modelom? Podaj 3–5 pytań lub użyję domyślnych dla języka."
3. NIE uruchamiaj skryptu dopóki nie masz odpowiedzi na powyższe.

## Komendy
```
/ai-visibility <brand> <domain> [--locations "lok1" "lok2"] [--prompts "pytanie1" "pytanie2"]
```

## Parametry

| Parametr      | Opis                                        | Przykład                                      |
|---------------|---------------------------------------------|-----------------------------------------------|
| `brand`       | Nazwa marki (keywords do szukania)          | `7p.marketing`                                |
| `domain`      | Domena do szukania w cytowaniach            | `7p.marketing`                                |
| `--locations` | Lokalizacje do porównania (1–3)             | `"Wroclaw, Poland" "Warsaw, Poland"`          |
| `--prompts`   | Własne pytania (opcjonalne)                 | `"agencja Google Ads Wrocław"`                |
| `--models`    | Nadpisz listę modeli (opcjonalne)           | `"openai/gpt-4o"`                             |
| `--out`       | Ścieżka do zapisu raportu                   | `raport-2025-03-18.md`                        |
| `--lang`      | Język pytań (pl/en, default: pl)            | `pl`                                          |

## Modele objęte audytem

10 modeli z web searchem (`:online`): GPT-4o, GPT-4o-mini, Perplexity Sonar Pro/Sonar,
Claude Sonnet 4, Gemini 2.0 Flash/2.5 Pro, Mistral Large, Grok 3 Mini, Llama 4 Maverick.

## Format raportu (multi-location)

Dla każdego modelu i pytania generowana jest tabela porównawcza lokalizacji:

### `gpt-4o` — score X/100

#### 💬 Treść pytania

| Lokalizacja     | Wzmianka | Cytat URL | Sentiment | Fragment                  |
|-----------------|----------|-----------|-----------|---------------------------|
| Wroclaw, Poland | ✅       | ✅        | 😊        | "Jura to lider..."        |
| Warsaw, Poland  | ❌       | ❌        | 😐        | —                         |
| Berlin, Germany | ✅       | ❌        | 😐        | "Jura is premium..."      |

## Visibility Score: 0–100 per model per lokalizacja

| Score  | Status         |
|--------|----------------|
| 80–100 | 🟢 Silna       |
| 50–79  | 🟡 Umiarkowana |
| 20–49  | 🟠 Słaba       |
| 0–19   | 🔴 Niewidoczna |
