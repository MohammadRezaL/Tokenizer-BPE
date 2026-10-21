# Advanced NLP Homework 1 — From-scratch BPE (English + Persian)

## Files in this submission

```text
student_id_HW1/
├── bpe.py                  # Main BPE implementation and command-line interface
├── test_bpe.py             # All 29 tests, executed
├── results/
│   ├── vocab_en.json       # 200 learned English merges
│   ├── vocab_fa.json       # 200 learned Persian merges
│   ├── comparison_table.csv# 18 comparisons with GPT-4 cl100k_base
│   └── plots/
│       ├── vocab_growth.png
│       └── token_distribution.png
├── report.pdf              # 4-page report in Persian
└── README.md
```

## Environment and run instructions

Requires **Python >= 3.10**. Unit tests and `bpe.py` use **Python standard library only**.

From the `student_id_HW1` directory:

```bash
python -m unittest test_bpe.py -v
python bpe.py
python bpe.py encode results/vocab_en.json "The curious child opened the old wooden box."
python bpe.py encode results/vocab_fa.json "امروز یک کودک کنجکاو جعبه قدیمی را باز کرد."
```

Training your own English/Persian corpus files (UTF-8 text; each line may contain a sentence):

```bash
python bpe.py train path/to/english.txt path/to/persian.txt --num-merges 200 --min-frequency 2
```

**Warning:** The `train` command overwrites the two vocabulary files. To preserve the provided evaluation, first make a copy of the folder. The original two 72-sentence teaching corpora are not bundled because the requested submission layout lists exactly these files. To exactly reproduce the original vocabularies from scratch, the original corpora are required. The included pretrained vocabulary JSON files permit exact regeneration of all 18 *custom* token counts and the plots; test assertions check those counts.

## Data, metrics, and provenance

- Original English and Persian training sets: 72 manually written teaching sentences each; **not TinyStories**. Original size (including whitespace): English 4,450 characters and 762 whitespace-delimited words; Persian 3,899 characters and 729 words.
- NFC Unicode normalization, with Persian zero-width non-joiners kept. Training parameters: **num_merges=200; min_frequency=2**.
- The implementation is **character-level BPE**, with word-ending `</w>` and whitespace splitting, not a byte-level BPE equivalent to GPT-4. It drops whitespace; `encode()` is **not lossless** for full input text.
- `vocab_growth.png`: cumulative vocabulary size versus actual sequence of 200 learned merges. Each merge adds one token type in this run; not a plot of corpus size versus performance.
- `token_distribution.png`: count of BPE token occurrences by **Unicode character length** on the **8 held-out sentences per language**. Different from GPT-4's bytes-per-token distribution; do not present them as identical measures.
- GPT-4 measurements: all **18 token counts are supplied manually by the student** from an online tokenizer configured to `gpt-4 / cl100k_base`. They were **not independently checked via tiktoken**. GPT-4 token IDs and boundaries are unknown and explicitly remain empty in the CSV.
- Two in-corpus reference sentences: `EN01`, `FA01`. Sixteen held-out examples: `EN02–EN09`, `FA02–FA09`.
- On held-out examples only, English BPE: 202 vs GPT-4: 82; Persian BPE: 178 vs GPT-4: 268. Token count alone does not measure tokenizer quality.
- The sensitivity experiment (18/36/72 line subsets) and 5-run median timing are described in the PDF; varying prefixes of a single corpus does **not** isolate corpus size from content effects.

## Glossary

- BPE: Byte Pair Encoding (this assignment uses a character-based educational variant)
- Corpus: training text collection
- Vocabulary: characters plus newly learned merged tokens
- Merge: merge of the most frequent adjacent token pair
- ZWNJ: zero-width non-joiner (Persian half-space)
