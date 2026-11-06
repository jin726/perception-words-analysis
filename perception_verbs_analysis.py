"""
Perception verbs in The Turn of the Screw: corpus-assisted stylistic analysis.

Usage:
    python perception_verbs_analysis.py extract   # stages 1-5: extraction, statistics, charts, coding sheet
    python perception_verbs_analysis.py coded     # stage 6: statistics after manual coding is finished

Directory layout (created automatically where needed):
    data/turn_of_the_screw.txt     original novel (UTF-8)
    output/                        all results
"""
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_FILE = Path("turn_of_the_screw.txt")
OUT = Path("output")
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- configuration
VERBS = {
    "see": ["see", "sees", "saw", "seen", "seeing"],
    "hear": ["hear", "hears", "heard", "hearing"],
    "feel": ["feel", "feels", "felt", "feeling"],
    "notice": ["notice", "notices", "noticed", "noticing"],
    "look": ["look", "looks", "looked", "looking"],
    "watch": ["watch", "watches", "watched", "watching"],
    "observe": ["observe", "observes", "observed", "observing"],
    "seem": ["seem", "seems", "seemed", "seeming"],
    "appear": ["appear", "appears", "appeared", "appearing"],
}
CATEGORIES = {
    "Visual": ["see", "look", "watch", "observe"],
    "Auditory": ["hear"],
    "Other_Sensory": ["feel"],
    "Attention": ["notice"],
    "Appearance": ["seem", "appear"],
}
# Heuristic flags for *probably non-perceptual* uses. They are only flagged for
# manual review, never deleted. Extend this list after reading your concordance.
NON_PERCEPTUAL_PATTERNS = [
    r"^look(s|ed|ing)?\s+(after|for|forward to|up to|down on)\b",
    r"^appear(s|ed|ing)?\s+(at|in|on|from|before|upon|suddenly)\b",
    r"^watch(es|ed|ing)?\s+over\b",
    r"^feel(s|ing)?\s+(like|for)\b",
]
CODING_COLUMNS = [
    "Perceiver", "Sensory_Type", "Grammatical_Pattern", "Perceptual_Status",
    "Narrative_Function", "Contextual_Confirmation", "Research_Notes",
]
FORM_TO_LEMMA = {f: lemma for lemma, forms in VERBS.items() for f in forms}
VERB_RE = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, FORM_TO_LEMMA), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"\b[A-Za-z]+(?:['’][A-Za-z]+)?\b")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])[\"”’']?\s+")


def get_category(lemma):
    for cat, items in CATEGORIES.items():
        if lemma in items:
            return cat
    return "Other"


# ---------------------------------------------------------------- step 1: load
def load_text():
    text = DATA_FILE.read_text(encoding="utf-8")
    print("Total characters (raw):", len(text))
    print(text[:500], "\n")

    # Strip Project Gutenberg header/footer if present.
    start = re.search(r"\*\*\* ?START OF.*?\*\*\*", text)
    if start:
        text = text[start.end():]
    end = re.search(r"\*\*\* ?END OF", text)
    if end:
        text = text[: end.start()]
    text = text.replace("\r\n", "\n")
    (OUT / "turn_of_the_screw_cleaned.txt").write_text(text, encoding="utf-8")
    return text


def split_chapters(text):
    """Split on lines that consist only of a Roman numeral / 'CHAPTER X'.
    Everything before the first heading is labelled 'Prologue'.
    CHECK the printed list against your edition and adjust HEADING_RE if needed."""
    HEADING_RE = re.compile(
        r"^[ \t]*(?:CHAPTER[ \t]+)?([IVXLCDM]+|\d+)\.?[ \t]*$", re.MULTILINE | re.IGNORECASE
    )
    matches = list(HEADING_RE.finditer(text))
    chapters = {}
    if not matches:
        print("WARNING: no chapter headings found; treating the text as one chapter.")
        return {"Whole_Text": text}
    chapters["Prologue"] = text[: matches[0].start()]
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chapters[f"Chapter {m.group(1).upper()}"] = text[m.end():end]
    print("Detected sections:", list(chapters))
    print("(Expect Prologue + 24 chapters for the Gutenberg edition.)\n")
    return chapters


def count_words(s):
    return len(WORD_RE.findall(s))


# ---------------------------------------------------------------- step 2: extract
def extract_candidates(chapters):
    records = []
    sentence_id = 0
    for chapter, ch_text in chapters.items():
        flat = re.sub(r"\s+", " ", ch_text).strip()
        sentences = [s.strip() for s in SENT_SPLIT_RE.split(flat) if s.strip()]
        for i, sent in enumerate(sentences):
            sentence_id += 1
            for m in VERB_RE.finditer(sent):
                form = m.group(1)
                lemma = FORM_TO_LEMMA[form.lower()]
                tail = sent[m.start(): m.start() + 40].lower()
                flag = any(re.search(p, tail) for p in NON_PERCEPTUAL_PATTERNS)
                records.append({
                    "Sentence_ID": sentence_id,
                    "Chapter": chapter,
                    "Lemma": lemma,
                    "Word_Form": form.lower(),
                    "Category": get_category(lemma),
                    "Auto_Flag_Non_Perceptual": flag,
                    "Prev_Sentence": sentences[i - 1] if i > 0 else "",
                    "Sentence": sent,
                    "Next_Sentence": sentences[i + 1] if i + 1 < len(sentences) else "",
                })
    return pd.DataFrame(records)


# ---------------------------------------------------------------- step 3-4: statistics
def frequency_tables(df, total_words, chapters):
    freq = df.groupby("Lemma").size().reindex(VERBS).fillna(0).astype(int).reset_index(name="Frequency")
    freq["Normalized_Frequency"] = freq["Frequency"] / total_words * 10000

    cat = df.groupby("Category").size().reset_index(name="Frequency")
    cat["Percentage"] = cat["Frequency"] / cat["Frequency"].sum() * 100
    cat["Normalized_Frequency"] = cat["Frequency"] / total_words * 10000

    form_freq = df.groupby(["Lemma", "Word_Form"]).size().reset_index(name="Frequency")

    ch_words = {c: count_words(t) for c, t in chapters.items()}
    ch = df.groupby(["Chapter", "Lemma"]).size().reset_index(name="Frequency")
    full = pd.MultiIndex.from_product([list(chapters), list(VERBS)], names=["Chapter", "Lemma"])
    ch = ch.set_index(["Chapter", "Lemma"]).reindex(full, fill_value=0).reset_index()
    ch["Total_Words"] = ch["Chapter"].map(ch_words)
    ch["Normalized_Frequency"] = ch["Frequency"] / ch["Total_Words"].where(ch["Total_Words"] > 0) * 10000
    ch["Normalized_Frequency"] = ch["Normalized_Frequency"].fillna(0)

    ch_cat = df.groupby(["Chapter", "Category"]).size().unstack(fill_value=0).reindex(list(chapters), fill_value=0)
    ch_cat["Total_Words"] = pd.Series(ch_words)
    for c in CATEGORIES:
        if c in ch_cat:
            ch_cat[f"{c}_per_10k"] = ch_cat[c] / ch_cat["Total_Words"] * 10000
    return freq, cat, form_freq, ch, ch_cat.reset_index()


# ---------------------------------------------------------------- step 6: charts
def bar_chart(x, y, xlabel, ylabel, title, path, rotate=30):
    plt.figure(figsize=(10, 6))
    plt.bar(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=rotate, ha="right" if rotate else "center")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def make_charts(freq, ch_cat):
    bar_chart(freq["Lemma"], freq["Frequency"], "Perception verbs", "Frequency",
              "Frequency of Perception Verbs in The Turn of the Screw",
              OUT / "perception_verb_frequency.png")
    cols = [f"{c}_per_10k" for c in CATEGORIES if f"{c}_per_10k" in ch_cat]
    if cols:
        ax = ch_cat.set_index("Chapter")[cols].plot(figsize=(12, 6), marker="o")
        ax.set_ylabel("Frequency per 10,000 words")
        ax.set_title("Perception verb categories across the narrative")
        plt.xticks(rotation=60, ha="right")
        plt.tight_layout()
        plt.savefig(OUT / "category_by_chapter.png", dpi=300)
        plt.close()


# ---------------------------------------------------------------- stage runners
def run_extract():
    text = load_text()
    words = WORD_RE.findall(text)
    total_words = len(words)
    print("Total words (running words):", total_words)

    chapters = split_chapters(text)
    df = extract_candidates(chapters)
    print(f"Candidate instances: {len(df)} "
          f"({df['Auto_Flag_Non_Perceptual'].sum()} auto-flagged as probably non-perceptual)\n")

    freq, cat, form_freq, ch, ch_cat = frequency_tables(df, total_words, chapters)
    print(freq, "\n")
    print(cat, "\n")

    with pd.ExcelWriter(OUT / "perception_verbs_raw.xlsx") as xw:
        df.to_excel(xw, sheet_name="Raw_Candidates", index=False)
        freq.to_excel(xw, sheet_name="Lemma_Frequency", index=False)
        form_freq.to_excel(xw, sheet_name="Form_Frequency", index=False)
        cat.to_excel(xw, sheet_name="Category_Frequency", index=False)
        ch.to_excel(xw, sheet_name="Chapter_Lemma", index=False)
        ch_cat.to_excel(xw, sheet_name="Chapter_Category", index=False)
    make_charts(freq, ch_cat)

    coding = df.copy()
    for col in CODING_COLUMNS:
        coding[col] = ""
    coding.to_excel(OUT / "perception_verbs_coding.xlsx", index=False)
    print("Saved: output/perception_verbs_raw.xlsx, perception_verbs_coding.xlsx, charts (.png)")
    print("Next: pilot-code 20-30 rows in perception_verbs_coding.xlsx, refine the scheme, "
          "code everything, then run:  python perception_verbs_analysis.py coded")


def run_coded():
    path = OUT / "perception_verbs_coding.xlsx"
    coded = pd.read_excel(path)
    filled = coded[coded["Narrative_Function"].notna() & (coded["Narrative_Function"].astype(str).str.strip() != "")]
    print(f"Coded rows: {len(filled)} / {len(coded)}")
    if filled.empty:
        print("No rows coded yet.")
        return

    tables = {}
    for col in ["Perceiver", "Sensory_Type", "Grammatical_Pattern",
                "Perceptual_Status", "Narrative_Function", "Contextual_Confirmation"]:
        t = filled[col].value_counts().reset_index()
        t.columns = [col, "Frequency"]
        t["Percentage"] = t["Frequency"] / t["Frequency"].sum() * 100
        tables[col] = t
        print(t, "\n")

    # Cross-tabs useful for the Results section
    tables["Lemma_x_Function"] = pd.crosstab(filled["Lemma"], filled["Narrative_Function"])
    tables["Chapter_x_Function"] = pd.crosstab(filled["Chapter"], filled["Narrative_Function"])
    tables["Perceiver_x_Status"] = pd.crosstab(filled["Perceiver"], filled["Perceptual_Status"])

    with pd.ExcelWriter(OUT / "coded_statistics.xlsx") as xw:
        for name, t in tables.items():
            t.to_excel(xw, sheet_name=name[:31], index=name in
                       ("Lemma_x_Function", "Chapter_x_Function", "Perceiver_x_Status"))

    nf = tables["Narrative_Function"]
    bar_chart(nf["Narrative_Function"], nf["Frequency"], "Narrative function", "Frequency",
              "Narrative Functions of Perception Verbs", OUT / "narrative_functions.png")
    print("Saved: output/coded_statistics.xlsx, narrative_functions.png")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "extract"
    {"extract": run_extract, "coded": run_coded}[mode]()