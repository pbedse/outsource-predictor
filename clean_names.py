import pandas as pd, re
from rapidfuzz import process, fuzz

df = pd.read_csv("trials_raw.csv")

# ---------------------------------------------------------------
# 1. NORMALIZE — strip legal suffixes, punctuation, casing
# ---------------------------------------------------------------
SUFFIXES = ["inc","llc","ltd","limited","gmbh","ag","sa","plc","corp","corporation",
            "co","company","pharmaceuticals","pharmaceutical","pharma","therapeutics",
            "biosciences","bioscience","biopharma","biopharmaceuticals","labs",
            "laboratories","holdings","group","and company","a s","as","kk","bv","nv",
            "spa","srl","llp","lp","usa","international","research","development",
            "sciences","science","health","healthcare","medical","technologies"]

def normalize(name):
    if not isinstance(name, str):
        return ""
    n = name.lower().strip()
    n = n.replace("&", " and ")
    n = re.sub(r"[^\w\s]", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    changed = True
    while changed:
        changed = False
        for s in sorted(SUFFIXES, key=len, reverse=True):
            if n.endswith(" " + s) and len(n) > len(s) + 4:   # never strip to nothing
                n = n[: -(len(s) + 1)].strip()
                changed = True
    return n

df["sponsor_clean"] = df["sponsor_name"].apply(normalize)

# ---------------------------------------------------------------
# 2. ALIAS DICTIONARY — subsidiary -> parent
#    Fuzzy matching CANNOT do this. "Genentech" and "Roche" share no letters.
# ---------------------------------------------------------------
ALIASES = {
    "hoffmann la roche": "roche",
    "f hoffmann la roche": "roche",
    "genentech": "roche",
    "chugai": "roche",
    "janssen": "johnson and johnson",
    "janssen pharmaceutica": "johnson and johnson",
    "janssen biotech": "johnson and johnson",
    "janssen vaccines and prevention": "johnson and johnson",
    "janssen cilag": "johnson and johnson",
    "merck sharp and dohme": "merck",
    "msd": "merck",
    "glaxosmithkline": "gsk",
    "bristol myers squibb": "bms",
    "celgene": "bms",
    "juno": "bms",
    "sanofi aventis": "sanofi",
    "genzyme": "sanofi",
    "shire": "takeda",
    "millennium": "takeda",
    "allergan": "abbvie",
    "pharmacyclics": "abbvie",
    "alexion": "astrazeneca",
    "medimmune": "astrazeneca",
    "seagen": "pfizer",
    "array biopharma": "pfizer",
    "wyeth": "pfizer",
    "hospira": "pfizer",
    "novo nordisk a s": "novo nordisk",
    "boehringer ingelheim": "boehringer",
    "eli lilly and": "eli lilly",
    "lilly": "eli lilly",
}
df["sponsor_clean"] = df["sponsor_clean"].replace(ALIASES)

# ---------------------------------------------------------------
# 3. FUZZY GROUPING — catch remaining near-identical spellings
# ---------------------------------------------------------------
THRESHOLD = 95
MIN_LEN = 12   # short names are too collision-prone to fuzzy-match

unique = sorted(df["sponsor_clean"].dropna().unique())
canonical, mapping, fuzzy_merges = [], {}, []

for name in unique:
    if not name:
        continue
    if len(name) < MIN_LEN or not canonical:
        canonical.append(name); mapping[name] = name; continue
    pool = [c for c in canonical if len(c) >= MIN_LEN]
    match = process.extractOne(name, pool, scorer=fuzz.token_sort_ratio) if pool else None
    if match and match[1] >= THRESHOLD:
        mapping[name] = match[0]
        fuzzy_merges.append((name, match[0], match[1]))
    else:
        canonical.append(name); mapping[name] = name

df["sponsor_final"] = df["sponsor_clean"].map(mapping)
df.to_csv("trials_clean.csv", index=False)

# ---------------------------------------------------------------
# 4. REPORT — these numbers go in your slide
# ---------------------------------------------------------------
raw   = df["sponsor_name"].nunique()
norm  = df["sponsor_clean"].nunique()
final = df["sponsor_final"].nunique()

print("=" * 58)
print("ENTITY RESOLUTION REPORT")
print("=" * 58)
print(f"Raw sponsor names:            {raw}")
print(f"After normalization + alias:  {norm}   (-{raw-norm})")
print(f"After fuzzy grouping:         {final}   (-{norm-final})")
print(f"Total collapsed:              {raw-final}  ({(raw-final)/raw:.1%})")

singletons = (df["sponsor_final"].value_counts() == 1).sum()
print(f"\nSponsors appearing only once: {singletons} of {final} "
      f"({singletons/final:.0%}) — this caps how much collapsing is possible")

if fuzzy_merges:
    print(f"\nFuzzy merges ({len(fuzzy_merges)}):")
    for a, b, s in fuzzy_merges[:15]:
        print(f"   {a:38s} -> {b:28s} [{s:.0f}]")

alias_hits = df[df["sponsor_name"].apply(normalize) != df["sponsor_clean"]]
print(f"\nAlias-dictionary hits (subsidiary -> parent): {len(alias_hits)} trials")
if len(alias_hits):
    print(alias_hits[["sponsor_name","sponsor_final"]]
          .drop_duplicates().head(12).to_string(index=False))

print("\nTop sponsors after resolution:")
print(df["sponsor_final"].value_counts().head(10).to_string())