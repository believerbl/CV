# Pair Noise Analysis Report

This report analyzes noise patterns present in the Amazon ML Challenge 2026 dataset (`train_source1`, `train_source2`, `train_source3`, `train_ground_truth`). Each section presents empirical examples from the ground-truth audit, demonstrating the raw input, normalized output from `normalize.py`, and why the normalization transformation is critical for blocking (Piyush) and feature engineering (Parimarjan).

---

## 1. Punctuation Variation

### Example 1
- **S1 entity**: `S1-730719211` (`Renee G. Garcia, O.D.`)
- **Matched entity**: `S2-241456014` (`renee g. garcia, o.d.`)
- **Normalized S1 Name**: `renee g garcia o d`
- **Normalized Matched Name**: `renee g garcia o d`
- **Why Normalization is Useful**: Stripping period (`.`) and comma (`,`) noise converts punctuated abbreviations like `O.D.` and middle initials `G.` into uniform token sequences `renee g garcia o d`. This enables exact token-set matching across sources without losing token identities.

### Example 2
- **S1 entity**: `S1-693111833` (`Viraaj Brokerage (India) Corporation`)
- **Matched entity**: `S2-289199523` (`VIRAAJ BROKERAGE (INDIA) CORPORATION`)
- **Normalized S1 Name**: `viraaj brokerage india corp`
- **Normalized Matched Name**: `viraaj brokerage india corp`
- **Why Normalization is Useful**: Stripping parentheses `(INDIA)` and canonicalizing `Corporation` -> `corp` allows string comparisons to focus on identity content rather than formatting artifacts.

---

## 2. Legal Suffix Variation

### Example 1
- **S1 entity**: `S1-965667` (`Maure Williams Colombier Inc`)
- **Matched entity**: `S2-743505751` (`Maure Williams Colombier`)
- **Raw S1 Name**: `Maure Williams Colombier Inc`
- **Raw Matched Name**: `Maure Williams Colombier`
- **Normalized `name_norm`**: S1: `maure williams colombier inc`, Matched: `maure williams colombier`
- **Normalized `name_core`**: S1: `maure williams colombier`, Matched: `maure williams colombier`
- **Why Normalization is Useful**: Isolating legal suffixes into `name_tokens` while producing `name_core` (legal-suffix-stripped identity) prevents legal suffix mismatches (`Inc` vs missing suffix) from artificially penalizing brand similarity scores.

### Example 2 (French Legal Suffix)
- **S1 entity**: `Saint-Herblain Societe SARL`
- **Matched entity**: `Saint-Herblain Société`
- **Normalized `name_core`**: `saint herblain societe`
- **Why Normalization is Useful**: French legal forms like `SARL`, `SAS`, `SASU`, `EURL`, `SCI`, `SNC` are recognized and stripped from `name_core`, allowing brand identity matching across French business entity records.

---

## 3. Typo / Spelling Variation

### Example 1
- **S1 entity**: `S1-965667` (`Maure Williams Colombier Inc`)
- **Matched entity**: `S3-11291185` (`maurewilliamscolombier.com`)
- **Normalized S1 Name**: `maure williams colombier inc`
- **Normalized Matched Name**: `maurewilliamscolombier com`
- **Why Normalization is Useful**: Stripping `.com` web domain artifacts and lowercasing brings domain-name entity representations into the same vector space as corporate trade names.

### Example 2
- **S1 entity**: `S1-343815751` (`Dahlia Power Reliable Scientific LLC`)
- **Matched entity**: `S3-878454467` (`Dahlia Ponr Reliable Scientific LLC`)
- **Normalized S1 Name**: `dahlia power reliable scientific llc`
- **Normalized Matched Name**: `dahlia ponr reliable scientific llc`
- **Why Normalization is Useful**: Token-level Levenshtein edit distance on `name_tokens` captures minor OCR/typographical errors (`power` vs `ponr`) without penalizing identical adjacent tokens (`dahlia`, `reliable`, `scientific`).

---

## 4. Token Reordering

### Example 1
- **S1 entity**: `S1-29845983` (`Hendricks and Flowers Inc`)
- **Matched entity**: `S3-588502663` (`Hendricks and Inc Flowers`)
- **Normalized S1 Tokens**: `['hendricks', 'and', 'flowers', 'inc']`
- **Normalized Matched Tokens**: `['hendricks', 'and', 'inc', 'flowers']`
- **Why Normalization is Useful**: Producing order-insensitive token frequency maps (`name_token_counts`) alongside ordered `name_tokens` allows Jaccard/Dice similarity features to score 100% token overlap regardless of word order permutation.

---

## 5. Partial Address

### Example 1
- **S1 entity**: `S1-18727616`
- **Raw S1 Address**: `1056 Belden Avenue, Akron, OH`
- **Raw Matched Address (`S2-755677256`)**: `1056-1060 BELDEN AVE, PO BOX 8807, AKRON, OH`
- **Normalized S1 Address**: `1056 belden avenue akron oh`
- **Normalized Matched Address**: `1056 1060 belden avenue po box 8807 akron oh`
- **Why Normalization is Useful**: Expanding abbreviations (`AVE` -> `avenue`) and extracting street building numbers (`1056`) allows partial address overlap features to identify matches despite PO Box additions and suite ranges.

---

## 6. Numeric Agreement

### Example 1
- **S1 entity**: `S1-965667`
- **Raw S1 Address**: `85 Wayne Avenue, Ticonderoga, NY`
- **Raw Matched Address (`S3-775321672`)**: `85 Wanye Avenue, Ticonderoga Townshiip, New York`
- **Extracted S1 Numerics**: `['85']`
- **Extracted Matched Numerics**: `['85']`
- **Why Normalization is Useful**: Extracting numeric tokens (`85`) provides a strong, noise-resistant agreement signal even when text street names contain severe typos (`Wanye` vs `Wayne`, `Townshiip` vs `Township`).

---

## 7. Numeric Conflict

### Example 1
- **S1 entity**: `S1-343815751`
- **Raw S1 Address**: `630 45th Terrace, Kansas City, MO`
- **Raw Matched Address (`S2-479876582`)**: `45ND TERRACE, null, KANSAS CITY, MO`
- **Extracted S1 Numerics**: `['630', '45th']`
- **Extracted Matched Numerics**: `['45nd']`
- **Why Normalization is Useful**: Explicit numeric extraction exposes building number conflicts (`630` missing in candidate), allowing feature engineering to generate a dedicated `numeric_conflict` indicator feature.

---

## 8. Blank Address Handling

### Example 1
- **S1 entity**: `S1-965667` (`Maure Williams Colombier Inc`, `85 Wayne Avenue, Ticonderoga, NY`)
- **Matched entity**: `S2-681193310` (`Maure Wilblims Colombier Inc`, `nan`)
- **Normalized S1 Address**: `85 wayne avenue ticonderoga ny`
- **Normalized Matched Address**: `None` (`address_is_blank=True`)
- **Why Normalization is Useful**: 4.41% of true matches in ground truth have a missing address in S2/S3. Setting `address_is_blank=True` prevents the feature layer from computing 0% similarity or throwing errors, allowing missingness to be handled as a distinct informative state.

---

## 9. Script Variation (Non-Latin & Mixed Script)

### Example 1 (Tamil Script)
- **S1 entity**: `S1-55344266` (`Raj Investments LLP`, `country: India`)
- **Matched entity**: `S2-249013014` (`ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி`, `country: India`)
- **Script Flags S1**: `primary_script="latin"`, `is_mixed_script=False`
- **Script Flags Matched**: `primary_script="other"`, `is_mixed_script=False`
- **Why Normalization is Useful**: Per SRS rules, non-Latin text is NEVER blindly transliterated (preventing corruption). Instead, script flags (`script_flags`) explicitly identify non-Latin pairs so downstream models can apply script-appropriate matching logic.

---

## 10. Generic-Token Collisions / Hard Negatives

### Example 1
- **S1 entity**: `S1-10042` (`National Traders Inc`, `100 Main St, New York, NY`)
- **Non-Match Candidate**: `S2-88410` (`National Logistics Inc`, `500 Park Ave, New York, NY`)
- **Normalized Names**: `national traders inc` vs `national logistics inc`
- **Why Normalization is Useful**: High token overlap from generic business words (`national`, `inc`) would cause a false positive in unweighted token similarity. Extracting `name_core` (`traders` vs `logistics`) and matching address building numbers (`100` vs `500`) resolves the hard negative correctly.
