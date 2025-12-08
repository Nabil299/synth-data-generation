# 🏗️ System Design & Evolution

## Overview
A synthetic review generation pipeline with quality guardrails, duplicate detection, and rating-aware generation to produce diverse, realistic product reviews.

---

## 📚 Research Foundation

Based on the survey paper ["Synthetic Data Generation Using Large Language Models: Advances in Text and Code"](https://arxiv.org/pdf/2503.14023):

> "The importance of maximizing diversity decreases when synthetic data is mixed with even modest amounts of real data, since the real samples anchor the data distribution and provide natural variety."

**Key Insight:** Few-shot prompting with real examples helps anchor the distribution while maintaining diversity.

---

## 🔄 Pipeline Evolution

### 1️⃣ **Few-Shot Prompting with Randomization**
- **Problem:** Static examples → repetitive vocabulary
- **Solution:** Randomly sample few-shot examples per batch
- **Result:** Increased lexical diversity across generated samples

### 2️⃣ **Semantic Duplicate Detection**
- **Implementation:** PostgreSQL + pgvector with `all-MiniLM-L6-v2` embeddings
- **Threshold:** 0.80 cosine similarity (tunable)
- **Logic:** Skip similarity check for first N samples (configurable)
- **Benefit:** Ensures semantic diversity across the dataset

### 3️⃣ **Rating-Aware Retry Mechanism**
- **Problem:** Rejected samples don't maintain rating distribution
- **Solution:** Track rejections by rating (1-5 stars)
- **Implementation:**
  ```
  rejected_by_rating = {1: 5, 3: 12, 4: 3}  # Example
  → Retry generates: 5×1-star, 12×3-star, 3×4-star reviews
  ```
- **Benefit:** Final dataset maintains intended rating distribution despite rejections

### 4️⃣ **Sentiment-Rating Alignment Validation**

**Initial Attempt:**
- Model: `distilbert-base-uncased-finetuned-sst-2-english`
- **Problem:** Binary classification (Positive/Negative) → No neutral detection
- **Impact:** 3-star reviews impossible to validate

**Current Solution:**
- Model: `cardiffnlp/twitter-roberta-base-sentiment`
- Labels: `LABEL_0` (Negative), `LABEL_1` (Neutral), `LABEL_2` (Positive)
- Mapping: Convert to 0-1 scale for threshold matching
- **Remaining Challenge:** Model struggles with nuanced 3-star reviews
- **Mitigation:** Enhanced prompt with explicit 3-star guidelines

### 5️⃣ **Multi-Model Support**
- Database stores `model_name` with each review
- Duplicate detection scoped per model
- Each model gets separate output folder with metrics

---

## 📊 Quality Metrics

### Calculated Automatically:

1. **Semantic Diversity Score**
   - Measures inter-review similarity (1 - avg_pairwise_similarity)
   - Target: > 0.6 (high diversity)

2. **Real-vs-Synthetic Similarity**
   - Nearest-neighbor similarity to real reviews
   - Target: 0.35-0.65 (balanced realism)

3. **Sentiment-Rating Alignment**
   - Pearson correlation between ratings and sentiment scores
   - Target: > 0.75 (strong alignment)

---

## 🎯 Current Architecture

```
User Config → Few-Shot Sampling → LLM Generation → Quality Checks → Database
                                                          ↓ (rejected)
                                                    Retry (Rating-Aware)
                                                          ↓
                                                    Final Dataset
                                                          ↓
                                                   Metrics Calculation
                                                          ↓
                                              Markdown Report + CSV + JSON
```

---

## ⚖️ Trade-offs & Design Decisions

### Speed vs. Quality

**Response Format Validation Overhead**
- **Impact:** Structured JSON output with schema validation addlatency per request
- **Justification:** Ensures parseable output, eliminates manual post-processing
- **Alternative Considered:** Free-form text → Would require regex/parsing → More brittle

**Model Size Selection**
- **Decision:** Use smaller, faster models (e.g., Llama-3.1-8B, Mistral-7B)
- **Rationale:** 
  - Generation speed > output perfection
  - Guardrails catch quality issues post-generation
  - 10x faster than 70B+ models with 80% of the quality
- **Trade-off:** Occasional format errors or less sophisticated language vs. significantly faster iteration

### Cost vs. Diversity

**Retry Mechanism Cost**
- Each rejected sample requires regeneration
- Rating-aware retries are more efficient than blind retries
- Typical overhead: 10-30% extra API calls depending on rejection rate

---

## ⚠️ Known Limitations

### 1. **Neutral Review Generation Challenge**
**Problem:** All tested LLMs struggle to generate truly neutral (3-star) reviews that:
- Express balanced sentiment (both positive AND negative aspects)
- Can be classified as "neutral" by sentiment analyzers

**Root Cause:**
- LLMs are trained on polarized data (strongly positive or negative reviews)
- Neutral sentiment is underrepresented in training data
- Even with explicit 3-star guidelines, models tend toward positive/negative extremes

**Evidence:**
- Generated 3-star reviews often classified as LABEL_0 (Negative) or LABEL_2 (Positive)
- Rarely classified as LABEL_1 (Neutral) even with `cardiffnlp/twitter-roberta-base-sentiment`

**Current Mitigation:**
- Enhanced prompt with explicit 3-star examples and structure templates
- More permissive sentiment thresholds for 3-star
- Manual review of generated 3-star samples recommended

### 2. **Embedding Model Choice**
- `all-MiniLM-L6-v2` is fast but may miss semantic similarities
- No systematic comparison with larger embedding models (e.g., `all-mpnet-base-v2`)
- Trade-off: Speed vs. duplicate detection precision

---

## 🔮 Future Improvements

1. **Dynamic threshold adjustment** based on dataset size
2. **Embedding model comparison** beyond all-MiniLM-L6-v2
3. **Distribution drift detection** across batches
4. **Fine-tune sentiment classifier** on product reviews for better neutral detection
