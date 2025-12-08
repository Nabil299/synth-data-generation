# 📊 Synthetic Review Quality Report

**Generated**: 2025-12-08T13:03:51.196865

**Model**: openai/gpt-oss-20b

**Execution Time**: 58m 54s

---

## 📁 Dataset Summary

- **Total Synthetic Reviews**: 300
- **Real Reviews (comparison)**: 50

### Rating Distribution

| Rating | Count |
|--------|-------|
| 1⭐ | 30 |
| 2⭐ | 60 |
| 3⭐ | 90 |
| 4⭐ | 90 |
| 5⭐ | 30 |

---

## 1️⃣ Semantic Diversity Score

**Purpose**: Measures how different synthetic reviews are from each other.

### Results

- **Diversity Score**: `0.6987`
- **Average Similarity**: `0.3013`
- **Min Similarity**: `-0.0927`
- **Max Similarity**: `0.9417`
- **Std Similarity**: `0.1445`

**Interpretation**: ✅ **EXCELLENT DIVERSITY**

**Scale**:
- `< 0.2`: Extremely repetitive
- `0.2-0.3`: Low diversity
- `0.3-0.5`: Moderate diversity
- `0.5-0.6`: Good diversity
- `> 0.6`: Excellent diversity ✨

---

## 2️⃣ Real-vs-Synthetic Similarity Score

**Purpose**: Compares synthetic dataset to real reviews for realism.

### Results

- **Average Nearest-Neighbor Similarity**: `0.2914`
- **Realism Score (1 - avg_sim)**: `0.7086`
- **Min Similarity**: `0.1148`
- **Max Similarity**: `0.5612`
- **Std Similarity**: `0.0791`

**Interpretation**: ✅ **SOMEWHAT UNREALISTIC - Limited resemblance**

**Scale**:
- `< 0.4`: Unrealistic
- `0.4-0.6`: Balanced (ideal) ✨
- `0.6-0.8`: Very realistic
- `> 0.8`: Too similar (possible copying)

---

## 3️⃣ Sentiment-Rating Alignment Score

**Purpose**: Validates if sentiment matches rating levels.

### Results

- **Pearson Correlation**: `0.8942`
- **P-value**: `0.000000`

**Interpretation**: ✅ **EXCELLENT ALIGNMENT**

### Per-Rating Sentiment Statistics

| Rating | Mean | Std | Min | Max | Count |
|--------|------|-----|-----|-----|-------|
| 1⭐ | 0.012 | 0.011 | 0.008 | 0.071 | 30 |
| 2⭐ | 0.103 | 0.074 | 0.022 | 0.306 | 60 |
| 3⭐ | 0.651 | 0.234 | 0.152 | 0.963 | 90 |
| 4⭐ | 0.948 | 0.053 | 0.719 | 0.991 | 90 |
| 5⭐ | 0.991 | 0.008 | 0.958 | 0.996 | 30 |

**Scale**:
- `< 0.40`: Weak alignment
- `0.50-0.70`: Acceptable alignment
- `0.70-0.75`: Good alignment
- `> 0.75`: Excellent alignment ✨

---

## 📈 Overall Summary

| Metric | Score | Status |
|--------|-------|--------|
| Semantic Diversity | 0.6987 | EXCELLENT DIVERSITY |
| Real-vs-Synthetic | 0.2914 | SOMEWHAT UNREALISTIC - Limited resemblance |
| Sentiment Alignment | 0.8942 | EXCELLENT ALIGNMENT |
| **Overall Score** | **0.7252** | - |

### Final Grade: ✅ **B**

---

*Generated on 2025-12-08 at 13:04:07*
