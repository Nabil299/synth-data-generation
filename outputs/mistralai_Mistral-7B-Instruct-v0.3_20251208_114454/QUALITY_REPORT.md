# 📊 Synthetic Review Quality Report

**Generated**: 2025-12-08T11:44:38.141701

**Model**: mistralai/Mistral-7B-Instruct-v0.3

**Execution Time**: 8m 50s

---

## 📁 Dataset Summary

- **Total Synthetic Reviews**: 300
- **Real Reviews (comparison)**: 50

### Rating Distribution

| Rating | Count |
|--------|-------|
| 1⭐ | 29 |
| 2⭐ | 39 |
| 3⭐ | 93 |
| 4⭐ | 75 |
| 5⭐ | 64 |

---

## 1️⃣ Semantic Diversity Score

**Purpose**: Measures how different synthetic reviews are from each other.

### Results

- **Diversity Score**: `0.6717`
- **Average Similarity**: `0.3283`
- **Min Similarity**: `-0.1169`
- **Max Similarity**: `0.9485`
- **Std Similarity**: `0.1609`

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

- **Average Nearest-Neighbor Similarity**: `0.3996`
- **Realism Score (1 - avg_sim)**: `0.6004`
- **Min Similarity**: `0.1007`
- **Max Similarity**: `0.7294`
- **Std Similarity**: `0.1258`

**Interpretation**: ✅ **BALANCED - Good mix of realism and diversity**

**Scale**:
- `< 0.4`: Unrealistic
- `0.4-0.6`: Balanced (ideal) ✨
- `0.6-0.8`: Very realistic
- `> 0.8`: Too similar (possible copying)

---

## 3️⃣ Sentiment-Rating Alignment Score

**Purpose**: Validates if sentiment matches rating levels.

### Results

- **Pearson Correlation**: `0.8773`
- **P-value**: `0.000000`

**Interpretation**: ✅ **EXCELLENT ALIGNMENT**

### Per-Rating Sentiment Statistics

| Rating | Mean | Std | Min | Max | Count |
|--------|------|-----|-----|-----|-------|
| 1⭐ | 0.014 | 0.012 | 0.008 | 0.075 | 29 |
| 2⭐ | 0.049 | 0.051 | 0.010 | 0.229 | 39 |
| 3⭐ | 0.596 | 0.265 | 0.091 | 0.971 | 93 |
| 4⭐ | 0.928 | 0.079 | 0.694 | 0.994 | 75 |
| 5⭐ | 0.995 | 0.002 | 0.986 | 0.997 | 64 |

**Scale**:
- `< 0.40`: Weak alignment
- `0.50-0.70`: Acceptable alignment
- `0.70-0.75`: Good alignment
- `> 0.75`: Excellent alignment ✨

---

## 📈 Overall Summary

| Metric | Score | Status |
|--------|-------|--------|
| Semantic Diversity | 0.6717 | EXCELLENT DIVERSITY |
| Real-vs-Synthetic | 0.3996 | BALANCED - Good mix of realism and diversity |
| Sentiment Alignment | 0.8773 | EXCELLENT ALIGNMENT |
| **Overall Score** | **0.7828** | - |

### Final Grade: 🌟 **A**

---

*Generated on 2025-12-08 at 11:44:54*
