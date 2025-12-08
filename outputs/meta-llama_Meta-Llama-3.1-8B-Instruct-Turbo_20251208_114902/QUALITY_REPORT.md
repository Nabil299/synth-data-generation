# 📊 Synthetic Review Quality Report

**Generated**: 2025-12-08T11:48:45.629052

**Model**: meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo

**Execution Time**: 3m 51s

---

## 📁 Dataset Summary

- **Total Synthetic Reviews**: 298
- **Real Reviews (comparison)**: 50

### Rating Distribution

| Rating | Count |
|--------|-------|
| 1⭐ | 34 |
| 2⭐ | 38 |
| 3⭐ | 90 |
| 4⭐ | 86 |
| 5⭐ | 50 |

---

## 1️⃣ Semantic Diversity Score

**Purpose**: Measures how different synthetic reviews are from each other.

### Results

- **Diversity Score**: `0.6054`
- **Average Similarity**: `0.3946`
- **Min Similarity**: `-0.1309`
- **Max Similarity**: `1.0000`
- **Std Similarity**: `0.1753`

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

- **Average Nearest-Neighbor Similarity**: `0.4263`
- **Realism Score (1 - avg_sim)**: `0.5737`
- **Min Similarity**: `0.1822`
- **Max Similarity**: `0.7316`
- **Std Similarity**: `0.0948`

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

- **Pearson Correlation**: `0.8690`
- **P-value**: `0.000000`

**Interpretation**: ✅ **EXCELLENT ALIGNMENT**

### Per-Rating Sentiment Statistics

| Rating | Mean | Std | Min | Max | Count |
|--------|------|-----|-----|-----|-------|
| 1⭐ | 0.014 | 0.014 | 0.008 | 0.087 | 34 |
| 2⭐ | 0.029 | 0.023 | 0.010 | 0.095 | 38 |
| 3⭐ | 0.541 | 0.301 | 0.093 | 0.980 | 90 |
| 4⭐ | 0.976 | 0.045 | 0.670 | 0.996 | 86 |
| 5⭐ | 0.994 | 0.002 | 0.985 | 0.996 | 50 |

**Scale**:
- `< 0.40`: Weak alignment
- `0.50-0.70`: Acceptable alignment
- `0.70-0.75`: Good alignment
- `> 0.75`: Excellent alignment ✨

---

## 📈 Overall Summary

| Metric | Score | Status |
|--------|-------|--------|
| Semantic Diversity | 0.6054 | EXCELLENT DIVERSITY |
| Real-vs-Synthetic | 0.4263 | BALANCED - Good mix of realism and diversity |
| Sentiment Alignment | 0.8690 | EXCELLENT ALIGNMENT |
| **Overall Score** | **0.7757** | - |

### Final Grade: 🌟 **A**

---

*Generated on 2025-12-08 at 11:49:02*
