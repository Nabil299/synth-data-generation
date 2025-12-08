# 🚀 Quick Start Guide

## Prerequisites
- Docker and Docker Compose installed
- API key for Together AI (or compatible OpenAI API)

## Setup Instructions

### 1. Configure API Key
Create a `.env` file in the project root:
```bash
cp .env.example .env
```
Then add your API key:
```
API_KEY=your_api_key_here
```

### 2. Configure Generation Settings
Edit `configs/config.yaml` with the following options:

#### **Data Generation Configuration**

```yaml
data_generation_configuration:
  # Persona for review generation
  persona:
    name: "John Doe"
    age: 30
    gender: "male"
    occupation: "Software Engineer"

  # Rating distribution [1-star, 2-star, 3-star, 4-star, 5-star]
  # Must sum to 1.0
  rating_distribution: [0.1, 0.2, 0.3, 0.3, 0.1]
  
  # Total number of reviews to generate
  min_generated_samples: 300
  
  # Reviews per batch (affects API calls)
  batch_size: 10
  
  # Number of real examples to use as few-shot learning
  num_few_shot_examples: 5
  
  # Main prompt for the generation task
  user_prompt: "Generate reviews for Amazon products"
  
  # Characteristics to guide review generation
  review_characteristics:
    - "Generate reviews with varied sentiment that naturally matches each rating level"
    - "1-2 star reviews should be critical and negative"
    - "3 star reviews should be neutral or mixed"
    - "4-5 star reviews should be positive and satisfied"
  
  # Retry mechanism configuration
  max_retry_rounds: 3        # Max attempts to replace rejected reviews
  retry_batch_size: 10       # Batch size for retry attempts
```

#### **CSV Configuration (Few-Shot Examples)**

```yaml
csv_configuration:
  csv_path: "./dataset/Amazon_Reviews.csv"
  rating_column: "Rating"
  review_column: "Review Text"
```

#### **Quality Guardrails Configuration**

```yaml
quality_guardrails:
  # Semantic similarity for duplicate detection
  embedding_model: "all-MiniLM-L6-v2"
  enable_duplicate_check: true
  similarity_threshold: 0.80      # Cosine similarity threshold (0-1)
  min_reviews_before_check: 30    # Skip check for first N reviews
  
  # Sentiment-rating alignment validation
  enable_sentiment_check: true
  sentiment_thresholds:
    rating_1:  # 1-star: very negative
      min: 0.0
      max: 0.3
    rating_2:  # 2-star: negative
      min: 0.0
      max: 0.4
    rating_3:  # 3-star: neutral (permissive range)
      min: 0.0
      max: 1.0
    rating_4:  # 4-star: positive
      min: 0.6
      max: 1.0
    rating_5:  # 5-star: very positive
      min: 0.7
      max: 1.0
```

#### **Model Configuration**

```yaml
model_configuration:
  # Single model or list of models
  model_name: "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
  # or multiple:
  # model_name: ["model1", "model2", "model3"]
  
  # Generation parameters
  temperature: 0.8    # Randomness (0-1)
  top_p: 0.9         # Nucleus sampling
  base_url: "https://api.together.xyz/v1"
```

### 3. Run Generation
**First time:**
```bash
docker compose up --build
```

**Subsequent runs:**
```bash
docker compose up
```

### 4. Check Results
Outputs are saved in `outputs/{model_name}_{timestamp}/`:
- 📊 **QUALITY_REPORT.md** - Quality metrics and analysis
- 📄 **generated_reviews.csv** - All generated reviews
- 📋 **quality_metrics.json** - Raw metrics data

## What Happens

1. ✅ Generates synthetic reviews for each configured model
2. ✅ Validates reviews (duplicates, sentiment-rating alignment)
3. ✅ Stores in PostgreSQL database (filtered by model)
4. ✅ Calculates quality metrics automatically
5. ✅ Exports results to organized folders

---

## 📚 Documentation

### Design & Architecture
For detailed information about the system design, evolution, and technical decisions, see:
- **[`docs/DESIGN.md`](docs/DESIGN.md)** - Complete design documentation including:
  - Pipeline evolution and trade-offs
  - Quality guardrails implementation
  - Known limitations and challenges
  - Future improvements

### Model Comparison
We evaluated three LLM models for synthetic review generation:
- **Meta-Llama-3.1-8B-Instruct-Turbo** 🥇
- **Mistral-7B-Instruct-v0.3** 🥈
- **OpenAI GPT-OSS-20B** 🥉

**Results:** Llama-3.1-8B emerged as the winner with the best balance of speed (3m 51s), quality, and realism. GPT-OSS-20B achieved perfect distribution matching but at 15x the time cost.

For the full comparison report with detailed metrics and analysis, see:
- **[`docs/MODEL_COMPARISON.md`](docs/MODEL_COMPARISON.md)**

---

**Questions?** Open an issue or check the documentation above.