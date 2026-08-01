# Training Estimate: OpenRAI 3.98B for English Language Recognition

## Executive Summary

**Quick Answer:**
- **Minimum (basic English):** 2-4 weeks on 1× RTX 4090
- **Production-quality:** 2-3 months on 8× A100
- **SOTA (state-of-art):** 6-12 months on cluster (64-256 GPUs)

---

## What "English Recognition" Means

### Disambiguation

**Option A: Text Classification** (Easiest)
- Input: Raw text
- Output: [relevance, importance, should_output, priority]
- Task: Score English sentences for reasoning quality
- **This document focuses on Option A**

**Option B: Language Model** (Harder)
- Input: Text sequence
- Output: Next token prediction
- Task: Generate/complete English text
- Requires architecture change (not current OpenRAI design)

**Option C: Speech Recognition** (Different domain)
- Input: Audio waveform
- Output: Text transcription
- Requires audio preprocessing (not in scope)

---

## Training Requirements Breakdown

### 1. Dataset

#### Size Requirements

| Quality Level | Samples | Text Volume | Storage |
|---------------|---------|-------------|---------|
| **Proof-of-Concept** | 1M | ~500MB | 1GB |
| **Basic** | 10M | ~5GB | 10GB |
| **Good** | 100M | ~50GB | 100GB |
| **Production** | 1B | ~500GB | 1TB |
| **SOTA** | 10B+ | ~5TB | 10TB |

#### Dataset Sources (Public)

1. **Common Crawl** (web text)
   - Size: Petabytes
   - Quality: Mixed
   - Cost: Free
   - URL: https://commoncrawl.org/

2. **Wikipedia** (English)
   - Size: ~20GB compressed
   - Quality: High
   - Cost: Free
   - URL: https://dumps.wikimedia.org/

3. **BookCorpus**
   - Size: ~5GB
   - Quality: Very high
   - Cost: Free (research)
   - Note: Books dataset

4. **OpenWebText** (GPT-2 dataset replica)
   - Size: ~38GB
   - Quality: High
   - Cost: Free
   - URL: https://skylion007.github.io/OpenWebTextCorpus/

5. **The Pile** (EleutherAI)
   - Size: 825GB
   - Quality: Curated
   - Cost: Free
   - URL: https://pile.eleuther.ai/

**Recommendation for OpenRAI:**
- Start with **The Pile** (825GB, diverse, curated)
- Subset: 100M samples (~50GB) for initial training
- Full dataset: 1B+ samples for production

---

### 2. Hardware Requirements

#### GPU Options

| GPU Model | VRAM | FP32 Speed | FP16 Speed | Price | Availability |
|-----------|------|------------|------------|-------|--------------|
| **RTX 4090** | 24GB | 82.6 TFLOPS | 165 TFLOPS | $1,600 | Consumer |
| **RTX 3090** | 24GB | 35.6 TFLOPS | 71 TFLOPS | $1,000 | Consumer |
| **A100 (40GB)** | 40GB | 19.5 TFLOPS | 312 TFLOPS | $10k | Data center |
| **A100 (80GB)** | 80GB | 19.5 TFLOPS | 312 TFLOPS | $15k | Data center |
| **H100** | 80GB | 51 TFLOPS | 1000 TFLOPS | $30k | Data center |

**Memory Calculation (3.98B params):**
```
Weights:           16GB (float32)
Gradients:         16GB (float32)
Optimizer (Adam):  32GB (2× params for momentum + variance)
Activations:       ~2GB (batch=32, layer buffers)
-------------------------------------------
Total:             ~66GB minimum
```

**Conclusion:** Need ?80GB VRAM OR multi-GPU with gradient checkpointing

#### Single GPU Training (RTX 4090)

**Configuration:**
- Mixed precision (FP16)
- Gradient checkpointing
- Batch size: 8 (small)
- Gradient accumulation: 16 steps

**Memory footprint:**
```
Weights (FP16):    8GB
Gradients (FP16):  8GB
Optimizer (FP16):  16GB (mixed precision master weights)
Activations:       2GB (checkpointing enabled)
-------------------------------------------
Total:             ~34GB ? DOESN'T FIT in 24GB ??
```

**Workaround:**
- Use **CPU offloading** (DeepSpeed ZeRO-Offload)
- Store optimizer states on CPU RAM
- Reduces GPU memory to ~18GB ?
- Cost: 30-50% slower training

**Time estimate (1× RTX 4090 + CPU offload):**
- **10M samples:** 2-3 weeks
- **100M samples:** 2-3 months
- **1B samples:** 1-2 years ??

---

#### Multi-GPU Training (Recommended)

**Setup 1: 4× RTX 4090**
- Total VRAM: 96GB
- Data parallelism (DDP)
- Batch size: 32 (8 per GPU)
- **Cost:** ~$6,400
- **Time (100M samples):** 2-3 weeks

**Setup 2: 8× A100 (40GB)**
- Total VRAM: 320GB
- Pipeline parallelism + DDP
- Batch size: 256 (32 per GPU)
- **Cost:** ~$80,000 (purchase) OR $10-20/hr (cloud)
- **Time (100M samples):** 3-7 days

**Setup 3: Cloud (Lambda Labs / RunPod)**
- 8× A100 (80GB)
- **Cost:** $15/hr × 8 = $120/hr
- **Time (100M samples):** 5-10 days
- **Total cost:** $120 × 24 × 7 = $20,160

---

### 3. Training Time Calculations

#### Assumptions
- Model: 3.98B parameters (12 layers)
- Dataset: 100M English text samples
- Batch size: 256 (effective, multi-GPU)
- Hardware: 8× A100 (40GB)
- Precision: Mixed (FP16)

#### Forward + Backward Pass Timing

**Per-sample complexity:**
```
Forward pass:  12 layers × 24k neurons ? 3.98B FLOPs
Backward pass: 3× forward (gradients) ? 12B FLOPs
Total:         ~16B FLOPs per sample
```

**Per-batch (256 samples):**
```
16B × 256 = 4.1 TFLOPs (4.1 trillion floating-point ops)
```

**Time per batch (8× A100):**
```
A100 FP16 throughput: 312 TFLOPS
Effective utilization: ~50% (memory bandwidth limited)
Actual throughput:     156 TFLOPS per GPU × 8 = 1,248 TFLOPS

Time per batch: 4.1 TFLOPS ÷ 1,248 TFLOPS = 0.0033 sec = 3.3ms
```

**But wait — memory ops dominate!**
- Weight loading: ~16GB per layer
- Realistic time: **50-100ms per batch** (memory bound)

#### Total Training Time

**Epochs needed:**
- Basic convergence: 3 epochs
- Good quality: 10 epochs
- Production: 20-30 epochs

**Batches per epoch (100M samples):**
```
100,000,000 ÷ 256 = 390,625 batches
```

**Time per epoch (100ms/batch):**
```
390,625 × 0.1 sec = 39,062 sec = 10.8 hours
```

**Total training time:**
| Epochs | Time (8× A100) | Cloud Cost ($15/hr × 8) |
|--------|----------------|-------------------------|
| **3 (PoC)** | **32 hours (1.3 days)** | **$3,840** |
| **10 (Good)** | **108 hours (4.5 days)** | **$12,960** |
| **30 (Prod)** | **324 hours (13.5 days)** | **$38,880** |

---

### 4. Training Pipeline Implementation

#### What's Currently Missing in OpenRAI

**Implemented:**
- ? Forward pass (inference)
- ? Layer definitions
- ? Weight initialization (Xavier)
- ? Batch processing

**NOT Implemented (Required for Training):**
- ? Backpropagation (gradient computation)
- ? Optimizer (Adam/SGD)
- ? Loss function
- ? Data loader (batch sampling)
- ? Training loop
- ? Checkpoint saving/loading
- ? Validation/evaluation
- ? Learning rate scheduling

#### Development Time to Add Training

**Minimal training pipeline:**
- Backprop implementation: 1-2 weeks
- Optimizer (Adam): 3-5 days
- Data loader: 3-5 days
- Training loop: 1 week
- Checkpointing: 2-3 days
- **Total:** 4-6 weeks development

**Production-ready training:**
- + Distributed training (DDP): 2-3 weeks
- + Mixed precision (FP16): 1 week
- + Gradient checkpointing: 1 week
- + Data pipeline optimization: 2 weeks
- + Monitoring/logging: 1 week
- + Hyperparameter tuning: 2-4 weeks
- **Total:** 3-4 months development

---

### 5. Training Cost Breakdown

#### Option A: Self-Hosted (4× RTX 4090)

**Hardware:**
- 4× RTX 4090: $6,400
- Motherboard + CPU: $1,500
- RAM (256GB): $800
- Storage (2TB NVMe): $200
- PSU (2000W): $400
- **Total upfront:** $9,300

**Operating costs:**
- Electricity: 2kW × $0.15/kWh × 24hr = $7.20/day
- **Monthly:** ~$216

**Training time (100M samples, 10 epochs):**
- ~3-4 weeks
- **Cost:** Hardware + ($7.20 × 28) = $9,501

**Amortized (if reused):**
- Hardware lasts 3-5 years
- Can train multiple models
- **Effective cost/training:** $1,000-2,000

---

#### Option B: Cloud (Lambda Labs 8× A100)

**Pricing:**
- 8× A100 (40GB): $12.00/hr (Lambda Labs)
- OR 8× A100 (80GB): $14.40/hr (RunPod)

**Training time (100M samples, 10 epochs):**
- ~5-7 days
- **Hours:** 120-168 hrs

**Total cost:**
- $12.00 × 144 hrs = **$1,728** (mid estimate)

**Pros:**
- No upfront cost
- Start immediately
- Professional infrastructure

**Cons:**
- Ongoing cost
- Need good internet (upload dataset)

---

#### Option C: Cloud (GCP/AWS/Azure)

**Pricing (GCP):**
- 1× A100 (40GB): $3.673/hr
- 8× A100: $29.38/hr
- Preemptible (spot): ~$8-12/hr for 8× A100

**Training time (100M samples, 10 epochs):**
- ~5-7 days
- **Cost (spot instances):** $10/hr × 144 hrs = **$1,440**

**Additional costs:**
- Storage (1TB): $20/month
- Egress (downloading model): ~$50

**Total:** ~$1,500-2,000

---

### 6. Dataset Preparation Timeline

#### Steps Required

1. **Download dataset** (The Pile, 825GB)
   - Time: 1-3 days (depends on internet)
   - Storage: 1TB SSD

2. **Preprocessing**
   - Tokenization (split into words/subwords)
   - Filtering (remove non-English, spam)
   - Chunking (split into training samples)
   - Time: 2-5 days (CPU-intensive)

3. **Create training labels**
   - For OpenRAI: [relevance, importance, should_output, priority]
   - Options:
     - Manual labeling: 100k samples = 200 hours (crowdwork)
     - Semi-supervised: Use heuristics + small labeled set
     - Self-supervised: Predict masked tokens (requires arch change)
   - Time: 1-4 weeks

4. **Data augmentation** (optional)
   - Paraphrasing
   - Back-translation
   - Noise injection
   - Time: 1 week

**Total data prep time:** 3-8 weeks

---

## Complete Training Timeline

### Scenario 1: Proof-of-Concept (10M samples)

**Goal:** Demonstrate that model can learn English patterns

| Phase | Duration | Resources |
|-------|----------|-----------|
| **Dataset prep** | 1 week | 1TB storage |
| **Training code** | 4 weeks | 1 developer |
| **Training** | 2 weeks | 1× RTX 4090 + CPU offload |
| **Evaluation** | 3 days | Same GPU |
| **TOTAL** | **7-8 weeks** | **~$2,000** |

---

### Scenario 2: Production Quality (100M samples)

**Goal:** Competitive English reasoning model

| Phase | Duration | Resources |
|-------|----------|-----------|
| **Dataset prep** | 4 weeks | 2TB storage |
| **Training code** | 8 weeks | 2 developers |
| **Training** | 1 week | 8× A100 (cloud) |
| **Evaluation** | 1 week | Same |
| **Hyperparameter tuning** | 2 weeks | Same |
| **TOTAL** | **16 weeks (4 months)** | **~$15,000-20,000** |

---

### Scenario 3: State-of-the-Art (1B samples)

**Goal:** Compete with GPT-2/BERT-Large

| Phase | Duration | Resources |
|-------|----------|-----------|
| **Dataset curation** | 8 weeks | 10TB storage |
| **Training infra** | 12 weeks | ML Ops team |
| **Training** | 6 weeks | 64× A100 (cluster) |
| **Evaluation** | 4 weeks | Research team |
| **Tuning** | 8 weeks | Same |
| **TOTAL** | **38 weeks (~9 months)** | **$200,000-500,000** |

---

## Comparison with Industry Timelines

| Model | Parameters | Training Time | Hardware | Cost |
|-------|-----------|---------------|----------|------|
| **OpenRAI (target)** | **3.98B** | **4-16 weeks** | **8-64× GPU** | **$15k-500k** |
| BERT-Base | 110M | 4 days | 16× TPU | $7k |
| BERT-Large | 340M | 4 days | 64× TPU | $30k |
| GPT-2 | 1.5B | ~1 week | 256× V100 | $50k |
| GPT-3 | 175B | 1 month+ | 10,000× V100 | $4.6M |
| LLaMA 7B | 7B | 21 days | 2,048× A100 | $500k |

**OpenRAI positioning:**
- Larger than BERT-Large (3.98B vs. 340M)
- Smaller than GPT-3 (3.98B vs. 175B)
- Similar scale to GPT-2 (1.5B)

**Realistic target:** Match GPT-2 quality in 4-6 months with $20k-50k budget

---

## Recommended Approach (Pragmatic)

### Phase 1: Minimum Viable Training (Week 1-8)

**Goal:** Prove the model can learn

1. **Implement training code** (Week 1-4)
   - Backpropagation (CUDA kernels)
   - Adam optimizer
   - Simple data loader

2. **Small-scale training** (Week 5-8)
   - Dataset: 1M Wikipedia sentences
   - Hardware: 1× RTX 4090 (CPU offload)
   - Target: Loss decreases, sensible outputs

**Cost:** $1,000 (development time) + $500 (GPU/electricity)

---

### Phase 2: Scale Up (Week 9-16)

**Goal:** Train on 100M samples

1. **Data pipeline** (Week 9-10)
   - Download The Pile subset (100GB)
   - Preprocessing + chunking

2. **Cloud training** (Week 11-12)
   - Rent 8× A100 (Lambda Labs)
   - Train for 5-7 days
   - Monitor loss curves

3. **Evaluation** (Week 13-14)
   - Benchmark on test set
   - Compare with baselines

4. **Iteration** (Week 15-16)
   - Adjust hyperparameters
   - Re-train if needed

**Cost:** $2,000 (labor) + $2,000 (cloud) = $4,000

---

### Phase 3: Production (Week 17-24)

**Goal:** Deploy a competitive model

1. **Full dataset training** (Week 17-20)
   - 1B samples from The Pile
   - 8× A100 for 2-3 weeks

2. **Fine-tuning** (Week 21-22)
   - Task-specific datasets
   - Smaller learning rate

3. **Deployment** (Week 23-24)
   - Model quantization (INT8)
   - Inference optimization

**Cost:** $15,000-20,000 (mostly cloud GPU)

---

## Summary Table

| Scenario | Dataset Size | Training Time | Hardware | Total Cost | Quality |
|----------|--------------|---------------|----------|------------|---------|
| **PoC** | 1-10M | 2-3 weeks | 1× RTX 4090 | $1,500 | Basic |
| **Good** | 100M | 1-2 weeks | 8× A100 | $5,000 | Competitive |
| **Production** | 1B | 2-3 weeks | 64× A100 | $50,000 | SOTA-adjacent |

---

## Answer to Your Question

### **"Koliko bi mi trebalo da istreniram sustav za prepoznavanje engleskog?"**

**Realistic answer:**

**Minimum (can demonstrate learning):**
- **Time:** 2 months (1 month dev + 1 month training)
- **Cost:** $2,000-5,000
- **Hardware:** 1× RTX 4090 OR cloud spot instances

**Production-ready (competitive quality):**
- **Time:** 4-6 months (2 months dev + 2-4 months training/tuning)
- **Cost:** $15,000-30,000
- **Hardware:** 8× A100 (cloud) OR 4× RTX 4090 (self-hosted)

**World-class (SOTA):**
- **Time:** 9-12 months (team of 3-5 engineers)
- **Cost:** $200,000-500,000
- **Hardware:** 64-256× A100 cluster

---

## Bottlenecks to Watch

1. **Memory:** 3.98B params need ?16GB GPU (or CPU offload)
2. **Data labeling:** Creating training targets is slow (consider self-supervised)
3. **Training code:** 4-8 weeks to implement backprop + optimizer
4. **Hyperparameters:** Expect 5-10 training runs to tune learning rate, batch size, etc.

---

## Quick Start Checklist

- [ ] Decide target quality (PoC / Good / Production)
- [ ] Allocate budget ($2k / $20k / $200k)
- [ ] Choose hardware (local 4090 vs. cloud A100)
- [ ] Download dataset (The Pile recommended)
- [ ] Implement backprop (4-6 weeks)
- [ ] Start small (1M samples) to debug
- [ ] Scale up gradually (10M ? 100M ? 1B)
- [ ] Monitor loss curves religiously
- [ ] Be ready to iterate 5-10 times

---

**Shortest path: 2 months + $5,000 for decent English recognition** ?
