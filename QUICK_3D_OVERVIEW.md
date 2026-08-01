# OpenRAI – 3D AI u 60 sekundi

## Što je "3D AI"?

```
        DEPTH (Semantic Layers)
         ?
         ?  ?? Goals (alignment)
         ?  ?? Knowledge Graph
         ?  ?? Vector Memory (similarity)
         ?  ?? Embedding (32-dim)
         ?  ?? Tokenizer (text?tokens)
         ?
         ??????????????????????????? WIDTH (Time/Iterations)
        ?                            ?
       ?  HEIGHT                     ?
      ?   (Batch)                    ?? Persistent learning
     ?    500 nodes                     (vector_memory.db)
    ?     @ 10k/sec
```

---

## 3 Dimenzije = 3 Problema Riješena

### 1?? **HEIGHT** ? Skalabilnost
```
Problem: Jedan node po threadu = sporo
Rješenje: 500 nodes × GPU batch = 10,000 node/sec
```

### 2?? **WIDTH** ? Temporalno uèenje
```
Problem: Svaka iteracija zaboravlja prošlost
Rješenje: Vector memory (5000 embeddings) + persistent storage
```

### 3?? **DEPTH** ? Semantièko razumijevanje
```
Problem: String matching = površno
Rješenje: 5-layer stack (Tokenizer?Embedding?Vector?Graph?Goals)
```

---

## Jedan Cycle (50ms)

```
INPUT: "alert: memory usage 80% node 7"
  ?
[HEIGHT] 500 nodes × GPU inference (500µs)
  ? relevance=0.82, importance=0.91
  ?
[DEPTH] Semantic processing:
  ? Tokens: [alert, memory, usage, 80, node, 7]
  ? Embedding: [0.12, -0.34, ..., 0.21] (32 floats)
  ? Vector store + Graph update + Goal scoring
  ?
[WIDTH] Memory flow:
  ? Current memory (buffer)
  ? Short-term (top-30)
  ? Reflection (top-10)
  ? Vector search: Find 4 similar past events
  ?
OUTPUT: "Conclusion: alert memory usage... | confidence=0.88"
  ? Stored in vector_memory.db (persistent)
```

---

## Metrics

| Dimenzija | Metrika | Vrijednost |
|-----------|---------|------------|
| **HEIGHT** | Throughput | 10,000 node/sec |
| **HEIGHT** | Batch size | 500 nodes/iter |
| **WIDTH** | Loop speed | 50ms (20 iter/sec) |
| **WIDTH** | Vector memory | 5000 embeddings |
| **DEPTH** | Layers | 5 (Token?Embed?Vector?Graph?Goals) |
| **DEPTH** | Embedding dim | 32 floats |
| Overall | NN params | 1.3M (32?256?512?256?128?4) |
| Overall | Input pool | 100k CSV samples |

---

## Usporedba

| AI Sistem | Tip | Dimenzije |
|-----------|-----|-----------|
| GPT-4 | LLM | 1D (stateless) |
| ChatGPT | Chatbot | 1.5D (1D + history) |
| AutoGPT | Agent | 2D (1D + RAG) |
| **OpenRAI** | **Engine** | **3D (Batch × Time × Semantic)** |

---

## 3 Kljuèna Fajla

1. **3D_AI_INFRASTRUCTURE.md** – Detaljna arhitektura
2. **README.md** – Quick start + komponente
3. **src/core.cpp** – Main loop implementacija

---

## Zašto "3D" a ne "multi-layer"?

**"Multi-layer"** = samo DEPTH (semantièki stack)

**"3D"** = HEIGHT + WIDTH + DEPTH = **ortogonalne dimenzije**:
- HEIGHT i WIDTH su nezavisni
- DEPTH dodaje znaèenje obje druge dimenzije
- Svaka dimenzija rješava drugaèiji problem

**Analogija:**
- 1D AI = toèka (jedan output)
- 2D AI = linija (history)
- 3D AI = prostor (batch × time × semantic)

---

**TL;DR:**  
OpenRAI = **Batch GPU inference** (HEIGHT) + **Persistent vector memory** (WIDTH) + **5-layer semantic stack** (DEPTH)
