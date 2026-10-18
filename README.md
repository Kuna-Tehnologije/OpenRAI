# Secondary-node training corpora

This starter set contains 30 labeled examples for each of the first six built-in expert nodes:

| File | Expert node | Focus |
| --- | --- | --- |
| `stability.txt` | E0:stability | Service health, failover, and recovery |
| `anomaly.txt` | E1:anomaly | Outliers, drift, and unusual activity |
| `memory.txt` | E2:memory | History retrieval and persistent context |
| `throughput.txt` | E3:throughput | Queues, latency, and processing capacity |
| `planning.txt` | E4:planning | Prioritized, safe operational steps |
| `reflection.txt` | E5:reflection | Evidence checks and confidence |

Each line uses the format `<importance><TAB><example>`, where importance is a value from `0.0` to `1.0`. The examples are in English to match the built-in domain vocabulary and generated corpora.

## Location and use

The files live in `data/corpus/` relative to the working directory. Run the program from the repository root to use them with the default configuration. Set `OPENRAI_CORPUS_DIR` if you keep the files elsewhere. To select these six nodes in PowerShell and retrain any cached models from the current corpora:

```powershell
$env:OPENRAI_EXPERTS = "6"
$env:OPENRAI_RETRAIN = "1"
```

Then start the OpenRAI executable as usual. The engine loads `<domain>.txt` for each expert; a missing or empty file is generated automatically. Existing model files are otherwise reused, so `OPENRAI_RETRAIN=1` is needed when you want edits to these examples to affect training.

These are curated starter corpora for experimentation, not a substitute for a large production dataset. Add domain-relevant examples in the same format to improve coverage.
