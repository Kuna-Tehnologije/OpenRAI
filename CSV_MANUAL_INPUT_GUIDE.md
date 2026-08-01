# Step Input Pool - Ruèno Unošenje

## Što je promijenjeno

OpenRAI sada **uèitava inpute SAMO iz CSV-a** - bez automatskog generiranja.

### Prije:
- Automatski generirao 100,000 test inputa
- Pisao u `data/step_input_pool.csv`
- Fallback na generirane inpute ako CSV nije bilo

### Sad:
- ? **Samo uèitava** iz `data/step_input_pool.csv`
- ? **Ne generira** inpute automatski
- Ti ruèno ureðuješ CSV

## Kako dodati inpute

### Format CSV-a

```csv
input
"Prvi input ovdje"
"Drugi input ovdje"
"Treæi input ovdje"
```

**Važno:**
- Prva linija mora biti `input` (header)
- Svaki input **mora** biti u navodnicima `"..."`
- Svaki input je nova linija

### Primjer `data/step_input_pool.csv`:

```csv
input
"What is artificial intelligence?"
"Explain machine learning concepts"
"How does a neural network process information?"
"Define supervised learning"
"What are the main types of AI?"
"Describe natural language processing"
"How do transformers work?"
"Explain backpropagation in neural networks"
"What is reinforcement learning?"
"Compare CNN and RNN architectures"
```

### Dodavanje novih inputa:

1. Otvori `data/step_input_pool.csv` u bilo kojem editoru
2. Dodaj nove linije sa formatom: `"tvoj input ovdje"`
3. Saèuvaj fajl
4. Pokreni OpenRAI

### Napomene:

- **Escape karakteri**: Ako imaš navodnike u tekstu, escape-aj ih:
  ```csv
  "He said \"Hello\" to me"
  ```

- **Višelinijski tekst**: Drži ga u jednoj liniji ili koristi `\n`:
  ```csv
  "First line\nSecond line\nThird line"
  ```

- **Prazne linije**: Automatski se preskaèu

## Logging

Kad OpenRAI startuje, vidjet æeš:
```
[InputPool] Loaded 42 inputs from CSV
```

Ako CSV ne postoji:
```
[InputPool] CSV file not found: data/step_input_pool.csv
[InputPool] Loaded 0 inputs from CSV
```

## Ako CSV ne postoji

Kreirati novi `data/step_input_pool.csv`:

```cmd
mkdir data
echo input > data\step_input_pool.csv
echo "Sample input 1" >> data\step_input_pool.csv
echo "Sample input 2" >> data\step_input_pool.csv
```

Ili ruèno kreiraj fajl u editoru.

## Rebuild potreban

Da bi promjene stupile na snagu:

```cmd
cmake --build C:\dev\openRAI\build --target openrai_chat_gui --config Release
```

## Gdje se koristi

Ovi inputi se koriste u `OpenRAI::step()` funkciji za batch procesiranje:
- Uèitavaju se pri startu
- Koriste se za treniranje memorije i knowledge graph-a
- Rotiraju se tokom rada sistema

Sad je potpuno pod tvojom kontrolom! ??
