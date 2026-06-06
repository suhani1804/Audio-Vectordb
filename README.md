# 🔊 Audio Semantic Search — CLAP + Pinecone + UrbanSound8K

A Jupyter notebook that builds a fully functional **semantic audio search engine** from scratch. Search a database of sound clips using either a **text description** or an **audio query**, powered by joint audio-text embeddings.

---

## 🧠 How It Works

```
Text query: "dog barking"
        │
        ▼
  CLAP text encoder  ──►  512-d vector
                               │
                               ▼
                       Pinecone ANN search
                               │
                               ▼
              Top-K most similar audio clips
```

Both audio and text are projected into the **same 512-dimensional embedding space** by CLAP. Cosine similarity between a query vector and indexed audio vectors drives retrieval — no keyword matching needed.

---

## 🛠️ Stack

| Component | Tool | Notes |
|-----------|------|-------|
| Audio dataset | [UrbanSound8K](https://huggingface.co/datasets/danavery/urbansound8k) | 8,732 labeled clips, 10 urban sound classes |
| Embeddings | [LAION CLAP](https://huggingface.co/laion/larger_clap_general) | Contrastive Language-Audio Pretraining, 512-d |
| Vector DB | [Pinecone](https://pinecone.io) | Serverless, free tier, cosine similarity |
| 3D Visualization | UMAP + Plotly | Interactive 3D embedding space explorer |

---

## 📓 Notebook Sections

| # | Section | Description |
|---|---------|-------------|
| 1 | Install dependencies | All required packages |
| 2 | Imports & config | API keys, hyperparameters |
| 3 | Load CLAP | Load `laion/larger_clap_general` from HuggingFace |
| 4 | Stream dataset | Stream UrbanSound8K from HuggingFace |
| 5 | Audio helpers | Resample, pad/clip, waveform + spectrogram plot |
| 6 | Inspect sample | Visualise waveform, mel spectrogram, play audio |
| 7 | CLAP embeddings | `embed_audio_batch()` and `embed_text()` functions |
| 8 | Pinecone index | Create serverless index on AWS us-east-1 |
| 9 | Ingest loop | Embed and upsert all clips with metadata |
| 10 | Text → audio search | Query by natural language description |
| 11 | Audio → audio search | Query by uploading a sound clip |
| 12 | Filtered search | Metadata pre-filtering (class, duration) |
| 13 | Similarity heatmap | Cosine similarity matrix across sound categories |
| 14 | UMAP 3D visualization | Interactive 3D cluster plot of embedding space |
| 15 | Clean up | Delete index to free storage |

---

## 🚀 Quickstart

### 1. Clone

```bash
git clone https://github.com/your-username/audio-semantic-search
cd audio-semantic-search
```

### 2. Install dependencies

```bash
pip install transformers datasets soundfile librosa torch torchaudio
pip install pinecone tqdm matplotlib ipython
pip install nomic umap-learn plotly pandas
pip install "datasets<2.18"   # avoids torchcodec dependency
```

### 3. Set your Pinecone API key

In cell 2, replace:
```python
PINECONE_API_KEY = "YOUR_PINECONE_API_KEY"
```
Get a free key at [pinecone.io](https://pinecone.io).

### 4. Run the notebook

```bash
jupyter notebook audio_semantic_search.ipynb
```

---

## 🔍 Example Queries

```
"dog barking"             → dog_bark clips, score ~0.45
"siren wailing"           → siren clips, score ~0.51
"drilling noise"          → drilling clips, score ~0.48
"children playing outside"→ children_playing clips, score ~0.42
"car engine idling"       → engine_idling clips, score ~0.44
```

---

## 📊 Visualizations

**Mel spectrogram** — inspect any audio clip visually

**Cosine similarity heatmap** — see how CLAP clusters the 10 UrbanSound8K classes in text embedding space

**UMAP 3D scatter plot** — interactive Plotly chart of all indexed audio embeddings colored by class label. Rotate, zoom, hover for clip ID and label.

---

## ⚙️ Configuration

All key parameters are in cell 2:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_SAMPLES` | 500 | Number of clips to index (max 8732) |
| `BATCH_SIZE` | 16 | Clips per embedding forward pass |
| `MAX_DURATION_SEC` | 10 | Clips longer than this are truncated |
| `SAMPLE_RATE` | 48000 | CLAP expects 48 kHz audio |
| `EMBEDDING_DIM` | 512 | CLAP output dimension |

---

## 📦 UrbanSound8K Classes

| ID | Class |
|----|-------|
| 0 | air_conditioner |
| 1 | car_horn |
| 2 | children_playing |
| 3 | dog_bark |
| 4 | drilling |
| 5 | engine_idling |
| 6 | gun_shot |
| 7 | jackhammer |
| 8 | siren |
| 9 | street_music |

---

## 📝 Notes

- **Free tier limits**: Pinecone's free serverless tier supports ~100k vectors, plenty for this dataset
- **GPU recommended**: Embedding 8k clips on CPU takes ~20 min; a Colab T4 GPU reduces this to ~3 min
- **Swap dataset**: Replace UrbanSound8K with any HuggingFace audio dataset by changing the `load_dataset` call and label field names
- **torchcodec**: If you hit torchcodec errors, install `datasets<2.18` which uses soundfile instead

---

## 📄 License

MIT
