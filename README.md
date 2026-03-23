# LEGO Classifier

Detect and identify LEGO pieces from images. Upload a photo, get bounding boxes, BrickLink part IDs, confidence scores, and a bill of materials.

## Architecture

```
Browser ──POST /api/analyze──> Next.js (Vercel) ──POST /api/analyze──> Python FastAPI (Railway/Render/Fly)
                                 │                                        │
                                 │  Validates image, proxies request      │  YOLOv8 detection
                                 │  Transforms response for frontend     │  CLIP embedding
                                 │                                        │  Cosine similarity matching
                                 ▼                                        ▼
                              Canvas overlay                        parts.json + embeddings.npy
                              Results panel                         (reference index)
```

**Two services:**
- **Frontend** (`web/`) — Next.js 15, React 19, Tailwind CSS 4. Deploys to Vercel.
- **ML API** (root) — FastAPI, YOLOv8m, CLIP ViT-B/32. Deploys to any Docker host (Railway, Render, Fly.io).

## Prerequisites

- Python 3.11+
- Node.js 20+ (LTS)

## Local Setup

### 1. Install dependencies

```bash
# Python (from project root)
pip install -r requirements.txt

# Node.js (from web/)
cd web
npm install
```

### 2. Build the embedding index

```bash
python scripts/build_embeddings.py
```

This downloads part images from Rebrickable, runs them through CLIP, and writes `data/embeddings.npy` + `data/embeddings_index.json`. Takes ~5-10 minutes.

### 3. Run both services

```bash
# Option A: Demo mode (no ML models, fake detections)
# Windows:
start_demo.bat
# Linux/Mac:
./start_demo.sh

# Option B: Full mode (requires embeddings built in step 2)
# Terminal 1 — Python API on port 8000
python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Next.js on port 3000
cd web
npm run dev
```

Open http://localhost:3000 and upload an image.

## Environment Variables

### Next.js (`web/.env.local`)

| Variable | Default | Description |
|----------|---------|-------------|
| `ML_SERVICE_URL` | `http://localhost:8000` | URL of the Python ML API |

### Python API

| Variable | Default | Description |
|----------|---------|-------------|
| `DEMO_MODE` | _(off)_ | Set to `1` to return fake detections without loading models |
| `CORS_ORIGINS` | _(none)_ | Comma-separated production origins, e.g. `https://my-app.vercel.app` |

## Deploy to Production

### Step 1: Deploy the Python ML API

The ML service needs a host with **2GB+ RAM** (for YOLO + CLIP models). Choose one:

#### Railway

```bash
# Install Railway CLI: https://docs.railway.com/guides/cli
railway login
railway init
railway up
```

Set environment variables in Railway dashboard:
- `CORS_ORIGINS` = `https://your-app.vercel.app`

#### Render

1. Create a new **Web Service** from the repo
2. Set **Root Directory** to `.` (project root)
3. Set **Build Command**: `pip install -r requirements.txt`
4. Set **Start Command**: `uvicorn server:app --host 0.0.0.0 --port $PORT`
5. Add env var: `CORS_ORIGINS` = `https://your-app.vercel.app`
6. Choose an instance with 2GB+ RAM

#### Fly.io

```bash
fly launch --dockerfile Dockerfile
fly secrets set CORS_ORIGINS=https://your-app.vercel.app
fly deploy
```

#### Docker (any host)

```bash
docker build -t lego-classifier-api .
docker run -p 8000:8000 \
  -e CORS_ORIGINS=https://your-app.vercel.app \
  lego-classifier-api
```

**After deploying**, note the service URL (e.g. `https://lego-api.railway.app`).

### Step 2: Build embeddings on the server

SSH into your ML service or run as a one-off task:

```bash
python scripts/build_embeddings.py
```

This generates `data/embeddings.npy` and `data/embeddings_index.json`. For Docker, mount a volume or bake the embeddings into the image.

### Step 3: Deploy the frontend to Vercel

```bash
cd web
npx vercel
```

Set environment variables in Vercel dashboard (or via CLI):

```bash
vercel env add ML_SERVICE_URL  # paste your ML API URL, e.g. https://lego-api.railway.app
```

Redeploy after adding the env var:

```bash
vercel --prod
```

### Step 4: Verify

```bash
# Health check
curl https://your-app.vercel.app/api/health

# Should return:
# {"status":"healthy","ml_service":"connected","matcher_ready":true,"indexed_parts":35}
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/analyze` | Upload image, get detections + part matches |
| `GET` | `/api/health` | Health check with model status |
| `GET` | `/api/parts` | List all indexed parts (optional `?category=Brick`) |
| `GET` | `/api/parts/{id}` | Get a single part by BrickLink ID |

## Performance Notes

- **Model sizes**: YOLOv8m (~50MB), CLIP ViT-B/32 (~340MB). First request triggers model download.
- **Cold start**: ~5-10s on first request (model loading). Subsequent requests: ~1-3s.
- **Memory**: ~1.5GB at runtime with both models loaded.
- **Vercel timeout**: The `analyze` route is configured for 60s max duration (requires Vercel Pro for >10s).
- **Demo mode**: Set `DEMO_MODE=1` on the ML API for testing without loading models (~0ms latency).

## Project Structure

```
lego_classifier/
├── server.py              # FastAPI ML service
├── Dockerfile             # Docker config for ML service
├── requirements.txt       # Python dependencies
├── run_demo_api.py        # Demo mode launcher
├── start_demo.bat/.sh     # Start both services in demo mode
├── ml/
│   ├── detector.py        # YOLOv8 object detection
│   ├── embedder.py        # CLIP image embedding
│   ├── matcher.py         # Cosine similarity part matching
│   └── pipeline.py        # End-to-end orchestrator
├── data/
│   ├── parts.json         # Part catalog (35-part seed)
│   ├── embeddings.npy     # Precomputed CLIP embeddings (generated)
│   └── embeddings_index.json  # Embedding-to-part mapping (generated)
├── scripts/
│   ├── build_embeddings.py    # Build the reference index
│   └── test_pipeline.py      # Integration test
└── web/                   # Next.js frontend (deploys to Vercel)
    ├── vercel.json        # Vercel project config
    ├── next.config.ts
    ├── package.json
    └── src/
        ├── app/
        │   ├── page.tsx           # Main page
        │   └── api/
        │       ├── analyze/route.ts   # Image upload proxy
        │       ├── health/route.ts    # Health check proxy
        │       └── parts/route.ts     # Parts catalog proxy
        ├── components/
        │   ├── ImageUploader.tsx   # Drag-and-drop upload
        │   ├── PreviewCanvas.tsx   # Bounding box overlay
        │   └── ResultsPanel.tsx    # Results display
        ├── hooks/
        │   └── useAnalyze.ts      # Upload + state hook
        └── lib/
            ├── ml-client.ts       # Python service HTTP client
            └── types.ts           # Shared TypeScript types
```
