# Miles MVP: Implementation Plan
**Senior Engineering Perspective**
**Target Timeline**: 10-12 days (assuming part-time availability)
**Last Updated**: Dec 03, 2025

---

## Table of Contents
1. [System Architecture Overview](#system-architecture-overview)
2. [Prerequisites & Environment Setup](#prerequisites--environment-setup)
3. [Phase 1: Foundation & Infrastructure](#phase-1-foundation--infrastructure-days-1-3)
4. [Phase 2: Data Pipeline](#phase-2-data-pipeline-days-4-5)
5. [Phase 3: ML Inference Service](#phase-3-ml-inference-service-days-6-8)
6. [Phase 4: Frontend & Intelligence Layer](#phase-4-frontend--intelligence-layer-days-9-10)
7. [Phase 5: Integration & Testing](#phase-5-integration--testing-days-11-12)
8. [Production Considerations](#production-considerations)
9. [Risk Mitigation](#risk-mitigation)

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        User Browser                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Frontend (Next.js on Vercel)                    │
│  - User input form (topic, vibe, preferences)                │
│  - Results display (virality scores + recommendations)       │
│  - Auth integration (Supabase Auth)                          │
└────────────┬────────────────────────┬───────────────────────┘
             │                        │
             │ API Routes             │ Edge Functions
             ▼                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Backend (Supabase)                              │
│  - Postgres DB (candidates, users, queries)                  │
│  - Edge Functions (recommendation logic, LLM wrapper)        │
│  - Row Level Security (RLS policies)                         │
└────────────┬────────────────────────┬───────────────────────┘
             │                        │
             │                        │ Query top patterns
             │                        ▼
             │              ┌─────────────────────┐
             │              │  LLM Service        │
             │              │  (OpenAI GPT-4o)    │
             │              └─────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│                   Data & Inference Layer                     │
│                                                               │
│  ┌──────────────────┐         ┌─────────────────────┐       │
│  │ Scraper Service  │────────▶│  Miles Inference    │       │
│  │ (GitHub Actions) │ Frames  │  (HuggingFace Space)│       │
│  │  - yt-dlp        │         │  - FastAPI          │       │
│  │  - Metadata      │         │  - ONNX Runtime     │       │
│  │  - Thumbnails    │         │  - BERT + ResNet    │       │
│  └────────┬─────────┘         └──────────┬──────────┘       │
│           │                               │                  │
│           └───────────┬───────────────────┘                  │
│                       │ Store scores                         │
│                       ▼                                      │
│              [Supabase Postgres]                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Prerequisites & Environment Setup

### Development Environment
```bash
# Required tools
- Node.js 18+ (for Next.js)
- Python 3.9+ (for ML inference & scraping)
- Docker (for local HuggingFace Space testing)
- Git (version control)
- Supabase CLI (for local dev)

# Accounts needed (all free tier)
- Vercel account (frontend deployment)
- Supabase account (backend/database)
- HuggingFace account (ML inference hosting)
- OpenAI account (LLM API, ~$5 budget for testing)
- GitHub (Actions for CRON jobs)
```

### Repository Structure
```
Miles/
├── frontend/                    # Next.js application
│   ├── src/
│   │   ├── app/                # App router pages
│   │   ├── components/         # React components
│   │   ├── lib/                # Utilities (Supabase client, etc.)
│   │   └── types/              # TypeScript types
│   ├── public/                 # Static assets
│   ├── next.config.js
│   └── package.json
│
├── inference-service/          # HuggingFace Space (FastAPI)
│   ├── app.py                  # FastAPI endpoints
│   ├── model_loader.py         # ONNX model loading
│   ├── preprocessing.py        # Image/text preprocessing
│   ├── requirements.txt
│   ├── Dockerfile
│   └── models/                 # .onnx files
│
├── scraper/                    # Data ingestion
│   ├── scraper.py              # yt-dlp wrapper
│   ├── db_uploader.py          # Supabase uploader
│   ├── requirements.txt
│   └── .env.example
│
├── supabase/                   # Supabase config
│   ├── migrations/             # SQL schema migrations
│   ├── functions/              # Edge Functions
│   │   └── generate-recommendations/
│   └── config.toml
│
├── scripts/                    # Utility scripts
│   ├── export_model_to_onnx.py
│   └── test_inference.py
│
├── .github/
│   └── workflows/
│       └── scraper-cron.yml    # GitHub Action for scraping
│
└── [Existing ML code]          # Your current src/, models/, etc.
```

---

## Phase 1: Foundation & Infrastructure (Days 1-3)

### Day 1: Backend Setup (Supabase)

**Objective**: Create database schema and authentication layer

#### 1.1 Initialize Supabase Project
```bash
# Install Supabase CLI
npm install -g supabase

# Initialize in project
cd /path/to/Miles
supabase init

# Link to your Supabase project (create one at supabase.com)
supabase link --project-ref <your-project-ref>
```

#### 1.2 Database Schema
Create migration file: `supabase/migrations/20251203_initial_schema.sql`

```sql
-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table (extends Supabase Auth)
CREATE TABLE public.users (
  id UUID REFERENCES auth.users(id) PRIMARY KEY,
  email TEXT UNIQUE NOT NULL,
  display_name TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Video candidates table (scraped data)
CREATE TABLE public.candidates (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  video_id TEXT UNIQUE NOT NULL,           -- YouTube video ID
  platform TEXT DEFAULT 'youtube',          -- 'youtube' or 'tiktok'
  title TEXT NOT NULL,
  description TEXT,
  thumbnail_url TEXT,
  channel_name TEXT,
  upload_date TIMESTAMPTZ,

  -- Engagement metrics (from scraper)
  view_count BIGINT,
  like_count BIGINT,
  comment_count BIGINT,

  -- Miles predictions
  miles_score FLOAT,                        -- Virality score [0,1]
  predicted_velocity FLOAT,                 -- Predicted view velocity

  -- Metadata
  tags TEXT[],
  duration_seconds INT,
  scraped_at TIMESTAMPTZ DEFAULT NOW(),
  analyzed_at TIMESTAMPTZ,                  -- When ML inference ran

  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- User queries table (track user requests)
CREATE TABLE public.queries (
  id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
  user_id UUID REFERENCES public.users(id),

  -- User input
  topic TEXT NOT NULL,                      -- e.g., "tech reviews"
  vibe TEXT,                                -- e.g., "funny", "educational"
  additional_params JSONB,                  -- Flexible storage

  -- System response
  recommendations JSONB,                    -- LLM-generated suggestions
  matched_candidates UUID[],                -- Array of candidate IDs

  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_candidates_miles_score ON public.candidates(miles_score DESC);
CREATE INDEX idx_candidates_platform_score ON public.candidates(platform, miles_score DESC);
CREATE INDEX idx_candidates_scraped_at ON public.candidates(scraped_at DESC);
CREATE INDEX idx_queries_user_id ON public.queries(user_id);

-- Row Level Security (RLS) Policies
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.candidates ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.queries ENABLE ROW LEVEL SECURITY;

-- Users can read their own data
CREATE POLICY "Users can read own data" ON public.users
  FOR SELECT USING (auth.uid() = id);

-- Anyone can read candidates (public data)
CREATE POLICY "Candidates are publicly readable" ON public.candidates
  FOR SELECT USING (true);

-- Only service role can insert/update candidates
CREATE POLICY "Service role can manage candidates" ON public.candidates
  FOR ALL USING (auth.role() = 'service_role');

-- Users can read their own queries
CREATE POLICY "Users can read own queries" ON public.queries
  FOR SELECT USING (auth.uid() = user_id);

-- Users can insert their own queries
CREATE POLICY "Users can insert own queries" ON public.queries
  FOR INSERT WITH CHECK (auth.uid() = user_id);
```

```bash
# Apply migration
supabase db push
```

#### 1.3 Environment Variables Setup
Create `.env.local` files:

**Frontend** (`frontend/.env.local`):
```bash
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
```

**Scraper** (`scraper/.env`):
```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-role-key  # DO NOT expose this!
INFERENCE_API_URL=http://localhost:7860     # Local dev, update later
```

---

### Day 2: Frontend Skeleton (Next.js)

**Objective**: Create basic UI with mock data

#### 2.1 Initialize Next.js App
```bash
cd Miles
npx create-next-app@latest frontend --typescript --tailwind --app --no-src-dir
cd frontend
npm install @supabase/supabase-js @supabase/auth-helpers-nextjs
```

#### 2.2 Core File Structure

**`frontend/lib/supabase.ts`** (Supabase client):
```typescript
import { createClient } from '@supabase/supabase-js'

export const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
)

export type Candidate = {
  id: string
  video_id: string
  title: string
  thumbnail_url: string
  miles_score: number
  predicted_velocity: number
  view_count: number
  like_count: number
}

export type Query = {
  id: string
  topic: string
  vibe: string | null
  recommendations: any
  created_at: string
}
```

**`frontend/app/page.tsx`** (Homepage):
```typescript
'use client'

import { useState } from 'react'
import { supabase } from '@/lib/supabase'

export default function Home() {
  const [topic, setTopic] = useState('')
  const [vibe, setVibe] = useState('')
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<any>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)

    // TODO: Call Supabase Edge Function for recommendations
    // For now, mock the response
    setTimeout(() => {
      setResults({
        recommendations: [
          { title: 'Mock Idea 1', reasoning: 'Based on trending patterns...' },
          { title: 'Mock Idea 2', reasoning: 'High engagement in this niche...' },
          { title: 'Mock Idea 3', reasoning: 'Similar to top performers...' }
        ],
        topCandidates: []
      })
      setLoading(false)
    }, 1500)
  }

  return (
    <main className="min-h-screen p-8 max-w-4xl mx-auto">
      <h1 className="text-4xl font-bold mb-8">Miles: Viral Video Recommender</h1>

      <form onSubmit={handleSubmit} className="space-y-4 mb-8">
        <div>
          <label className="block text-sm font-medium mb-2">
            What topic are you creating content about?
          </label>
          <input
            type="text"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            placeholder="e.g., tech reviews, cooking tutorials, fitness tips"
            className="w-full p-3 border rounded-lg"
            required
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-2">
            What vibe are you going for? (optional)
          </label>
          <select
            value={vibe}
            onChange={(e) => setVibe(e.target.value)}
            className="w-full p-3 border rounded-lg"
          >
            <option value="">Any</option>
            <option value="funny">Funny/Comedic</option>
            <option value="educational">Educational</option>
            <option value="inspirational">Inspirational</option>
            <option value="dramatic">Dramatic</option>
          </select>
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full bg-blue-600 text-white p-3 rounded-lg hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? 'Analyzing trends...' : 'Get Recommendations'}
        </button>
      </form>

      {results && (
        <div className="space-y-6">
          <h2 className="text-2xl font-bold">Recommended Video Ideas</h2>
          {results.recommendations.map((rec: any, idx: number) => (
            <div key={idx} className="p-4 border rounded-lg bg-gray-50">
              <h3 className="font-semibold text-lg">{rec.title}</h3>
              <p className="text-gray-600 mt-2">{rec.reasoning}</p>
            </div>
          ))}
        </div>
      )}
    </main>
  )
}
```

#### 2.3 Test Locally
```bash
cd frontend
npm run dev
# Visit http://localhost:3000
```

---

### Day 3: Mock Inference Endpoint

**Objective**: Create a placeholder API that mimics the ML service

**`frontend/app/api/predict/route.ts`** (Mock endpoint):
```typescript
import { NextResponse } from 'next/server'

export async function POST(request: Request) {
  const { title, thumbnail_url } = await request.json()

  // Simulate processing delay
  await new Promise(resolve => setTimeout(resolve, 500))

  // Return mock prediction
  return NextResponse.json({
    viral_score: Math.random() * 0.4 + 0.6,  // Random score 0.6-1.0
    predicted_velocity: Math.floor(Math.random() * 2000) + 500,
    processing_time_ms: 120
  })
}
```

**Testing**:
```bash
curl -X POST http://localhost:3000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "Test video", "thumbnail_url": "https://example.com/thumb.jpg"}'
```

---

## Phase 2: Data Pipeline (Days 4-5)

### Day 4: Build Scraper Service

**Objective**: Extract YouTube Shorts metadata and store in Supabase

#### 4.1 Scraper Implementation

**`scraper/requirements.txt`**:
```
yt-dlp==2023.12.30
python-dotenv==1.0.0
supabase==2.3.0
requests==2.31.0
```

**`scraper/scraper.py`**:
```python
import os
import json
import yt_dlp
from datetime import datetime
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

# Supabase client (with service role key for write access)
supabase = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_SERVICE_KEY')
)

SEARCH_QUERIES = [
    'tech reviews shorts',
    'cooking hacks shorts',
    'fitness tips shorts',
    'productivity shorts',
    'travel destinations shorts'
]

def scrape_youtube_shorts(query: str, max_results: int = 20):
    """
    Scrape YouTube Shorts metadata without downloading videos.
    Returns list of video metadata dictionaries.
    """
    ydl_opts = {
        'quiet': True,
        'extract_flat': False,  # Get full metadata
        'skip_download': True,   # Don't download video
        'format': 'bestaudio',   # Dummy format (we're not downloading)
        'noplaylist': True,
        'ignoreerrors': True,
    }

    videos = []
    search_url = f"ytsearch{max_results}:{query}"

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(search_url, download=False)

            if 'entries' in info:
                for entry in info['entries']:
                    if entry and entry.get('duration', 0) <= 60:  # Shorts are <60s
                        videos.append({
                            'video_id': entry.get('id'),
                            'title': entry.get('title'),
                            'description': entry.get('description', ''),
                            'thumbnail_url': entry.get('thumbnail'),
                            'channel_name': entry.get('uploader'),
                            'upload_date': parse_upload_date(entry.get('upload_date')),
                            'view_count': entry.get('view_count', 0),
                            'like_count': entry.get('like_count', 0),
                            'comment_count': entry.get('comment_count', 0),
                            'duration_seconds': entry.get('duration'),
                            'tags': entry.get('tags', []),
                            'platform': 'youtube'
                        })
    except Exception as e:
        print(f"Error scraping {query}: {e}")

    return videos

def parse_upload_date(date_str: str):
    """Convert YYYYMMDD to ISO timestamp"""
    if not date_str:
        return None
    try:
        dt = datetime.strptime(date_str, '%Y%m%d')
        return dt.isoformat()
    except:
        return None

def upload_to_supabase(videos: list):
    """
    Upsert videos to Supabase candidates table.
    Uses video_id as unique constraint to avoid duplicates.
    """
    for video in videos:
        try:
            # Upsert (insert or update if exists)
            result = supabase.table('candidates').upsert(
                video,
                on_conflict='video_id'
            ).execute()
            print(f"✓ Uploaded: {video['title'][:50]}...")
        except Exception as e:
            print(f"✗ Failed to upload {video['video_id']}: {e}")

def main():
    print("Starting YouTube Shorts scraper...")
    print(f"Timestamp: {datetime.now().isoformat()}\n")

    all_videos = []

    for query in SEARCH_QUERIES:
        print(f"Scraping: {query}")
        videos = scrape_youtube_shorts(query, max_results=20)
        print(f"  Found {len(videos)} videos\n")
        all_videos.extend(videos)

    print(f"\nTotal videos scraped: {len(all_videos)}")
    print("Uploading to Supabase...")

    upload_to_supabase(all_videos)

    print("\n✓ Scraping complete!")

if __name__ == '__main__':
    main()
```

#### 4.2 Test Locally
```bash
cd scraper
pip install -r requirements.txt
python scraper.py
```

**Expected output**:
```
Starting YouTube Shorts scraper...
Scraping: tech reviews shorts
  Found 18 videos
Scraping: cooking hacks shorts
  Found 20 videos
...
Total videos scraped: 95
Uploading to Supabase...
✓ Uploaded: Amazing iPhone 15 Feature You Didn't Know...
✓ Uploaded: 5-Minute Pasta Hack That Will Change Your Life...
...
✓ Scraping complete!
```

---

### Day 5: Automate Scraping with GitHub Actions

**Objective**: Run scraper every 6 hours automatically

**`.github/workflows/scraper-cron.yml`**:
```yaml
name: YouTube Scraper CRON

on:
  schedule:
    # Run every 6 hours (0, 6, 12, 18 UTC)
    - cron: '0 */6 * * *'
  workflow_dispatch:  # Allow manual trigger

jobs:
  scrape:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          cd scraper
          pip install -r requirements.txt

      - name: Run scraper
        env:
          SUPABASE_URL: ${{ secrets.SUPABASE_URL }}
          SUPABASE_SERVICE_KEY: ${{ secrets.SUPABASE_SERVICE_KEY }}
          INFERENCE_API_URL: ${{ secrets.INFERENCE_API_URL }}
        run: |
          cd scraper
          python scraper.py

      - name: Trigger inference (if scraper found new videos)
        env:
          SUPABASE_URL: ${{ secrets.SUPABASE_URL }}
          SUPABASE_SERVICE_KEY: ${{ secrets.SUPABASE_SERVICE_KEY }}
          INFERENCE_API_URL: ${{ secrets.INFERENCE_API_URL }}
        run: |
          cd scraper
          python trigger_inference.py  # We'll create this next
```

**Setup GitHub Secrets**:
1. Go to your repo → Settings → Secrets and variables → Actions
2. Add:
   - `SUPABASE_URL`
   - `SUPABASE_SERVICE_KEY`
   - `INFERENCE_API_URL` (will update in Phase 3)

**Manual trigger test**:
```bash
# Go to GitHub → Actions tab → Select "YouTube Scraper CRON" → Run workflow
```

---

## Phase 3: ML Inference Service (Days 6-8)

### Day 6: Export Model to ONNX

**Objective**: Convert your PyTorch model to ONNX for faster CPU inference

#### 6.1 Export Script

**`scripts/export_model_to_onnx.py`**:
```python
import torch
import torch.onnx
from pathlib import Path
import sys

# Add your src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.multimodal_fusion import MultimodalFusionModel
from PIL import Image
import torchvision.transforms as transforms

def export_to_onnx(
    checkpoint_path: str,
    output_path: str = 'inference-service/models/miles_model.onnx',
    opset_version: int = 14
):
    """
    Export trained Miles model to ONNX format.
    """
    print(f"Loading checkpoint from {checkpoint_path}...")

    # Load model (adjust based on your actual model class)
    model = MultimodalFusionModel(
        bert_model_name='bert-base-uncased',
        resnet_model_name='resnet50',
        num_scalar_features=10,  # Adjust to your config
        fusion_dim=256,
        dropout=0.3
    )

    # Load trained weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create dummy inputs matching your model's input signature
    dummy_input_ids = torch.randint(0, 30522, (1, 128))  # BERT vocab size
    dummy_attention_mask = torch.ones(1, 128)
    dummy_image = torch.randn(1, 3, 224, 224)  # ResNet input
    dummy_scalars = torch.randn(1, 10)

    # Export
    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask, dummy_image, dummy_scalars),
        output_path,
        export_params=True,
        opset_version=opset_version,
        input_names=['input_ids', 'attention_mask', 'image', 'scalars'],
        output_names=['viral_logits', 'velocity'],
        dynamic_axes={
            'input_ids': {0: 'batch_size'},
            'attention_mask': {0: 'batch_size'},
            'image': {0: 'batch_size'},
            'scalars': {0: 'batch_size'},
            'viral_logits': {0: 'batch_size'},
            'velocity': {0: 'batch_size'}
        }
    )

    print(f"✓ Model exported to {output_path}")
    print(f"  File size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")

if __name__ == '__main__':
    # Replace with your actual checkpoint path
    export_to_onnx(
        checkpoint_path='checkpoints/best_model.pt',
        output_path='inference-service/models/miles_model.onnx'
    )
```

**Run**:
```bash
python scripts/export_model_to_onnx.py
```

#### 6.2 Optimize with Quantization (Optional but Recommended)
```python
# Add to export script after ONNX export
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_model(onnx_path: str):
    """Apply int8 dynamic quantization to reduce model size."""
    quantized_path = onnx_path.replace('.onnx', '_quantized.onnx')

    quantize_dynamic(
        onnx_path,
        quantized_path,
        weight_type=QuantType.QInt8
    )

    print(f"✓ Quantized model saved to {quantized_path}")
    print(f"  Size reduction: {Path(onnx_path).stat().st_size / Path(quantized_path).stat().st_size:.2f}x")
```

---

### Day 7: Build FastAPI Inference Service

**Objective**: Create HTTP API for model inference

#### 7.1 Service Structure

**`inference-service/requirements.txt`**:
```
fastapi==0.109.0
uvicorn[standard]==0.27.0
onnxruntime==1.16.3
transformers==4.36.2
torch==2.1.2  # For tokenizer only
torchvision==0.16.2
pillow==10.2.0
requests==2.31.0
python-multipart==0.0.6
```

**`inference-service/app.py`**:
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
from transformers import BertTokenizer
from PIL import Image
import requests
from io import BytesIO
import time
from typing import Optional
import torchvision.transforms as transforms

app = FastAPI(title="Miles Inference API")

# Load ONNX model
print("Loading ONNX model...")
session = ort.InferenceSession(
    "models/miles_model.onnx",
    providers=['CPUExecutionProvider']  # Free tier = CPU only
)
print("✓ Model loaded")

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Image preprocessing (same as training)
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class PredictionRequest(BaseModel):
    title: str
    description: Optional[str] = ""
    thumbnail_url: str
    view_count: int = 0
    like_count: int = 0
    comment_count: int = 0
    duration_seconds: int = 30

class PredictionResponse(BaseModel):
    viral_score: float
    predicted_velocity: float
    confidence: float
    processing_time_ms: int

def preprocess_text(title: str, description: str) -> dict:
    """Tokenize text for BERT"""
    text = f"{title} [SEP] {description}"
    encoded = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='np'
    )
    return {
        'input_ids': encoded['input_ids'].astype(np.int64),
        'attention_mask': encoded['attention_mask'].astype(np.int64)
    }

def preprocess_image(url: str) -> np.ndarray:
    """Download and preprocess thumbnail"""
    try:
        response = requests.get(url, timeout=5)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        tensor = image_transform(image)
        return tensor.unsqueeze(0).numpy()
    except Exception as e:
        print(f"Image preprocessing error: {e}")
        # Return blank image as fallback
        return np.zeros((1, 3, 224, 224), dtype=np.float32)

def compute_scalars(req: PredictionRequest) -> np.ndarray:
    """Extract scalar features (same as training)"""
    # Normalize features (use same stats as training)
    features = np.array([
        req.view_count / 1_000_000,  # Scale to millions
        req.like_count / 10_000,
        req.comment_count / 1_000,
        req.duration_seconds / 60,
        req.like_count / max(req.view_count, 1),  # Like rate
        req.comment_count / max(req.view_count, 1),  # Comment rate
        len(req.title.split()),  # Title word count
        len(req.description.split()),  # Description word count
        1.0 if req.duration_seconds < 30 else 0.0,  # Is short
        1.0 if req.view_count > 100_000 else 0.0  # Is popular
    ], dtype=np.float32).reshape(1, -1)

    return features

@app.get("/")
def health_check():
    return {"status": "healthy", "model": "Miles v1.0"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest):
    """
    Run inference on a video candidate.
    Returns viral score [0-1] and predicted velocity.
    """
    start_time = time.time()

    try:
        # Preprocess inputs
        text_inputs = preprocess_text(req.title, req.description or "")
        image_input = preprocess_image(req.thumbnail_url)
        scalar_input = compute_scalars(req)

        # Run inference
        outputs = session.run(
            None,  # Return all outputs
            {
                'input_ids': text_inputs['input_ids'],
                'attention_mask': text_inputs['attention_mask'],
                'image': image_input,
                'scalars': scalar_input
            }
        )

        # Parse outputs
        viral_logits = outputs[0][0]  # Shape: (2,) for binary classification
        velocity_pred = outputs[1][0][0]  # Shape: (1,)

        # Convert logits to probability
        viral_prob = 1 / (1 + np.exp(-viral_logits[1]))  # Sigmoid on positive class

        processing_time = int((time.time() - start_time) * 1000)

        return PredictionResponse(
            viral_score=float(viral_prob),
            predicted_velocity=float(velocity_pred),
            confidence=float(abs(viral_prob - 0.5) * 2),  # 0-1 scale
            processing_time_ms=processing_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=7860)
```

#### 7.2 Local Testing
```bash
cd inference-service
pip install -r requirements.txt

# Create models directory and add your .onnx file
mkdir -p models
# (Copy miles_model.onnx here)

# Run server
python app.py
```

**Test request**:
```bash
curl -X POST http://localhost:7860/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "10 Python Tips Every Developer Should Know",
    "description": "Quick tutorial on Python best practices",
    "thumbnail_url": "https://i.ytimg.com/vi/dQw4w9WgXcQ/maxresdefault.jpg",
    "view_count": 150000,
    "like_count": 5000,
    "comment_count": 300,
    "duration_seconds": 45
  }'
```

---

### Day 8: Deploy to HuggingFace Spaces

**Objective**: Host inference service on free infrastructure

#### 8.1 Create Dockerfile

**`inference-service/Dockerfile`**:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:7860/')"

# Run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

#### 8.2 Deploy to HuggingFace

1. **Create HuggingFace Space**:
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Name: `miles-inference`
   - SDK: Docker
   - Visibility: Public (free tier requirement)

2. **Push Code**:
```bash
cd inference-service
git init
git remote add hf https://huggingface.co/spaces/<your-username>/miles-inference
git add .
git commit -m "Initial deployment"
git push hf main
```

3. **Wait for Build** (~5-10 mins)

4. **Test Deployment**:
```bash
# Your Space URL will be: https://<your-username>-miles-inference.hf.space
curl https://<your-username>-miles-inference.hf.space/
```

#### 8.3 Update Scraper to Call Inference

**`scraper/trigger_inference.py`**:
```python
import os
import requests
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

supabase = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_SERVICE_KEY')
)

INFERENCE_URL = os.getenv('INFERENCE_API_URL')

def process_unanalyzed_videos():
    """
    Find videos without Miles scores and run inference.
    """
    # Query videos that haven't been analyzed yet
    response = supabase.table('candidates')\
        .select('*')\
        .is_('analyzed_at', 'null')\
        .limit(50)\
        .execute()

    videos = response.data
    print(f"Found {len(videos)} videos to analyze")

    for video in videos:
        try:
            # Call inference API
            result = requests.post(
                f"{INFERENCE_URL}/predict",
                json={
                    'title': video['title'],
                    'description': video.get('description', ''),
                    'thumbnail_url': video['thumbnail_url'],
                    'view_count': video.get('view_count', 0),
                    'like_count': video.get('like_count', 0),
                    'comment_count': video.get('comment_count', 0),
                    'duration_seconds': video.get('duration_seconds', 30)
                },
                timeout=30
            ).json()

            # Update database with predictions
            supabase.table('candidates').update({
                'miles_score': result['viral_score'],
                'predicted_velocity': result['predicted_velocity'],
                'analyzed_at': 'now()'
            }).eq('id', video['id']).execute()

            print(f"✓ Analyzed: {video['title'][:50]} (score: {result['viral_score']:.3f})")

        except Exception as e:
            print(f"✗ Failed {video['id']}: {e}")

if __name__ == '__main__':
    process_unanalyzed_videos()
```

**Update GitHub Action** (add step after scraper):
```yaml
- name: Run inference on new videos
  run: |
    cd scraper
    python trigger_inference.py
```

---

## Phase 4: Frontend & Intelligence Layer (Days 9-10)

### Day 9: Build Supabase Edge Function for Recommendations

**Objective**: Create LLM-powered recommendation generator

#### 9.1 Initialize Edge Functions
```bash
cd Miles
supabase functions new generate-recommendations
```

**`supabase/functions/generate-recommendations/index.ts`**:
```typescript
import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

serve(async (req) => {
  // Handle CORS
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    const { topic, vibe, userId } = await req.json()

    // Initialize Supabase client
    const supabaseClient = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ?? ''
    )

    // 1. Query top-performing candidates matching topic
    const { data: topCandidates, error: dbError } = await supabaseClient
      .from('candidates')
      .select('*')
      .gte('miles_score', 0.75)  // High virality threshold
      .order('miles_score', { ascending: false })
      .limit(10)

    if (dbError) throw dbError

    // 2. Extract patterns from top videos
    const patterns = {
      avgTitleLength: topCandidates.reduce((sum, v) => sum + v.title.length, 0) / topCandidates.length,
      commonWords: extractCommonWords(topCandidates.map(v => v.title)),
      avgDuration: topCandidates.reduce((sum, v) => sum + (v.duration_seconds || 30), 0) / topCandidates.length,
      topTitles: topCandidates.slice(0, 5).map(v => v.title)
    }

    // 3. Generate recommendations using OpenAI
    const prompt = buildPrompt(topic, vibe, patterns)

    const openaiResponse = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${Deno.env.get('OPENAI_API_KEY')}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: 'gpt-4o-mini',
        messages: [
          { role: 'system', content: 'You are a viral content strategist analyzing short-form video trends.' },
          { role: 'user', content: prompt }
        ],
        temperature: 0.8,
        max_tokens: 500
      })
    })

    const aiResult = await openaiResponse.json()
    const recommendations = parseRecommendations(aiResult.choices[0].message.content)

    // 4. Save query to database
    await supabaseClient.from('queries').insert({
      user_id: userId,
      topic,
      vibe,
      recommendations,
      matched_candidates: topCandidates.slice(0, 5).map(v => v.id)
    })

    // 5. Return response
    return new Response(
      JSON.stringify({
        success: true,
        recommendations,
        topCandidates: topCandidates.slice(0, 3).map(v => ({
          title: v.title,
          thumbnail_url: v.thumbnail_url,
          miles_score: v.miles_score,
          view_count: v.view_count
        }))
      }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    )

  } catch (error) {
    return new Response(
      JSON.stringify({ error: error.message }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    )
  }
})

function buildPrompt(topic: string, vibe: string, patterns: any): string {
  return `
Based on analysis of trending short-form videos, generate 3 specific video ideas for a creator in the "${topic}" niche.

Context:
- Desired vibe: ${vibe || 'any'}
- Top-performing video titles in this niche:
${patterns.topTitles.map((t: string, i: number) => `  ${i + 1}. ${t}`).join('\n')}

- Common words in viral titles: ${patterns.commonWords.slice(0, 10).join(', ')}
- Average title length: ${Math.round(patterns.avgTitleLength)} characters
- Average duration: ${Math.round(patterns.avgDuration)} seconds

Requirements:
1. Each idea should be specific and actionable (not generic advice)
2. Include a catchy title (similar length to top performers)
3. Briefly explain WHY this concept has viral potential (1-2 sentences)
4. Format as JSON array: [{"title": "...", "reasoning": "..."}]

Generate ideas that leverage proven patterns while being fresh and unique:
`.trim()
}

function extractCommonWords(titles: string[]): string[] {
  const wordCounts: Record<string, number> = {}
  const stopWords = new Set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'])

  titles.forEach(title => {
    title.toLowerCase().split(/\W+/).forEach(word => {
      if (word.length > 3 && !stopWords.has(word)) {
        wordCounts[word] = (wordCounts[word] || 0) + 1
      }
    })
  })

  return Object.entries(wordCounts)
    .sort((a, b) => b[1] - a[1])
    .map(([word]) => word)
}

function parseRecommendations(content: string): any {
  try {
    // Try to extract JSON from markdown code blocks
    const jsonMatch = content.match(/```json\s*([\s\S]*?)\s*```/) ||
                     content.match(/\[[\s\S]*\]/)

    if (jsonMatch) {
      return JSON.parse(jsonMatch[1] || jsonMatch[0])
    }

    // Fallback: return raw content
    return [{ title: 'Parse Error', reasoning: content }]
  } catch {
    return [{ title: 'Parse Error', reasoning: content }]
  }
}
```

#### 9.2 Deploy Edge Function
```bash
# Set OpenAI API key as secret
supabase secrets set OPENAI_API_KEY=sk-your-key-here

# Deploy function
supabase functions deploy generate-recommendations
```

---

### Day 10: Complete Frontend Integration

**Objective**: Wire frontend to real backend services

#### 10.1 Update Homepage with Real API Calls

**`frontend/app/page.tsx`** (updated):
```typescript
'use client'

import { useState } from 'react'
import { supabase } from '@/lib/supabase'

type Recommendation = {
  title: string
  reasoning: string
}

type TopCandidate = {
  title: string
  thumbnail_url: string
  miles_score: number
  view_count: number
}

export default function Home() {
  const [topic, setTopic] = useState('')
  const [vibe, setVibe] = useState('')
  const [loading, setLoading] = useState(false)
  const [recommendations, setRecommendations] = useState<Recommendation[]>([])
  const [topCandidates, setTopCandidates] = useState<TopCandidate[]>([])
  const [error, setError] = useState('')

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError('')

    try {
      // Call Supabase Edge Function
      const { data, error: funcError } = await supabase.functions.invoke(
        'generate-recommendations',
        {
          body: {
            topic,
            vibe,
            userId: null  // TODO: Add auth
          }
        }
      )

      if (funcError) throw funcError

      setRecommendations(data.recommendations)
      setTopCandidates(data.topCandidates)
    } catch (err: any) {
      setError(err.message || 'Failed to generate recommendations')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="min-h-screen p-8 max-w-6xl mx-auto bg-gradient-to-br from-blue-50 to-purple-50">
      <header className="mb-12 text-center">
        <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
          Miles AI
        </h1>
        <p className="text-gray-600 text-lg">
          AI-Powered Viral Video Recommendations
        </p>
      </header>

      <form onSubmit={handleSubmit} className="max-w-2xl mx-auto space-y-6 mb-12 bg-white p-8 rounded-2xl shadow-lg">
        <div>
          <label className="block text-sm font-semibold mb-3 text-gray-700">
            What's your content topic?
          </label>
          <input
            type="text"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            placeholder="e.g., tech reviews, cooking, fitness, travel"
            className="w-full p-4 border-2 border-gray-200 rounded-xl focus:border-blue-500 focus:outline-none transition"
            required
          />
        </div>

        <div>
          <label className="block text-sm font-semibold mb-3 text-gray-700">
            Choose your vibe
          </label>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {['funny', 'educational', 'inspirational', 'dramatic'].map(v => (
              <button
                key={v}
                type="button"
                onClick={() => setVibe(v === vibe ? '' : v)}
                className={`p-3 rounded-xl border-2 transition capitalize ${
                  vibe === v
                    ? 'border-blue-500 bg-blue-50 text-blue-700 font-semibold'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                {v}
              </button>
            ))}
          </div>
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white p-4 rounded-xl font-semibold hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition shadow-lg"
        >
          {loading ? (
            <span className="flex items-center justify-center">
              <svg className="animate-spin h-5 w-5 mr-3" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
              </svg>
              Analyzing trends...
            </span>
          ) : (
            'Get Viral Ideas'
          )}
        </button>

        {error && (
          <div className="p-4 bg-red-50 border-2 border-red-200 rounded-xl text-red-700">
            {error}
          </div>
        )}
      </form>

      {recommendations.length > 0 && (
        <div className="max-w-4xl mx-auto space-y-8">
          <section>
            <h2 className="text-3xl font-bold mb-6 text-gray-800">
              Your Viral Video Ideas
            </h2>
            <div className="space-y-4">
              {recommendations.map((rec, idx) => (
                <div key={idx} className="p-6 bg-white rounded-2xl shadow-md hover:shadow-xl transition">
                  <div className="flex items-start gap-4">
                    <div className="flex-shrink-0 w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-500 rounded-xl flex items-center justify-center text-white font-bold text-xl">
                      {idx + 1}
                    </div>
                    <div className="flex-1">
                      <h3 className="font-bold text-xl mb-2 text-gray-800">
                        {rec.title}
                      </h3>
                      <p className="text-gray-600 leading-relaxed">
                        {rec.reasoning}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {topCandidates.length > 0 && (
            <section>
              <h2 className="text-2xl font-bold mb-6 text-gray-800">
                Inspired by these trending videos
              </h2>
              <div className="grid md:grid-cols-3 gap-6">
                {topCandidates.map((video, idx) => (
                  <div key={idx} className="bg-white rounded-xl shadow-md overflow-hidden hover:shadow-xl transition">
                    <img
                      src={video.thumbnail_url}
                      alt={video.title}
                      className="w-full h-48 object-cover"
                    />
                    <div className="p-4">
                      <p className="font-semibold text-sm mb-2 line-clamp-2">
                        {video.title}
                      </p>
                      <div className="flex items-center justify-between text-xs text-gray-500">
                        <span>{(video.view_count / 1000).toFixed(0)}K views</span>
                        <span className="px-2 py-1 bg-green-100 text-green-700 rounded-full font-semibold">
                          {(video.miles_score * 100).toFixed(0)}% viral
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}
        </div>
      )}
    </main>
  )
}
```

#### 10.2 Deploy Frontend to Vercel
```bash
cd frontend
npm run build  # Verify build works

# Install Vercel CLI
npm i -g vercel

# Deploy
vercel --prod
```

**Set environment variables in Vercel dashboard**:
- `NEXT_PUBLIC_SUPABASE_URL`
- `NEXT_PUBLIC_SUPABASE_ANON_KEY`

---

## Phase 5: Integration & Testing (Days 11-12)

### Day 11: End-to-End Testing

**Objective**: Verify full pipeline works

#### 11.1 Test Checklist

```markdown
## Data Pipeline
- [ ] Scraper runs successfully (scraper/scraper.py)
- [ ] Videos are stored in Supabase (check candidates table)
- [ ] No duplicate video_ids
- [ ] Thumbnails are valid URLs

## Inference Service
- [ ] Health check returns 200 (curl https://your-space.hf.space/)
- [ ] /predict endpoint processes requests in <2s on average
- [ ] Scores are in valid range [0, 1]
- [ ] Cold start warning displays in UI

## Backend (Supabase)
- [ ] Edge function deploys successfully
- [ ] OpenAI API key is set correctly
- [ ] Recommendations are generated (3 ideas per query)
- [ ] Queries are logged in database
- [ ] RLS policies prevent unauthorized access

## Frontend
- [ ] Form validation works
- [ ] Loading states display correctly
- [ ] Error messages are user-friendly
- [ ] Results display with proper formatting
- [ ] Responsive on mobile/tablet

## Integration
- [ ] GitHub Action runs on schedule
- [ ] New videos trigger inference automatically
- [ ] Frontend → Edge Function → LLM flow works
- [ ] All environment variables are set
```

#### 11.2 Load Testing (Optional)
```python
# scripts/load_test.py
import requests
import time
from concurrent.futures import ThreadPoolExecutor

INFERENCE_URL = "https://your-space.hf.space/predict"

def test_request():
    start = time.time()
    response = requests.post(INFERENCE_URL, json={
        "title": "Load test video",
        "thumbnail_url": "https://via.placeholder.com/224",
        "view_count": 10000,
        "like_count": 500,
        "comment_count": 50,
        "duration_seconds": 30
    })
    elapsed = time.time() - start
    return response.status_code, elapsed

# Simulate 10 concurrent users
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(test_request) for _ in range(50)]
    results = [f.result() for f in futures]

avg_time = sum(r[1] for r in results) / len(results)
success_rate = sum(1 for r in results if r[0] == 200) / len(results)

print(f"Average response time: {avg_time:.2f}s")
print(f"Success rate: {success_rate * 100:.1f}%")
```

---

### Day 12: Documentation & Polish

**Objective**: Finalize MVP for demo

#### 12.1 Create User Guide

**`USER_GUIDE.md`**:
```markdown
# Miles MVP: User Guide

## What is Miles?
Miles is an AI-powered recommendation system that analyzes trending short-form videos and suggests viral content ideas for creators.

## How to Use

### 1. Enter Your Topic
Specify the niche you create content in (e.g., "tech reviews", "cooking tutorials").

### 2. Choose a Vibe (Optional)
Select the tone for your videos: funny, educational, inspirational, or dramatic.

### 3. Get Recommendations
Miles will analyze 100+ trending videos and generate 3 personalized ideas with:
- Catchy titles optimized for virality
- Reasoning explaining why each idea has potential
- Examples of similar trending videos

## Behind the Scenes
- **Data**: Real YouTube Shorts scraped every 6 hours
- **AI Model**: Multimodal BERT + ResNet-50 fusion model (75% accuracy)
- **Intelligence**: GPT-4o-mini generates ideas based on proven patterns

## Limitations
- First request may take 20-30s (cold start)
- Recommendations are based on recent trends (may not reflect niche-specific patterns)
- Free tier limits: 50 requests/day

## Support
Issues? Email: your-email@example.com
```

#### 12.2 Performance Optimizations

**Add caching to Edge Function**:
```typescript
// Cache recommendations for 1 hour per topic+vibe combo
const cacheKey = `${topic.toLowerCase()}-${vibe}`
const cached = await supabaseClient
  .from('recommendation_cache')
  .select('*')
  .eq('cache_key', cacheKey)
  .gte('created_at', new Date(Date.now() - 3600000).toISOString())  // 1 hour
  .single()

if (cached.data) {
  return new Response(JSON.stringify(cached.data.result), { headers: corsHeaders })
}

// ... generate recommendations ...

// Store in cache
await supabaseClient.from('recommendation_cache').insert({
  cache_key: cacheKey,
  result: { recommendations, topCandidates }
})
```

**Add loading spinner for cold starts**:
```typescript
// frontend/components/ColdStartWarning.tsx
export function ColdStartWarning({ visible }: { visible: boolean }) {
  if (!visible) return null

  return (
    <div className="fixed bottom-4 right-4 bg-yellow-50 border-2 border-yellow-200 rounded-xl p-4 shadow-lg max-w-sm">
      <p className="text-sm text-yellow-800">
        <strong>First request?</strong> Our AI is waking up... this may take 20-30 seconds.
      </p>
    </div>
  )
}
```

---

## Production Considerations

### Security
1. **API Keys**: Never commit to Git (use `.env` files + `.gitignore`)
2. **RLS Policies**: Enforce on all Supabase tables
3. **Rate Limiting**: Add to Edge Functions to prevent abuse
4. **Input Validation**: Sanitize user inputs (prevent injection attacks)

### Monitoring
```typescript
// Add to Edge Function for tracking
await supabaseClient.from('analytics').insert({
  event_type: 'recommendation_generated',
  user_id: userId,
  topic,
  processing_time_ms: Date.now() - startTime,
  success: true
})
```

### Cost Management
| Service | Free Tier Limit | Monitor |
|---------|----------------|---------|
| Vercel | 100GB bandwidth/mo | Check usage dashboard |
| Supabase | 500MB DB, 50k MAU | Enable email alerts |
| HuggingFace | Unlimited requests (CPU) | Check uptime |
| OpenAI | Pay-per-use | Set billing alerts at $10 |
| GitHub Actions | 2000 minutes/mo | Optimize CRON frequency |

### Scaling Path (Future)
1. **Database**: Migrate to Supabase Pro ($25/mo) for 8GB storage
2. **Inference**: Upgrade to GPU instance ($0.60/hr on HF Spaces Pro)
3. **Caching**: Add Redis for sub-second recommendations
4. **CDN**: Use Cloudflare for image caching
5. **Analytics**: Integrate PostHog for user behavior tracking

---

## Risk Mitigation

### Technical Risks
| Risk | Mitigation |
|------|-----------|
| HF Space sleeps | Add "wake-up" ping in CRON job |
| YouTube blocks scraper | Rotate user agents, add delays |
| OpenAI quota exceeded | Implement fallback to HuggingChat API |
| ONNX model incompatibility | Test export on sample data first |

### Product Risks
| Risk | Mitigation |
|------|-----------|
| Poor recommendation quality | A/B test prompts, collect feedback |
| Low engagement | Add gamification (voting, favorites) |
| Data staleness | Increase CRON frequency to hourly |

---

## Success Metrics

**MVP Launch Criteria**:
- [ ] 100+ videos in database
- [ ] Inference latency < 2s (p50)
- [ ] 3 recommendations generated per query
- [ ] Frontend deployed and accessible
- [ ] No critical bugs in 10 test queries

**Post-Launch KPIs** (Week 1):
- 50+ unique users
- 200+ queries processed
- 85%+ recommendation acceptance rate (user feedback)
- <5% error rate

---

## Next Steps After MVP

### Phase 6: User Feedback & Iteration
- Add thumbs up/down on recommendations
- Collect email signups for waitlist
- Analyze which topics get most engagement

### Phase 7: Advanced Features
- User accounts (save favorites, history)
- Weekly trend reports (email newsletter)
- Multi-platform support (TikTok, Instagram Reels)
- API access for power users

### Phase 8: Monetization
- Premium tier: unlimited queries, priority inference
- Affiliate links for recommended tools
- Sponsored content suggestions

---

**End of Implementation Plan**

This plan is designed to be executed incrementally. Each phase builds on the previous one, allowing for course correction if issues arise. Prioritize getting a working end-to-end flow before optimizing individual components.

Questions? Review the [mvp.md](mvp.md) architecture document or check existing ML code in `src/`.
