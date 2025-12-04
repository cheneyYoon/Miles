Miles-V1: Engineering Architecture Report

Target Audience: Senior Software Engineers / Academic Reviewers
Date: Dec 03, 2025
Author: [User Name]
Scope: End-to-End Recommendation System on "Free Tier" Infrastructure
1. Executive Summary

This document outlines the architecture for the Miles-V1 extension: a consumer-facing web application that recommends short-form video concepts by predicting their viral potential. The system decouples the resource-intensive ML inference from the user-facing application, utilizing a hybrid "Serverless + Edge" architecture. By leveraging Next.js for the frontend, Supabase for backend-as-a-service, and Hugging Face Spaces for inference, the system remains within free-tier limits while simulating a production-grade microservices environment.
2. High-Level Architecture

The system is composed of three distinct layers connected via REST APIs:

text
graph TD
    User[User (Browser)] -->|Request| FE[Frontend (Next.js on Vercel)]
    FE -->|Query/Auth| DB[Backend (Supabase: Postgres + Edge Functions)]
    
    subgraph "Data & Inference Layer"
        Scraper[Scraper Service (Python + yt-dlp)] -->|Metadata| DB
        Scraper -->|Video Frames| Model[Miles Inference (Hugging Face Space)]
        Model -->|Virality Score| DB
    end

    subgraph "Intelligence Layer"
        DB -->|Top Patterns| LLM[LLM Wrapper (OpenAI API)]
        LLM -->|Suggestions| FE
    end

3. Technical Stack (Free Tier Optimized)

To minimize costs while maximizing performance, we use industry-standard tools with generous free allowances.
Component	Technology	"Free Tier" Justification
Frontend	Next.js (React)	Vercel: Free global CDN, automatic SSL, serverless API routes.
Backend/DB	Supabase	Free Tier: 500MB Postgres DB, Auth (50k MAU), Edge Functions (500k invocations/mo). Replaces AWS RDS/Lambda.
ML Inference	Hugging Face Spaces	Free Tier: persistent hosting of PyTorch models on CPU (2 vCPU, 16GB RAM). Best option for "always-on" demos without paying for GPUs.
Video Scraping	yt-dlp + Python	GitHub Actions / Local: Run scheduled scraping jobs via GitHub Actions (2,000 free minutes/mo) or locally, pushing data to Supabase.
LLM	OpenAI (gpt-4o-mini)	Low Cost: Not free, but gpt-4o-mini costs pennies per 1k requests. Alternative: Use HuggingChat API (unofficial/free) if budget is strictly $0.
4. Component Deep Dive
A. The "Ears": Data Ingestion Service

Goal: Fetch recent trending candidate videos without hitting API quotas.
Tools: yt-dlp (robust, handles throttling better than raw requests).

    Implementation: A standalone Python script (scraper.py).

    Workflow:

        Search: Use yt-dlp to search for "keywords" with filters (--match-filter "original_url!^=" to avoid duplicates).

        Extract: Download JSON metadata (views, like_count, upload_date) and the thumbnail (jpg), not the full video (to save bandwidth).

        Store: Upsert metadata into Supabase candidates table.

        Automation: Run this as a GitHub Action CRON job (runs every 6 hours) to populate the DB with fresh candidates.

python
# Snippet: Efficient Metadata Extraction with yt-dlp
import yt_dlp

ydl_opts = {
    'quiet': True,
    'extract_flat': True, # Don't download video, just metadata
    'force_json': True,
    'search_query': 'ytsearch20:tech trends shorts' # Search query
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info("ytsearch20:tech trends shorts", download=False)
    # Process 'entries' and push to Supabase...

B. The "Brain": Miles Inference Engine

Goal: Run BERT + ResNet-50 fusion model on candidate data.
Constraint: Vercel/Supabase functions cannot load PyTorch (too large).

    Solution: Deploy the model as a Dockerized FastAPI app on Hugging Face Spaces.

    Optimization:

        ONNX Runtime: Convert your PyTorch model to ONNX format. This reduces inference time by ~30% on CPUs (critical for free tier).

        Quantization: Use int8 quantization to shrink model size (approx. 4x smaller), allowing it to fit easily in the 16GB RAM limit.

    API Endpoint: Expose a single route POST /predict that accepts JSON (text + image URL) and returns {"viral_score": 0.85, "predicted_velocity": 1000}.

C. The "Voice": Recommendation Logic

Goal: Translate raw scores into human advice.

    Supabase Edge Function:

        Query candidates table for top-scoring items (Virality > 0.8).

        Aggregate their metadata (e.g., common words in titles, avg length).

        Construct a prompt for the LLM: "Based on these 5 trending video titles [A, B, C...], generate 3 new video concepts for a [User Topic] channel."

        Return JSON to frontend.

5. Development Roadmap (Senior Engineer's View)

Phase 1: The "Skeleton" (Days 1-3)

    Repo Setup: Monorepo with /frontend (Next.js) and /backend (Python scripts).

    DB Schema: Define Supabase tables: users, queries, candidates (id, title, thumbnail_url, metadata, miles_score).

    Mocking: Create a fake POST /predict endpoint that returns random scores. Connect frontend to this mock.

Phase 2: The "Data Pipe" (Days 4-5)

    Scraper: Write yt-dlp script. Test locally.

    Ingest: Push scraped data to Supabase. Ensure no duplicates.

    Action: Configure GitHub Action to run scraper daily.

Phase 3: The "Brain Transplant" (Days 6-8)

    Export: Save trained Miles model to .onnx.

    Deploy: Upload app.py (FastAPI) and model.onnx to Hugging Face Spaces.

    Connect: Update scraper script to call this HF API for every new video it finds.

Phase 4: The "UI Polish" (Days 9-10)

    Frontend: Build form (Topic, Vibe) and Results Cards.

    LLM: Integrate OpenAI/GPT-4o-mini call in a Supabase Edge Function to generate the final text.

6. Best Practices & Gotchas

    Rate Limiting: yt-dlp can get your IP banned. Do NOT run it from Vercel/Supabase directly (their IPs are shared/blocked). Run it from GitHub Actions or your local machine.

    Cold Starts: Hugging Face Spaces (Free) will "sleep" after inactivity. Your first request might take 30s to wake up. Add a loading spinner on the UI saying "Waking up the AI..."

    Security: Never expose your Supabase service_role key in the frontend. Use RLS (Row Level Security) policies to ensure users can only read recommended data, not write to it.
