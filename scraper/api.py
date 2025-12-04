#!/usr/bin/env python3
"""
Miles Scraper API Service
FastAPI service that exposes scraping functionality via HTTP endpoints.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
import uvicorn

from scraper_module import scrape_topic, scrape_multiple_topics

# Initialize FastAPI app
app = FastAPI(
    title="Miles Scraper API",
    description="Dynamic YouTube Shorts scraping service for viral video analysis",
    version="1.0.0"
)

# Add CORS middleware to allow frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class ScrapeRequest(BaseModel):
    """Request model for scraping a single topic"""
    topic: str = Field(..., description="Topic to scrape (e.g., 'tech reviews')", min_length=1)
    max_results: int = Field(20, description="Maximum number of videos to scrape", ge=1, le=50)
    max_duration: int = Field(60, description="Maximum video duration in seconds", ge=1, le=120)

    class Config:
        schema_extra = {
            "example": {
                "topic": "skateboarding tricks",
                "max_results": 20,
                "max_duration": 60
            }
        }


class ScrapeMultipleRequest(BaseModel):
    """Request model for scraping multiple topics"""
    topics: List[str] = Field(..., description="List of topics to scrape", min_length=1, max_length=10)
    max_results_per_topic: int = Field(15, description="Max videos per topic", ge=1, le=50)

    class Config:
        schema_extra = {
            "example": {
                "topics": ["tech reviews", "cooking recipes", "fitness tips"],
                "max_results_per_topic": 15
            }
        }


class VideoMetadata(BaseModel):
    """Video metadata response model"""
    video_id: str
    title: str
    description: Optional[str]
    thumbnail_url: Optional[str]
    channel_name: Optional[str]
    upload_date: Optional[str]
    view_count: int
    like_count: int
    comment_count: int
    duration_seconds: Optional[int]
    tags: List[str]
    platform: str
    search_topic: str


class ScrapeResponse(BaseModel):
    """Response model for scrape operations"""
    success: bool
    topic: str
    videos_found: int
    videos: List[Dict]
    timestamp: str
    processing_time_ms: Optional[int] = None


# API Endpoints
@app.get("/")
def root():
    """Health check and API info"""
    return {
        "status": "healthy",
        "service": "Miles Scraper API",
        "version": "1.0.0",
        "endpoints": {
            "scrape": "POST /scrape - Scrape a single topic",
            "scrape_multiple": "POST /scrape-multiple - Scrape multiple topics",
            "health": "GET /health - Health check"
        }
    }


@app.get("/health")
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "scraper-api",
        "dependencies": {
            "yt_dlp": "available"
        }
    }


@app.post("/scrape", response_model=ScrapeResponse)
async def scrape(request: ScrapeRequest):
    """
    Scrape YouTube Shorts for a specific topic.

    This endpoint performs real-time scraping and may take 10-30 seconds.

    Args:
        request: ScrapeRequest with topic and optional parameters

    Returns:
        ScrapeResponse with scraped video metadata

    Raises:
        HTTPException: If scraping fails
    """
    start_time = datetime.now()

    try:
        print(f"[{start_time.strftime('%H:%M:%S')}] Scraping topic: {request.topic}")

        # Perform scraping
        videos = scrape_topic(
            topic=request.topic,
            max_results=request.max_results,
            max_duration=request.max_duration
        )

        end_time = datetime.now()
        processing_time = int((end_time - start_time).total_seconds() * 1000)

        print(f"  ✓ Found {len(videos)} videos in {processing_time}ms")

        return ScrapeResponse(
            success=True,
            topic=request.topic,
            videos_found=len(videos),
            videos=videos,
            timestamp=end_time.isoformat(),
            processing_time_ms=processing_time
        )

    except Exception as e:
        print(f"  ✗ Scraping failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Scraping failed for topic '{request.topic}': {str(e)}"
        )


@app.post("/scrape-multiple")
async def scrape_multiple(request: ScrapeMultipleRequest):
    """
    Scrape multiple topics in parallel.

    This endpoint scrapes several topics and returns organized results.
    May take 30-60 seconds depending on number of topics.

    Args:
        request: ScrapeMultipleRequest with list of topics

    Returns:
        Dictionary mapping topics to their scraped videos
    """
    start_time = datetime.now()

    try:
        print(f"[{start_time.strftime('%H:%M:%S')}] Scraping {len(request.topics)} topics")

        # Scrape all topics
        results = scrape_multiple_topics(
            topics=request.topics,
            max_results_per_topic=request.max_results_per_topic
        )

        end_time = datetime.now()
        processing_time = int((end_time - start_time).total_seconds() * 1000)

        total_videos = sum(len(videos) for videos in results.values())
        print(f"  ✓ Found {total_videos} total videos in {processing_time}ms")

        return {
            "success": True,
            "topics_scraped": len(request.topics),
            "total_videos": total_videos,
            "results": results,
            "timestamp": end_time.isoformat(),
            "processing_time_ms": processing_time
        }

    except Exception as e:
        print(f"  ✗ Scraping failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Multi-topic scraping failed: {str(e)}"
        )


# Run server
if __name__ == "__main__":
    import os

    # Use PORT from environment (for Railway/Render) or default to 8001 for local
    port = int(os.environ.get("PORT", 8001))

    print("=" * 70)
    print("Starting Miles Scraper API")
    print("=" * 70)
    print()
    print("API will be available at:")
    print(f"  - Port: {port}")
    print(f"  - Docs: http://localhost:{port}/docs (if running locally)")
    print()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
