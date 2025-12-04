#!/usr/bin/env python3
"""
YouTube Shorts Scraper
Fetches metadata from YouTube Shorts without downloading videos.
"""

import os
import json
import yt_dlp
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Supabase client (with service role key for write access)
supabase: Client = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_SERVICE_KEY')
)

# Search queries for different niches
SEARCH_QUERIES = [
    'tech reviews shorts',
    'cooking hacks shorts',
    'fitness tips shorts',
    'productivity shorts',
    'travel destinations shorts',
    'AI tutorial shorts',
    'life hacks shorts',
]

def scrape_youtube_shorts(query: str, max_results: int = 15) -> list:
    """
    Scrape YouTube Shorts metadata without downloading videos.

    Args:
        query: Search query string
        max_results: Maximum number of results to fetch

    Returns:
        List of video metadata dictionaries
    """
    ydl_opts = {
        'quiet': True,
        'extract_flat': False,  # Get full metadata
        'skip_download': True,   # Don't download video
        'format': 'bestaudio',   # Dummy format (we're not downloading)
        'noplaylist': True,
        'ignoreerrors': True,
        'no_warnings': True,
    }

    videos = []
    search_url = f"ytsearch{max_results}:{query}"

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"  Searching: {query}")
            info = ydl.extract_info(search_url, download=False)

            if 'entries' in info:
                for entry in info['entries']:
                    if entry and entry.get('duration', 0) <= 60:  # Shorts are ≤60s
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
        print(f"  ✗ Error scraping {query}: {e}")

    return videos


def parse_upload_date(date_str: str) -> str:
    """Convert YYYYMMDD to ISO timestamp"""
    if not date_str:
        return None
    try:
        dt = datetime.strptime(date_str, '%Y%m%d')
        return dt.isoformat()
    except:
        return None


def upload_to_supabase(videos: list) -> tuple:
    """
    Upsert videos to Supabase candidates table.
    Uses video_id as unique constraint to avoid duplicates.

    Returns:
        Tuple of (uploaded_count, duplicate_count, error_count)
    """
    uploaded = 0
    duplicates = 0
    errors = 0

    for video in videos:
        try:
            # Check if video already exists
            existing = supabase.table('candidates').select('id').eq('video_id', video['video_id']).execute()

            if existing.data:
                duplicates += 1
                print(f"  ⊘ Duplicate: {video['title'][:50]}...")
                continue

            # Insert new video
            result = supabase.table('candidates').insert(video).execute()
            uploaded += 1
            print(f"  ✓ Uploaded: {video['title'][:50]}...")

        except Exception as e:
            errors += 1
            print(f"  ✗ Failed {video['video_id']}: {e}")

    return (uploaded, duplicates, errors)


def main():
    """Main scraping workflow"""
    print("=" * 70)
    print("YouTube Shorts Scraper")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    all_videos = []
    total_scraped = 0

    for query in SEARCH_QUERIES:
        print(f"📹 Scraping: {query}")
        videos = scrape_youtube_shorts(query, max_results=15)
        total_scraped += len(videos)
        print(f"  Found {len(videos)} shorts")
        all_videos.extend(videos)
        print()

    print("-" * 70)
    print(f"Total videos scraped: {total_scraped}")
    print("Uploading to Supabase...")
    print()

    uploaded, duplicates, errors = upload_to_supabase(all_videos)

    print("-" * 70)
    print("Summary:")
    print(f"  ✓ Uploaded: {uploaded}")
    print(f"  ⊘ Duplicates skipped: {duplicates}")
    print(f"  ✗ Errors: {errors}")
    print()
    print("✓ Scraping complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
