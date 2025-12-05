#!/usr/bin/env python3
"""
Modular YouTube Shorts Scraper
Can be called as a function by API services.
"""

import yt_dlp
from datetime import datetime
from typing import List, Dict, Optional


def scrape_topic(
    topic: str,
    max_results: int = 20,
    max_duration: int = 60
) -> List[Dict]:
    """
    Scrape YouTube Shorts for a specific topic.

    Args:
        topic: Search query (e.g., "skateboarding tricks", "cooking recipes")
        max_results: Maximum number of videos to return (default: 20)
        max_duration: Maximum video length in seconds (default: 60 for Shorts)

    Returns:
        List of video metadata dictionaries

    Example:
        >>> videos = scrape_topic("tech reviews", max_results=10)
        >>> print(f"Found {len(videos)} videos")
        >>> print(videos[0]['title'])
    """
    ydl_opts = {
        'quiet': True,
        'extract_flat': False,  # Get full metadata
        'skip_download': True,   # Don't download video
        'format': 'bestaudio',   # Dummy format (we're not downloading)
        'noplaylist': True,
        'ignoreerrors': True,
        'no_warnings': True,
        # Anti-bot detection measures
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'http_headers': {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-us,en;q=0.5',
            'Sec-Fetch-Mode': 'navigate',
        },
        'sleep_interval': 1,  # Add 1 second delay between requests
        'max_sleep_interval': 3,
        'extractor_retries': 3,  # Retry failed extractions
    }

    videos = []

    # Append "shorts" to topic for better YouTube Shorts targeting
    search_query = f"{topic} shorts"
    search_url = f"ytsearch{max_results}:{search_query}"

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(search_url, download=False)

            if 'entries' in info:
                for entry in info['entries']:
                    # Only include videos shorter than max_duration (Shorts)
                    if entry and entry.get('duration', 0) <= max_duration:
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
                            'platform': 'youtube',
                            'search_topic': topic  # Track what topic this came from
                        })

    except Exception as e:
        raise Exception(f"Scraping failed for topic '{topic}': {str(e)}")

    return videos


def parse_upload_date(date_str: Optional[str]) -> Optional[str]:
    """
    Convert YYYYMMDD format to ISO timestamp.

    Args:
        date_str: Date string in YYYYMMDD format (e.g., "20231215")

    Returns:
        ISO format timestamp string or None if parsing fails

    Example:
        >>> parse_upload_date("20231215")
        '2023-12-15T00:00:00'
    """
    if not date_str:
        return None
    try:
        dt = datetime.strptime(date_str, '%Y%m%d')
        return dt.isoformat()
    except:
        return None


def scrape_multiple_topics(
    topics: List[str],
    max_results_per_topic: int = 15
) -> Dict[str, List[Dict]]:
    """
    Scrape multiple topics and return results organized by topic.

    Args:
        topics: List of topic strings to scrape
        max_results_per_topic: Max videos per topic

    Returns:
        Dictionary mapping topic -> list of videos

    Example:
        >>> results = scrape_multiple_topics(['tech', 'cooking', 'fitness'])
        >>> print(f"Found {len(results['tech'])} tech videos")
    """
    results = {}

    for topic in topics:
        try:
            videos = scrape_topic(topic, max_results=max_results_per_topic)
            results[topic] = videos
        except Exception as e:
            print(f"Failed to scrape topic '{topic}': {e}")
            results[topic] = []

    return results


if __name__ == '__main__':
    # Test the module
    print("Testing scraper module...")
    test_topic = "tech reviews"

    try:
        videos = scrape_topic(test_topic, max_results=5)
        print(f"✓ Successfully scraped {len(videos)} videos for '{test_topic}'")

        if videos:
            print("\nFirst video:")
            print(f"  Title: {videos[0]['title']}")
            print(f"  Views: {videos[0]['view_count']:,}")
            print(f"  Duration: {videos[0]['duration_seconds']}s")
    except Exception as e:
        print(f"✗ Test failed: {e}")
