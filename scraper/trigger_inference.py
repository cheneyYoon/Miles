#!/usr/bin/env python3
"""
Inference Trigger
Processes unanalyzed videos by calling the ML inference API.
"""

import os
import requests
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

supabase: Client = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_SERVICE_KEY')
)

INFERENCE_URL = os.getenv('INFERENCE_API_URL', 'http://localhost:3000/api/predict')


def process_unanalyzed_videos(batch_size: int = 50):
    """
    Find videos without Miles scores and run inference.

    Args:
        batch_size: Number of videos to process in one run
    """
    print("=" * 70)
    print("Miles Inference Trigger")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    # Query videos that haven't been analyzed yet
    print(f"Fetching unanalyzed videos (limit: {batch_size})...")
    response = supabase.table('candidates')\
        .select('*')\
        .is_('analyzed_at', 'null')\
        .limit(batch_size)\
        .execute()

    videos = response.data
    print(f"Found {len(videos)} videos to analyze")
    print()

    if not videos:
        print("✓ All videos are up to date!")
        print("=" * 70)
        return

    success_count = 0
    error_count = 0

    for idx, video in enumerate(videos, 1):
        try:
            print(f"[{idx}/{len(videos)}] {video['title'][:50]}...")

            # Call inference API
            result = requests.post(
                INFERENCE_URL,
                json={
                    'title': video['title'],
                    'description': video.get('description', ''),
                    'thumbnail_url': video.get('thumbnail_url', ''),
                    'view_count': video.get('view_count', 0),
                    'like_count': video.get('like_count', 0),
                    'comment_count': video.get('comment_count', 0),
                    'duration_seconds': video.get('duration_seconds', 30)
                },
                timeout=30
            )

            if result.status_code == 200:
                prediction = result.json()

                # Update database with predictions
                supabase.table('candidates').update({
                    'miles_score': prediction['viral_score'],
                    'predicted_velocity': prediction.get('predicted_velocity', 0),
                    'analyzed_at': datetime.now().isoformat()
                }).eq('id', video['id']).execute()

                success_count += 1
                print(f"  ✓ Score: {prediction['viral_score']:.3f}")
            else:
                error_count += 1
                print(f"  ✗ API error: {result.status_code}")

        except Exception as e:
            error_count += 1
            print(f"  ✗ Failed: {e}")

    print()
    print("-" * 70)
    print("Summary:")
    print(f"  ✓ Successfully analyzed: {success_count}")
    print(f"  ✗ Errors: {error_count}")
    print()
    print("✓ Inference complete!")
    print("=" * 70)


if __name__ == '__main__':
    process_unanalyzed_videos()
