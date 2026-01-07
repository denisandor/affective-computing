"""Example runner for mood-shift playlist generation.

This script demonstrates using `generate_mood_shift_playlist`. If you set
SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables it will
attempt to use Spotify; otherwise it will use the fallback catalog.
"""
import os
from recommendation.recommender import generate_mood_shift_playlist


def _create_sp_client_from_env():
    try:
        from recommendation.spotify_client import SpotifyClient
    except Exception:
        return None

    cid = os.environ.get('SPOTIFY_CLIENT_ID')
    csec = os.environ.get('SPOTIFY_CLIENT_SECRET')
    if not cid or not csec:
        return None
    try:
        return SpotifyClient(client_id=cid, client_secret=csec)
    except Exception:
        return None


if __name__ == '__main__':
    sp = _create_sp_client_from_env()

    # Example: detected emotion from your AI
    emotion = 'angry'

    # Optional: path to a local CSV dataset (offline/static mode)
    local_csv = os.environ.get('LOCAL_TRACKS_PATH')  # e.g., recommendation/data/tracks.csv

    playlist = generate_mood_shift_playlist(emotion, sp_client=sp, local_csv_path=local_csv)

    for track in playlist:
        print(f"[Stage {track.get('stage')}] {track.get('title')} - {track.get('artist')} ({track.get('reason')})")
