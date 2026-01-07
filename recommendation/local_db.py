"""Local CSV-based track recommender utilities.

This module provides functions to load a local CSV containing Spotify-like
track metadata (including `valence` and `energy` columns) and to query it
for nearest matches to target audio parameters.
"""
from typing import List, Dict, Optional
import os
import pandas as pd
import numpy as np


def load_local_tracks(csv_path: str) -> pd.DataFrame:
    """Load CSV into a DataFrame and normalize expected columns.

    Expected columns (case-insensitive): title/track_name, artist(s), valence,
    energy, genre (optional), spotify_url (optional), album_art (optional).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Local tracks CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    # Normalize likely column names
    col_map = {}
    lowered = {c.lower(): c for c in df.columns}
    if 'title' in lowered:
        col_map[lowered['title']] = 'title'
    elif 'track_name' in lowered:
        col_map[lowered['track_name']] = 'title'
    if 'artist' in lowered:
        col_map[lowered['artist']] = 'artist'
    elif 'artists' in lowered:
        col_map[lowered['artists']] = 'artist'
    if 'valence' in lowered:
        col_map[lowered['valence']] = 'valence'
    if 'energy' in lowered:
        col_map[lowered['energy']] = 'energy'
    if 'genre' in lowered:
        col_map[lowered['genre']] = 'genre'
    elif 'track_genre' in lowered:
        col_map[lowered['track_genre']] = 'genre'
    if 'spotify_url' in lowered:
        col_map[lowered['spotify_url']] = 'spotify_url'
    elif 'url' in lowered:
        col_map[lowered['url']] = 'spotify_url'

    df = df.rename(columns=col_map)

    # Ensure valence and energy numeric
    for col in ('valence', 'energy'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = np.nan

    # Ensure string columns exist
    for col in ('title', 'artist', 'genre', 'spotify_url'):
        if col not in df.columns:
            df[col] = None

    return df


def recommend_from_local_db(df: pd.DataFrame, stage: Dict, limit: int = 2, preferred_genres: Optional[List[str]] = None) -> List[Dict]:
    """Return up to `limit` tracks closest to target valence/energy from df.

    - `stage` should contain 'target_valence', 'target_energy', and optional 'seed_genres'.
    - If `preferred_genres` provided, they are given priority.
    """
    target_v = float(stage.get('target_valence', 0.5))
    target_e = float(stage.get('target_energy', 0.5))
    seed_genres = stage.get('seed_genres') or []

    genres = list(dict.fromkeys((preferred_genres or []) + seed_genres)) if preferred_genres or seed_genres else []

    df_copy = df.copy()
    if genres:
        # filter rows where genre contains any of the genres (case-insensitive)
        mask = False
        for g in genres:
            mask = mask | df_copy['genre'].astype(str).str.contains(g, case=False, na=False)
        df_filtered = df_copy[mask]
        if df_filtered.empty:
            df_filtered = df_copy
    else:
        df_filtered = df_copy

    # Compute distance in (valence,energy)
    vals = df_filtered[['valence', 'energy']].fillna(0).to_numpy(dtype=float)
    targets = np.array([target_v, target_e], dtype=float)
    if vals.size == 0:
        return []
    dists = np.linalg.norm(vals - targets, axis=1)
    df_filtered = df_filtered.assign(_dist=dists)
    df_filtered = df_filtered.sort_values('_dist')

    out = []
    for _, row in df_filtered.head(limit).iterrows():
        out.append({
            'title': row.get('title'),
            'artist': row.get('artist'),
            'valence': float(row.get('valence')) if row.get('valence') is not None else None,
            'energy': float(row.get('energy')) if row.get('energy') is not None else None,
            'genre': row.get('genre'),
            'spotify_url': row.get('spotify_url'),
        })

    return out
