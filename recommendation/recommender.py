"""Simple rule-based music recommender based on multimodal sentiment.

Usage:
from recommendation.recommender import aggregate_sentiments, recommend_music

combined = aggregate_sentiments(audio_label='sad', audio_score=0.8,
                               face_label='neutral', face_score=0.6)
recs = recommend_music(combined)
"""
from typing import Dict, List, Optional
import logging

try:
    from recommendation.spotify_client import SpotifyClient
except Exception:  # optional
    SpotifyClient = None

_LOG = logging.getLogger(__name__)


SENTiment_MAP = {
    'positive': 1.0,
    'neutral': 0.0,
    'negative': -1.0,
}


CATALOG = [
    # Uplift/energize
    {'title': 'Happy', 'artist': 'Pharrell Williams', 'mood': 'uplift', 'genre': 'pop'},
    {'title': 'Uptown Funk', 'artist': 'Mark Ronson ft. Bruno Mars', 'mood': 'uplift', 'genre': 'funk'},
    {'title': "Walking On Sunshine", 'artist': 'Katrina & The Waves', 'mood': 'uplift', 'genre': 'pop'},
    # Calm/sustain
    {'title': 'Someone Like You', 'artist': 'Adele', 'mood': 'sustain', 'genre': 'pop'},
    {'title': 'Weightless', 'artist': 'Marconi Union', 'mood': 'calm', 'genre': 'ambient'},
    {'title': 'Skinny Love', 'artist': 'Bon Iver', 'mood': 'sustain', 'genre': 'indie'},
    # Neutral/comfort
    {'title': 'Here Comes The Sun', 'artist': 'The Beatles', 'mood': 'comfort', 'genre': 'rock'},
    {'title': 'Shape of You (Acoustic)', 'artist': 'Ed Sheeran', 'mood': 'comfort', 'genre': 'acoustic'},
]


def aggregate_sentiments(audio_label: str, audio_score: float,
                         face_label: str, face_score: float,
                         audio_weight: float = 0.6,
                         face_weight: float = 0.4) -> Dict:
    """Aggregate two modality labels+confidences into a combined sentiment.

    Returns a dict: {'label': 'positive'|'neutral'|'negative', 'score': float}
    Higher absolute score indicates stronger sentiment.
    """
    a = SENTiment_MAP.get(audio_label.lower(), 0.0) * float(audio_score)
    f = SENTiment_MAP.get(face_label.lower(), 0.0) * float(face_score)
    combined_score = a * audio_weight + f * face_weight

    if combined_score >= 0.3:
        label = 'positive'
    elif combined_score <= -0.3:
        label = 'negative'
    else:
        label = 'neutral'

    return {'label': label, 'score': combined_score}


def _select_by_mood(mood: str, top_k: int = 3) -> List[Dict]:
    picks = [c for c in CATALOG if c['mood'] == mood]
    return picks[:top_k]


def get_music_parameters(detected_emotion: str) -> Dict:
    """Map a detected emotion to target valence/energy and seed genres.

    Returns a dict. For negative emotions returns a dict with 'stages':
    a list of parameter dicts (iso-principle: match -> bridge -> uplift).
    For positive/neutral returns a single-stage dict under 'stages' as well.
    """
    e = detected_emotion.lower()
    if e in ('sad', 'sadness'):
        # Iso-principle: match (melancholic) -> neutral -> uplift
        return {
            'emotion': 'sad',
            'stages': [
                {'target_valence': 0.10, 'target_energy': 0.20, 'seed_genres': ['acoustic', 'piano', 'indie']},
                {'target_valence': 0.25, 'target_energy': 0.35, 'seed_genres': ['acoustic', 'indie']},
                {'target_valence': 0.40, 'target_energy': 0.50, 'seed_genres': ['pop', 'acoustic']},
            ],
        }
    if e in ('angry', 'anger'):
        # Diffuse: start higher energy/valence then move to calm/ambient
        return {
            'emotion': 'angry',
            'stages': [
                {'target_valence': 0.20, 'target_energy': 0.80, 'seed_genres': ['rock']},
                {'target_valence': 0.35, 'target_energy': 0.55, 'seed_genres': ['rock', 'ambient']},
                {'target_valence': 0.50, 'target_energy': 0.30, 'seed_genres': ['ambient', 'classical']},
            ],
        }
    if e in ('fear', 'anxious', 'anxiety'):
        # Calming progression
        return {
            'emotion': 'fear',
            'stages': [
                {'target_valence': 0.40, 'target_energy': 0.30, 'seed_genres': ['classical', 'chill', 'ambient']},
                {'target_valence': 0.50, 'target_energy': 0.20, 'seed_genres': ['classical', 'ambient']},
                {'target_valence': 0.60, 'target_energy': 0.10, 'seed_genres': ['ambient', 'meditation']},
            ],
        }
    if e in ('happy', 'joy', 'excited'):
        # Sustain: keep high valence/energy
        return {
            'emotion': 'happy',
            'stages': [
                {'target_valence': 0.80, 'target_energy': 0.80, 'seed_genres': ['pop', 'dance', 'funk']},
            ],
        }
    # Neutral / default: small boost
    return {
        'emotion': 'neutral',
        'stages': [
            {'target_valence': 0.50, 'target_energy': 0.60, 'seed_genres': ['pop', 'indie']},
        ],
    }


def _recommend_with_spotify(stage: Dict, sp_client: SpotifyClient, limit: int = 2) -> List[Dict]:
    try:
        return sp_client.recommend_tracks(seed_genres=stage.get('seed_genres'),
                                          target_valence=stage.get('target_valence'),
                                          target_energy=stage.get('target_energy'),
                                          limit=limit)
    except Exception as exc:
        _LOG.exception('Spotify recommendation failed: %s', exc)
        return []


def build_dynamic_playlist(detected_emotion: str, sp_client: Optional[object] = None, per_stage_limit: int = 2) -> List[Dict]:
    """Build a dynamic playlist according to strategy:

    - Negative emotions: Iso-principle 3-stage playlist (match -> bridge -> uplift)
    - Positive emotions: Sustain (repeated high-energy tracks)
    - Neutral: small boost

    If `sp_client` (an instance of `SpotifyClient`) is provided the function
    will try to fetch recommendations from Spotify; otherwise it falls back to
    selecting tracks from the local `CATALOG` by mapping stages to moods.
    """
    params = get_music_parameters(detected_emotion)
    stages = params.get('stages', [])
    playlist: List[Dict] = []

    use_spotify = sp_client is not None and SpotifyClient is not None

    for idx, stage in enumerate(stages):
        if use_spotify:
            recs = _recommend_with_spotify(stage, sp_client, limit=per_stage_limit)
            if recs:
                # tag stage index and reason
                for r in recs:
                    r['stage'] = idx + 1
                    r['reason'] = f"stage={idx+1}; emotion={params.get('emotion')}"
                playlist.extend(recs)
                continue

        # Fallback: choose from CATALOG mapping
        # Map stage to a mood heuristic
        mood = 'comfort'
        if params.get('emotion') == 'happy':
            mood = 'uplift'
        elif params.get('emotion') in ('sad', 'angry', 'fear'):
            # first stage -> comfort/calm, last stage -> uplift/sustain depending on idx
            if idx == 0:
                mood = 'comfort'
            elif idx == len(stages) - 1:
                mood = 'uplift'
            else:
                mood = 'calm'

        picks = _select_by_mood(mood, top_k=per_stage_limit)
        for p in picks:
            p_copy = p.copy()
            p_copy['stage'] = idx + 1
            p_copy['reason'] = f"fallback_stage={idx+1}; emotion={params.get('emotion')}"
            playlist.append(p_copy)

    return playlist


def generate_mood_shift_playlist(current_emotion: str, sp_client: Optional[object] = None,
                                 local_csv_path: Optional[str] = None,
                                 preferred_genres: Optional[List[str]] = None) -> List[Dict]:
    """Generate a 3-stage (or single-stage for positive) mood-shift playlist.

    Trajectories follow the Iso-Principle: for negative emotions return three
    mini-batches (match -> bridge -> uplift). For positive/neutral return a
    single stage with a longer list (sustain/boost).

    If `sp_client` is provided it will be used to fetch recommendations from
    Spotify (via `SpotifyClient.recommend_tracks`). Otherwise the function
    falls back to selecting from the local `CATALOG`.
    """
    # Define trajectories (valence 0-1, energy 0-1, genres, count per stage)
    trajectories = {
        "sad": [
            {"target_valence": 0.2, "target_energy": 0.3, "seed_genres": ["acoustic", "piano", "sad"], "count": 2},
            {"target_valence": 0.5, "target_energy": 0.5, "seed_genres": ["indie", "chill"], "count": 2},
            {"target_valence": 0.8, "target_energy": 0.7, "seed_genres": ["pop", "happy"], "count": 2},
        ],
        "angry": [
            {"target_valence": 0.3, "target_energy": 0.8, "seed_genres": ["rock", "metal"], "count": 2},
            {"target_valence": 0.5, "target_energy": 0.5, "seed_genres": ["alternative", "study"], "count": 2},
            {"target_valence": 0.7, "target_energy": 0.3, "seed_genres": ["ambient", "classical"], "count": 2},
        ],
        "fear": [
            {"target_valence": 0.5, "target_energy": 0.2, "seed_genres": ["classical", "piano"], "count": 2},
            {"target_valence": 0.6, "target_energy": 0.4, "seed_genres": ["acoustic", "folk"], "count": 2},
            {"target_valence": 0.8, "target_energy": 0.6, "seed_genres": ["soul", "pop"], "count": 2},
        ],
        "happy": [
            {"target_valence": 0.9, "target_energy": 0.8, "seed_genres": ["pop", "dance", "disco"], "count": 6},
        ],
        "neutral": [
            {"target_valence": 0.6, "target_energy": 0.6, "seed_genres": ["pop", "indie-pop"], "count": 6},
        ],
    }

    key = current_emotion.lower()
    steps = trajectories.get(key, trajectories["neutral"]).copy()
    playlist: List[Dict] = []

    # deduplication set to avoid repeats across stages
    seen_tracks = set()

    use_sp = sp_client is not None and SpotifyClient is not None
    use_local = local_csv_path is not None
    df_local = None
    if use_local:
        try:
            from recommendation.local_db import load_local_tracks, recommend_from_local_db
            df_local = load_local_tracks(local_csv_path)
            use_local = True
        except Exception as exc:
            _LOG.warning('Local CSV load failed (%s); falling back to other methods.', exc)
            df_local = None
            use_local = False

    for i, step in enumerate(steps):
        count = int(step.get('count', 2))
        recs = []

        # If using local DB, request a few extra items to allow filtering duplicates
        if use_local and df_local is not None:
            try:
                recs = recommend_from_local_db(df_local, step, limit=count + 5, preferred_genres=preferred_genres)
            except Exception:
                recs = []
        elif use_sp:
            try:
                recs = sp_client.recommend_tracks(seed_genres=step.get('seed_genres'),
                                                  target_valence=step.get('target_valence'),
                                                  target_energy=step.get('target_energy'),
                                                  limit=count)
            except Exception:
                recs = []
        else:
            recs = []

        # Add with deduplication
        added_this_stage = 0
        if recs:
            for r in recs:
                if added_this_stage >= count:
                    break

                track_key = r.get('uri') or r.get('spotify_url') or f"{r.get('title')}:{r.get('artist')}"
                if track_key in seen_tracks:
                    continue

                seen_tracks.add(track_key)
                r_copy = r.copy()
                r_copy['stage'] = i + 1
                r_copy['reason'] = f"stage={i+1}; emotion={key}"
                playlist.append(r_copy)
                added_this_stage += 1

            # if we added any tracks for this stage, move to next stage
            if added_this_stage > 0:
                continue

        # Fallback selection from CATALOG: simple heuristic mapping
        if key == 'happy':
            mood = 'uplift'
        elif key in ('sad', 'angry', 'fear'):
            if i == 0:
                mood = 'comfort'
            elif i == len(steps) - 1:
                mood = 'uplift'
            else:
                mood = 'calm'
        else:
            mood = 'comfort'

        picks = _select_by_mood(mood, top_k=count)
        for p in picks:
            track_key = f"{p.get('title')}:{p.get('artist')}"
            if track_key in seen_tracks:
                continue

            seen_tracks.add(track_key)
            p_copy = p.copy()
            p_copy['stage'] = i + 1
            p_copy['reason'] = f"fallback_stage={i+1}; emotion={key}"
            playlist.append(p_copy)

    return playlist


def recommend_music(combined: Dict, top_k: int = 3, user_pref: Optional[Dict] = None) -> List[Dict]:
    """Return a short list of recommended tracks (dicts) based on combined sentiment.

    - If sentiment is negative -> return uplifting tracks to improve mood.
    - If sentiment is positive -> return sustaining/comfort tracks to maintain mood.
    - If neutral -> return calm/comfort tracks.

    `user_pref` can be a dict like {'preferred_genres': ['pop','indie']} to bias results.
    """
    label = combined.get('label', 'neutral')
    if label == 'negative':
        mood = 'uplift'
    elif label == 'positive':
        mood = 'sustain'
    else:
        mood = 'comfort'

    results = _select_by_mood(mood, top_k=top_k)

    # Apply simple user preference bias if provided
    if user_pref and 'preferred_genres' in user_pref:
        preferred = user_pref['preferred_genres']
        preferred_hits = [r for r in results if r['genre'] in preferred]
        if preferred_hits:
            results = preferred_hits + [r for r in results if r not in preferred_hits]

    # Enrich recommendations with reason
    for r in results:
        r['reason'] = f"Mood match={mood}; sentiment_score={combined.get('score'):.2f}"

    return results


if __name__ == '__main__':
    # quick demo
    comb = aggregate_sentiments('sad', 0.9, 'neutral', 0.5)
    print('Combined:', comb)
    print('Recs:', recommend_music(comb))
