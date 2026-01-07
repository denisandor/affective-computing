Recommendation module
=====================

This folder contains a small rule-based music recommender that maps
multimodal sentiment (audio + face) to suggested tracks or genres.

Quick example:

```python
from recommendation.recommender import aggregate_sentiments, recommend_music
comb = aggregate_sentiments('sad', 0.9, 'neutral', 0.4)
recs = recommend_music(comb)
print(recs)
```

Optional Spotify integration:
- Install `spotipy` and set Spotify credentials.
- Use `recommendation.spotify_client.SpotifyClient` to search or fetch URIs.

Next steps:
- Integrate `recommend_music` into your fusion pipeline after inference.
- Optionally replace the static `CATALOG` with your dataset or Spotify search results.
