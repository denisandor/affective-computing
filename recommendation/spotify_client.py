"""Optional Spotify client wrapper. Uses `spotipy` when available.

This module is intentionally lightweight: if `spotipy` is not installed or
credentials are not provided, the client raises a clear error and the
rule-based recommender continues to work offline.
"""
from typing import List, Dict, Optional


try:
	import spotipy
	from spotipy.oauth2 import SpotifyClientCredentials
except Exception:  # pragma: no cover - optional dependency
	spotipy = None


class SpotifyClient:
	def __init__(self, client_id: Optional[str] = None, client_secret: Optional[str] = None):
		if spotipy is None:
			raise RuntimeError('spotipy is not installed. Install spotipy to use SpotifyClient.')
		if not client_id or not client_secret:
			raise ValueError('Provide Spotify `client_id` and `client_secret` to initialize SpotifyClient.')
		auth = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
		self._sp = spotipy.Spotify(client_credentials_manager=auth)

	def search_tracks(self, query: str, limit: int = 5) -> List[Dict]:
		"""Search Spotify and return a list of dicts with name, artists, uri."""
		res = self._sp.search(q=query, type='track', limit=limit)
		items = res.get('tracks', {}).get('items', [])
		out = []
		for it in items:
			out.append({'title': it['name'], 'artist': ', '.join(a['name'] for a in it['artists']), 'uri': it['uri']})
		return out

	def recommend_tracks(self, seed_genres: Optional[List[str]] = None,
	                     target_valence: Optional[float] = None,
	                     target_energy: Optional[float] = None,
	                     limit: int = 5) -> List[Dict]:
		"""Call Spotify's recommendations endpoint and return simplified track dicts.

		Parameters are passed directly to `spotipy.Spotify.recommendations`.
		"""
		query_args = {}
		if seed_genres:
			query_args['seed_genres'] = seed_genres[:5]
		if target_valence is not None:
			query_args['target_valence'] = float(target_valence)
		if target_energy is not None:
			query_args['target_energy'] = float(target_energy)

		res = self._sp.recommendations(limit=limit, **query_args)
		items = res.get('tracks', [])
		out = []
		for it in items:
			out.append({
				'title': it['name'],
				'artist': ', '.join(a['name'] for a in it['artists']),
				'uri': it['uri'],
				'album_art': it['album']['images'][0]['url'] if it['album'].get('images') else None,
			})
		return out
