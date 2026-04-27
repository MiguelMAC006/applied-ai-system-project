from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Song:
    """Represents a song and its audio features."""
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float


@dataclass
class UserProfile:
    """Stores a user's stated taste preferences."""
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    target_valence: float
    target_acousticness: float
    target_tempo_bpm: float
    target_danceability: float


class Recommender:
    """OOP wrapper around the scoring and recommendation logic."""

    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        user_prefs = {
            "genre":        user.favorite_genre,
            "mood":         user.favorite_mood,
            "energy":       user.target_energy,
            "valence":      user.target_valence,
            "acousticness": user.target_acousticness,
            "tempo_bpm":    user.target_tempo_bpm,
            "danceability": user.target_danceability,
        }
        scored = []
        for song in self.songs:
            song_dict = _song_to_dict(song)
            total, _ = score_song(user_prefs, song_dict)
            scored.append((song, total))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        user_prefs = {
            "genre":        user.favorite_genre,
            "mood":         user.favorite_mood,
            "energy":       user.target_energy,
            "valence":      user.target_valence,
            "acousticness": user.target_acousticness,
            "tempo_bpm":    user.target_tempo_bpm,
            "danceability": user.target_danceability,
        }
        _, reasons = score_song(user_prefs, _song_to_dict(song))
        return " · ".join(reasons) if reasons else "no matching criteria"


def _song_to_dict(song: Song) -> dict:
    return {
        "id": song.id, "title": song.title, "artist": song.artist,
        "genre": song.genre, "mood": song.mood, "energy": song.energy,
        "tempo_bpm": song.tempo_bpm, "valence": song.valence,
        "danceability": song.danceability, "acousticness": song.acousticness,
    }


def load_songs(csv_path: str) -> List[Dict]:
    """
    Parse songs.csv and return a list of dicts with numeric fields cast.

    Guardrails:
    - Missing or non-numeric fields are logged and the row is skipped.
    """
    import csv

    songs = []
    required = ("id", "title", "artist", "genre", "mood",
                 "energy", "tempo_bpm", "valence", "danceability", "acousticness")

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not all(k in row for k in required):
                logger.warning("Skipping row with missing fields: %s", row)
                continue
            try:
                songs.append({
                    "id":           int(row["id"]),
                    "title":        row["title"],
                    "artist":       row["artist"],
                    "genre":        row["genre"],
                    "mood":         row["mood"],
                    "energy":       float(row["energy"]),
                    "tempo_bpm":    float(row["tempo_bpm"]),
                    "valence":      float(row["valence"]),
                    "danceability": float(row["danceability"]),
                    "acousticness": float(row["acousticness"]),
                })
            except (ValueError, KeyError) as exc:
                logger.warning("Skipping malformed row %s: %s", row.get("id", "?"), exc)

    logger.info("Loaded %d songs from %s", len(songs), csv_path)
    return songs


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Return (total_score, reasons) for one song. Max possible score: 7.75."""
    score = 0.0
    reasons: List[str] = []

    if song["genre"] == user_prefs.get("genre", ""):
        score += 1.0
        reasons.append(f"genre match ({song['genre']}) +1.0")

    if song["mood"] == user_prefs.get("mood", ""):
        score += 1.0
        reasons.append(f"mood match ({song['mood']}) +1.0")

    if "energy" in user_prefs:
        pts = 3.0 * (1 - abs(song["energy"] - user_prefs["energy"]))
        score += pts
        reasons.append(f"energy similarity +{pts:.2f}")

    if "valence" in user_prefs:
        pts = 1.0 * (1 - abs(song["valence"] - user_prefs["valence"]))
        score += pts
        reasons.append(f"valence similarity +{pts:.2f}")

    if "danceability" in user_prefs:
        pts = 0.75 * (1 - abs(song["danceability"] - user_prefs["danceability"]))
        score += pts
        reasons.append(f"danceability similarity +{pts:.2f}")

    if "acousticness" in user_prefs:
        pts = 0.5 * (1 - abs(song["acousticness"] - user_prefs["acousticness"]))
        score += pts
        reasons.append(f"acousticness similarity +{pts:.2f}")

    if "tempo_bpm" in user_prefs:
        raw_diff = abs(song["tempo_bpm"] - user_prefs["tempo_bpm"]) / 120
        pts = max(0.0, 0.5 * (1 - raw_diff))
        score += pts
        reasons.append(f"tempo similarity +{pts:.2f}")

    return score, reasons


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Score every song in the catalog, sort descending, return top-k."""
    scored = []
    for song in songs:
        total_score, reasons = score_song(user_prefs, song)
        explanation = " · ".join(reasons)
        scored.append((song, total_score, explanation))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
