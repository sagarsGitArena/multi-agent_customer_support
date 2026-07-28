
import pytest
import json
from customer_support.tools import (
    search_albums_by_artist,
    search_tracks_by_artist,
    browse_songs_by_genre,
    search_songs_by_title,
    get_track_details
)

import logging

logger = logging.getLogger(__name__)

class TestMusicCatalogTools:
    def test_search_album_by_artist(self):
        result = search_albums_by_artist.invoke(
            {
                "artist_name": "AC/DC"
            }
        )
        assert "Let There Be Rock" in result

    def test_search_album_by_artist_no_match(self):
        result = search_albums_by_artist.invoke(
            {
                "artist_name": "ThisArtistDoesNotExistZZZ"
            }
        )
        data = json.loads(result)

        assert data == {
            "message": "No albums found for artist: ThisArtistDoesNotExistZZZ"
        }

    def test_search_tracks_by_artist(self):
        result = search_tracks_by_artist.invoke(
            {
                "artist_name": "AC/DC"
            }
        )
        data = json.loads(result)

        assert "total_tracks" in data
        assert "sample_tracks" in data
        assert isinstance(data["total_tracks"], int)
        assert data["total_tracks"] >= len(data["sample_tracks"])
        assert len(data["sample_tracks"]) <= 20

    def test_search_tracks_by_artist_no_match(self):
        result = search_tracks_by_artist.invoke(
            {
                "artist_name": "ThisArtistDoesNotExistZZZ"
            }
        )
        data = json.loads(result)

        assert data == {
            "message": "No tracks found for artist: ThisArtistDoesNotExistZZZ"
        }

    def test_browse_rock_genre(self):
        result = browse_songs_by_genre.invoke(
            {
                "genre_name": "Rock"
            }
        )
        data = json.loads(result)

        assert "total_tracks" in data
        assert "representative_tracks" in data
        tracks = data["representative_tracks"]
        assert len(tracks) <= 10

    def test_browse_genre_no_match(self):
        result = browse_songs_by_genre.invoke(
            {
                "genre_name": "ThisGenreDoesNotExistZZZ"
            }
        )
        data = json.loads(result)

        assert data == {
            "message": "No songs found for genre: ThisGenreDoesNotExistZZZ"
        }

    def test_search_songs_by_title(self):
        result = search_songs_by_title.invoke(
            {
                "song_title": "Rock and Roll"
            }
        )
        data = json.loads(result)

        assert len(data) <= 10
        assert all(
            "rock and roll" in track["track_title"].lower() for track in data
        )
        expected_fields = {
            "track_title",
            "artist_name",
            "album_title",
            "genre_name",
            "composer",
            "duration_seconds",
        }
        assert expected_fields.issubset(data[0].keys())

    def test_search_songs_by_title_no_match(self):
        result = search_songs_by_title.invoke(
            {
                "song_title": "ThisSongDoesNotExistZZZ"
            }
        )
        data = json.loads(result)

        assert data == {
            "message": "No songs found matching title: ThisSongDoesNotExistZZZ"
        }

    def test_get_track_details(self):
        result = get_track_details.invoke(
            {
                "track_id": "1"
            }
        )
        data = json.loads(result)

        assert len(data) == 1
        track = data[0]
        assert track["track_id"] == 1
        assert track["track_title"] == "For Those About To Rock (We Salute You)"
        assert track["artist_name"] == "AC/DC"
        expected_fields = {
            "track_id",
            "track_title",
            "artist_name",
            "album_title",
            "genre_name",
            "media_type",
            "composer",
            "duration_seconds",
            "unit_price",
        }
        assert expected_fields.issubset(track.keys())

    def test_get_track_details_not_found(self):
        result = get_track_details.invoke(
            {
                "track_id": "999999"
            }
        )
        data = json.loads(result)

        assert data == {"message": "No track found with ID: 999999"}

    def test_get_track_details_non_numeric(self):
        result = get_track_details.invoke(
            {
                "track_id": "abc"
            }
        )
        data = json.loads(result)

        assert "error" in data
