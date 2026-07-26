
import pytest
import json
from customer_support.tools import (
    search_albums_by_artist,
    search_tracks_by_artist,
    browse_songs_by_genre
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
        
        
    def test_search_tracks_by_artist(self):
        result = search_tracks_by_artist.invoke(
            {
                "artist_name": "AC/DC"
            }            
        )
        data = json.loads(result)
        
        assert "total_tracks" in data
        assert "sample_tracks" in data
        assert len(data["sample_tracks"]) <= 20
        
    def test_browse_rock_genre(self):
        result = browse_songs_by_genre.invoke(
            {
                "genre_name": "Rock"
            }
        )
        data = result # json.loads(result)
        
        assert "total_tracks" in data
        assert "representative_tracks" in data
        tracks = data["representative_tracks"]
        assert len(tracks) <= 10
