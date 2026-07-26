import json
from langchain_core.tools import tool

from customer_support.db.database import execute_query


@tool
def search_albums_by_artist(artist_name: str) -> str:
    """
    Search albums by artist name
    
    Use this tool when a user asks for:
    - albums by an artist
    - music collection by an artist
    - artist discography

    Args:
        artist_name (str): Name of the artist

    Returns:
        str: JSON string containing matching albums.
    """
    
    query = """
        SELECT
            Artist.Name AS artist_name,
            Album.Title AS album_title
        FROM Artist
        JOIN Album
            on Artist.ArtistId = Album.ArtistId
        WHERE Artist.Name LIKE :artist_name
        ORDER BY Album.Title
    """
    params = {
        "artist_name": f"%{artist_name}%"
    }
    
    result = execute_query(query, params)
    print(result)
    return result


@tool
def search_tracks_by_artist(artist_name: str) -> str:
    """
    Search tracks by artist name
    
    Use this tool when a user asks for:
    - songs by an artist
    - tracks by an artist
    - music available from an artist

    Args:
        artist_name (str): Name of the artist

    Returns:
        - total number of tracks by the artist
        - sample of up to 20 tracks
    """
    tracks_count_query = """
        SELECT  COUNT(*) AS total_tracks
        FROM Artist
        JOIN Album
            ON Artist.ArtistId = Album.ArtistId
        JOIN Track
            ON Album.AlbumId = Track.AlbumId
        WHERE Artist.Name LIKE :artist_name
        
    """
    tracks_sample_query = """
        SELECT
            Artist.Name AS artist_name,
            Album.Title AS album_title,
            Track.Name AS track_name,
            Track.Milliseconds AS duration_ms
        FROM Artist
        JOIN Album
            ON Artist.ArtistId = Album.ArtistId
        JOIN Track
            ON Album.AlbumId = Track.AlbumId
        WHERE Artist.Name LIKE :artist_name
        ORDER BY Album.Title, Track.Name
        LIMIT 20
    """
    params = {
        "artist_name": f"%{artist_name}%"
    }
    
    count_result = execute_query(
        tracks_sample_query,
        params
    )

    sample_result = execute_query(
        tracks_sample_query,
        params
    )
    
    
    
    return json.dumps(
        {
            "total_tracks": json.loads(count_result),
            "sample_tracks": json.loads(sample_result)
        },
        indent=2
    )
    
    
@tool
def browse_songs_by_genre(genre_name: str) -> str:
    """
    Browse songs by genre

    Args:
        genre_name (str): Genre name such as Rock, Jazz, Metal, etc.

    Returns:
        - total number of songs in the genre
        - up to 10 representative tracks
        - one track per artist to ensure diversity
    """
    
    params = {
        "genre_name": f"%{genre_name}%"
    }
    
    genre_count_query = """
        SELECT COUNT(*) AS total_tracks
        FROM Track
        JOIN Genre
            ON Track.GenreId = Genre.GenreId
        WHERE Genre.Name LIKE :genre_name
    """
    
    
    # SELECT
    #     Artist.Name AS artist_name,
    #     Track.Name AS track_name,
    #     Album.Title AS album_title,
    #     Genre.Name AS genre_name
    # FROM Track
    # JOIN Genre
    #     ON Track.GenreId = Genre.GenreId
    # JOIN Album
    #     ON Track.AlbumId = Album.AlbumId
    # JOIN Artist
    #     ON Album.ArtistId = Artist.ArtistId
    # WHERE Genre.Name LIKE :genre_name
    # GROUP BY Artist.ArtistId
    # ORDER BY Artist.Name
    # LIMIT 10
    genre_sample_query = """
        SELECT
            artist_name,
            track_name,
            album_title,
            genre_name
        FROM
        (
            SELECT
                Artist.Name AS artist_name,
                Track.Name AS track_name,
                Album.Title AS album_title,
                Genre.Name AS genre_name,
                ROW_NUMBER() OVER (
                    PARTITION BY Artist.ArtistId
                    ORDER BY Track.Name
                ) AS rn

            FROM Track

            JOIN Genre
                ON Track.GenreId = Genre.GenreId

            JOIN Album
                ON Track.AlbumId = Album.AlbumId

            JOIN Artist
                ON Album.ArtistId = Artist.ArtistId

            WHERE Genre.Name LIKE :genre_name
        )
        WHERE rn = 1
        LIMIT 10
    """
# But there is a subtle SQL issue
# The query:
# GROUP BY Artist.ArtistId
# works in SQLite, but SQLite chooses an arbitrary track from that artist.
# For a demo project this is acceptable.
# However, for production-quality SQL, I would explicitly choose the first track.
# A better SQLite approach:
# SELECT
#     artist_name,
#     track_name,
#     album_title,
#     genre_name
# FROM
# (
#     SELECT
#         Artist.Name AS artist_name,
#         Track.Name AS track_name,
#         Album.Title AS album_title,
#         Genre.Name AS genre_name,
#         ROW_NUMBER() OVER (
#             PARTITION BY Artist.ArtistId
#             ORDER BY Track.Name
#         ) AS rn
#     FROM Track
#     JOIN Genre
#         ON Track.GenreId = Genre.GenreId
#     JOIN Album
#         ON Track.AlbumId = Album.AlbumId
#     JOIN Artist
#         ON Album.ArtistId = Artist.ArtistId
#     WHERE Genre.Name LIKE :genre_name
# )

# WHERE rn = 1

# LIMIT 10

# This says:

# For each artist:
#      rank their tracks

# Pick:
#      track number 1

# Then:
#      return max 10 artists

# This is the better implementation.    
    
    
    count_result = json.loads(
        execute_query(genre_count_query, params)
    )

    sample_result = json.loads(
        execute_query(genre_sample_query, params)
    )

    return json.dumps(
        {
            "genre": genre_name,
            "total_tracks": count_result[0]["total_tracks"],
            "representative_tracks": sample_result
        },
        indent=2
    )
    