import json
from langchain_core.tools import tool

from customer_support.db.database import execute_query
from customer_support.tools.validation import parse_id
from customer_support.tools.responses import not_found


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
        str: JSON string containing matching albums, or a
        human-readable message if none are found.
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
    if not json.loads(result):
        return not_found(f"No albums found for artist: {artist_name}")
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
        JSON string containing the total number of tracks by the
        artist and a sample of up to 20 tracks, or a human-readable
        message if none are found.
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

    count_result = json.loads(
        execute_query(tracks_count_query, params)
    )
    total_tracks = count_result[0]["total_tracks"]

    if total_tracks == 0:
        return not_found(f"No tracks found for artist: {artist_name}")

    sample_result = json.loads(
        execute_query(tracks_sample_query, params)
    )

    return json.dumps(
        {
            "total_tracks": total_tracks,
            "sample_tracks": sample_result
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
        JSON string containing the total number of songs in the
        genre and up to 10 representative tracks (one per artist),
        or a human-readable message if none are found.
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
    genre_sample_query = """
        SELECT
            artist_name,
            track_name,
            album_title,
            genre_name
        FROM
        (
            SELECT
                Artist.ArtistId AS artist_id,
                Artist.Name AS artist_name,
                Track.Name AS track_name,
                Album.Title AS album_title,
                Genre.Name AS genre_name,
                ROW_NUMBER() OVER (
                    PARTITION BY Artist.ArtistId
                    ORDER BY Track.TrackId
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
        ORDER BY artist_id
        LIMIT 10
    """
    
    
    count_result = json.loads(
        execute_query(genre_count_query, params)
    )
    total_tracks = count_result[0]["total_tracks"]

    if total_tracks == 0:
        return not_found(f"No songs found for genre: {genre_name}")

    sample_result = json.loads(
        execute_query(genre_sample_query, params)
    )

    return json.dumps(
        {
            "genre": genre_name,
            "total_tracks": total_tracks,
            "representative_tracks": sample_result
        },
        indent=2
    )


@tool
def search_songs_by_title(song_title: str) -> str:
    """
    Search songs by title using a fuzzy match.

    Use this tool when a user asks for:
    - a song
    - a track
    - songs with a particular title
    - tracks matching part of a title

    Args:
        song_title:
            Full or partial song title.

    Returns:
        JSON string containing up to 10 matching tracks, or a
        human-readable message if none are found.
    """

    query = """
        SELECT
            Track.Name AS track_title,
            Artist.Name AS artist_name,
            Album.Title AS album_title,
            Genre.Name AS genre_name,
            Track.Composer AS composer,
            ROUND(Track.Milliseconds / 1000.0, 1) AS duration_seconds
        FROM Track
        JOIN Album
            ON Track.AlbumId = Album.AlbumId
        JOIN Artist
            ON Album.ArtistId = Artist.ArtistId
        JOIN Genre
            ON Track.GenreId = Genre.GenreId
        WHERE Track.Name LIKE :song_title
        ORDER BY Track.Name
        LIMIT 10
    """

    params = {
        "song_title": f"%{song_title}%"
    }

    result = execute_query(query, params)
    if not json.loads(result):
        return not_found(f"No songs found matching title: {song_title}")
    return result


@tool
def get_track_details(track_id: str) -> str:
    """
    Get full details for a specific track by its ID.

    Use this tool when a user asks for:
    - details about a specific track
    - more information about a song they already found
    - the price, duration, or media type of a track

    Args:
        track_id (str): The unique ID of the track. Must be numeric.

    Returns:
        JSON string containing the track's full details, a
        human-readable message if no track matches the given ID, or
        an error message if track_id is not a valid number.
    """

    parsed_track_id, error = parse_id(track_id, "track_id")
    if error:
        return error

    query = """
        SELECT
            Track.TrackId AS track_id,
            Track.Name AS track_title,
            Artist.Name AS artist_name,
            Album.Title AS album_title,
            Genre.Name AS genre_name,
            MediaType.Name AS media_type,
            Track.Composer AS composer,
            ROUND(Track.Milliseconds / 1000.0, 1) AS duration_seconds,
            Track.UnitPrice AS unit_price
        FROM Track
        LEFT JOIN Album
            ON Track.AlbumId = Album.AlbumId
        LEFT JOIN Artist
            ON Album.ArtistId = Artist.ArtistId
        LEFT JOIN Genre
            ON Track.GenreId = Genre.GenreId
        JOIN MediaType
            ON Track.MediaTypeId = MediaType.MediaTypeId
        WHERE Track.TrackId = :track_id
    """

    params = {
        "track_id": parsed_track_id
    }

    result = execute_query(query, params)
    if not json.loads(result):
        return not_found(f"No track found with ID: {track_id}")
    return result