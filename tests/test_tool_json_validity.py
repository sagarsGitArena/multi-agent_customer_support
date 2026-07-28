
import json

import pytest

from customer_support.tools import (
    search_albums_by_artist,
    search_tracks_by_artist,
    browse_songs_by_genre,
    search_songs_by_title,
    get_track_details,
    get_customer_invoices,
    get_customer_purchased_tracks,
    get_invoice_support_rep,
    get_invoice_line_items,
)

# Each entry: (tool, found_args, not_found_args, invalid_args or None)
TOOL_CASES = [
    (
        search_albums_by_artist,
        {"artist_name": "AC/DC"},
        {"artist_name": "ThisArtistDoesNotExistZZZ"},
        None,
    ),
    (
        search_tracks_by_artist,
        {"artist_name": "AC/DC"},
        {"artist_name": "ThisArtistDoesNotExistZZZ"},
        None,
    ),
    (
        browse_songs_by_genre,
        {"genre_name": "Rock"},
        {"genre_name": "ThisGenreDoesNotExistZZZ"},
        None,
    ),
    (
        search_songs_by_title,
        {"song_title": "Rock and Roll"},
        {"song_title": "ThisSongDoesNotExistZZZ"},
        None,
    ),
    (
        get_track_details,
        {"track_id": "1"},
        {"track_id": "999999"},
        {"track_id": "abc"},
    ),
    (
        get_customer_invoices,
        {"customer_id": "1"},
        {"customer_id": "999999"},
        {"customer_id": "abc"},
    ),
    (
        get_customer_purchased_tracks,
        {"customer_id": "1"},
        {"customer_id": "999999"},
        {"customer_id": "abc"},
    ),
    (
        get_invoice_support_rep,
        {"invoice_id": "1"},
        {"invoice_id": "999999"},
        {"invoice_id": "abc"},
    ),
    (
        get_invoice_line_items,
        {"invoice_id": "1"},
        {"invoice_id": "999999"},
        {"invoice_id": "abc"},
    ),
]


class TestToolOutputsAreValidJson:
    @pytest.mark.parametrize(
        "tool, args, not_found_args, invalid_args", TOOL_CASES
    )
    def test_found_case_is_valid_json(
        self, tool, args, not_found_args, invalid_args
    ):
        result = tool.invoke(args)

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, (list, dict))

    @pytest.mark.parametrize(
        "tool, args, not_found_args, invalid_args", TOOL_CASES
    )
    def test_not_found_case_is_valid_json(
        self, tool, args, not_found_args, invalid_args
    ):
        result = tool.invoke(not_found_args)

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, (list, dict))

    @pytest.mark.parametrize(
        "tool, args, not_found_args, invalid_args",
        [case for case in TOOL_CASES if case[3] is not None],
    )
    def test_invalid_input_case_is_valid_json(
        self, tool, args, not_found_args, invalid_args
    ):
        result = tool.invoke(invalid_args)

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "error" in parsed
