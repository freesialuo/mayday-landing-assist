# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import json

import pytest

from mayday_landing.compare_traces import SCHEMA, compare_traces, load_trace, main, summarize_trace


NAMES = ("visibility", "left_rpm", "right_valid", "left_repair", "left_valid", "left_fire",
         "right_thrust", "rain", "left_state", "wind_along")
BASE = dict.fromkeys(NAMES, 0.0) | {"left_valid": 1, "right_valid": 1, "left_state": 2}


def record(path, host, session, rows, names=NAMES):
    with path.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps({"schema": SCHEMA, "features": names}) + "\n")
        for index, (source, local, values, extras) in enumerate(rows):
            fields = BASE | values
            row = {
                "seq": index, "time": local, "session": session, "host": host, "valid": True,
                "sourceTime": source, "receivedAt": local, "engineSourceTime": None,
                "observation": [fields[name] for name in names],
            } | extras
            stream.write(json.dumps(row) + "\n")
    return load_trace(path)


def test_pairs_different_local_session_numbers_on_motion_clock(tmp_path):
    host = record(tmp_path / "host.jsonl", True, 4, [
        (20.0, 500.0, {"left_rpm": 0.4, "right_thrust": 0.7, "visibility": 400, "rain": 0.8}, {}),
        (20.1, 500.1, {"left_rpm": 0.6, "right_thrust": 0.7, "visibility": 400, "rain": 0.8}, {}),
        (20.2, 500.2, {"left_rpm": 0.7, "right_thrust": 0.7, "visibility": 400, "rain": 0.8}, {}),
    ])
    client = record(tmp_path / "client.jsonl", False, 29, [
        (20.0, 10.0, {"left_rpm": 0.5, "right_thrust": 0.6, "visibility": 0, "rain": 0.7}, {}),
        (20.11, 10.1, {"left_valid": 0, "left_rpm": 0.0, "right_thrust": 0.6, "visibility": 350}, {}),
        (20.2, 10.2, {"left_rpm": 0.0, "right_thrust": 0.0, "visibility": 1}, {"valid": False}),
        (20.2, 10.3, {"left_rpm": None, "right_valid": 0, "visibility": 370}, {}),
    ], names=NAMES[::-1])
    unconfirmed = compare_traces(host, client)
    assert unconfirmed["motion_alignment"]["status"] == "unavailable"
    paired = compare_traces(host, client, same_flight=True)
    assert paired["motion_alignment"]["host_session"] == 4
    assert paired["motion_alignment"]["client_session"] == 29
    assert paired["motion_alignment"]["paired_samples"] == 3
    rpm = paired["signals"]["left_rpm"]
    assert rpm["samples"] == 1
    assert rpm["mean_client_minus_host"] == pytest.approx(0.1)
    assert paired["signals"]["right_thrust"]["samples"] == 2
    assert paired["signals"]["visibility"]["samples"] == 2
    assert paired["signals"]["visibility"]["max_absolute_difference"] == 50
    assert paired["event_delay"]["status"] == "unavailable"
    assert "calibration" in paired["event_delay"]["reason"]


def test_estimates_only_uniquely_paired_edges_with_calibrated_sample_clocks(tmp_path):
    host = record(tmp_path / "host.jsonl", True, 1, [
        (1.0, 11.0, {}, {}),
        (1.1, 11.1, {"left_fire": 0.35, "left_repair": 0.0}, {}),
        (1.2, 11.2, {"left_fire": 0.36, "left_repair": 0.0}, {}),
        (1.3, 11.3, {"left_fire": 0.37, "left_repair": 0.5}, {}),
        (1.4, 11.4, {"left_fire": 0.38, "left_repair": 1.0}, {}),
        (1.5, 11.5, {"left_fire": 0.39, "left_repair": 1.0}, {}),
    ])
    client = record(tmp_path / "client.jsonl", False, 2, [
        (1.0, 1.02, {}, {}),
        (1.1, 1.12, {}, {}),
        (1.2, 1.22, {"left_fire": 0.35}, {}),
        (1.3, 1.32, {"left_fire": 0.36}, {}),
        (1.4, 1.42, {"left_fire": 0.37, "left_repair": 0.5}, {}),
        (1.5, 1.52, {"left_fire": 0.38, "left_repair": 1.0}, {}),
    ])
    assert compare_traces(host, client, same_flight=True)["event_delay"]["status"] == "unavailable"
    report = compare_traces(host, client, same_flight=True, client_clock_offset=10.0)["event_delay"]
    assert report["status"] == "estimated"
    assert [item["event"] for item in report["pairs"]] == [
        "left_fire_start", "left_repair_start", "left_repair_complete",
    ]
    fire = report["pairs"][0]
    assert fire["first_observation_delay_s"] == pytest.approx(0.12)
    assert fire["sampling_bounds_s"] == pytest.approx([0.02, 0.22])
    assert report["median_s"] == pytest.approx(0.12)
    bad_clock = compare_traces(host, client, same_flight=True, client_clock_offset=5.0)["event_delay"]
    assert bad_clock["status"] == "unavailable"
    assert bad_clock["inconsistent_pairs"] == 3


def test_invalid_gap_and_repeated_events_do_not_create_unambiguous_delay(tmp_path):
    rows = [
        (1.0, 1.0, {}, {}),
        (1.1, 1.1, {"left_fire": 1.0}, {}),
        (1.2, 1.2, {}, {}),
        (1.3, 1.3, {"left_fire": 1.0}, {}),
        (1.4, 1.4, {}, {}),
    ]
    host = record(tmp_path / "host.jsonl", True, 4, rows)
    client = record(tmp_path / "client.jsonl", False, 4, [
        (1.0, 1.0, {}, {}),
        (1.1, 1.1, {"left_fire": 1.0}, {}),
        (1.2, 1.2, {}, {"valid": False}),
        (1.3, 1.3, {}, {}),
        (1.4, 1.4, {}, {}),
    ])
    delays = compare_traces(host, client, same_flight=True, client_clock_offset=0.0)["event_delay"]
    assert delays["status"] == "unavailable"
    assert delays["host_edges"] == 4
    assert delays["client_edges"] == 1
    assert delays["pairs"] == []


def test_requires_unambiguous_local_session_and_monotonic_motion(tmp_path):
    host = record(tmp_path / "host.jsonl", True, 1, [(20.0, 5, {}, {}), (21.0, 6, {}, {})])
    client = record(tmp_path / "client.jsonl", False, 2, [
        (20.0, 7, {}, {}), (21.0, 8, {}, {}),
        (20.0, 30, {}, {"session": 3}), (21.0, 31, {}, {"session": 3}),
        (21.0, 50, {}, {"session": 4}), (20.0, 51, {}, {"session": 4}),
    ])
    assert "ambiguous" in compare_traces(host, client, same_flight=True)["motion_alignment"]["reason"]
    assert compare_traces(host, client, same_flight=True, client_session=2)["motion_alignment"]["status"] == "aligned"
    assert "no monotonic" in compare_traces(host, client, same_flight=True, client_session=4)["motion_alignment"]["reason"]


def test_single_recording_cli_keeps_signal_summary_without_motion_clock(tmp_path, capsys):
    path = tmp_path / "trace.jsonl"
    trace = record(path, False, 4, [
        (None, 4.0, {"left_rpm": 0.3}, {}),
        (None, 5.0, {"left_rpm": 0.9}, {"valid": False}),
        (None, 6.0, {"left_rpm": 0.9, "left_valid": 0.0}, {}),
    ])
    summary = summarize_trace(trace)
    assert summary[0]["motion_source_time_range_s"] is None
    assert summary[0]["valid_rows"] == 2
    assert summary[0]["signals"]["left_rpm"] == {"samples": 1, "min": 0.3, "max": 0.3}
    assert "visibility" not in summary[0]["signals"]
    assert main([str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["recording"] == summary


def test_rejects_invalid_observation_with_line_number(tmp_path):
    path = tmp_path / "trace.jsonl"
    with pytest.raises(ValueError, match=r"trace\.jsonl:2: valid row"):
        record(path, True, 6, [(1.0, 1.0, {}, {"observation": [1, 2, 3]})])
