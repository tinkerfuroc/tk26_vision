#!/usr/bin/env python3
"""Interactive scenario bench client for `/detect_waving_persons`.

Walks an operator through the waving_bench scenario suite (see
config/waving_bench.yaml), fires the configured DetectWaving calls per case,
and judges the responses via `_waving_bench_eval` — no evaluation logic
lives in this file, only argument parsing, service I/O, printing, and
JSONL writing.
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import yaml

from ament_index_python.packages import (
    get_package_share_directory,
    PackageNotFoundError,
)
import rclpy
from rclpy.node import Node

from tinker_vision_msgs_26.srv import DetectWaving

from tk_vision_specialized._waving_bench_eval import (
    CallRecord,
    distance_of,
    evaluate_call,
    evaluate_case,
    load_suite,
    suite_passed,
)

DEFAULT_SERVICE = '/detect_waving_persons'
SERVICE_WAIT_TIMEOUT_SEC = 5.0


def record_from_response(resp) -> CallRecord:
    return CallRecord(
        status=resp.status,
        points=[(p.point.x, p.point.y, p.point.z) for p in resp.waving_persons],
        frame_ids=[p.header.frame_id for p in resp.waving_persons],
        error_msg=resp.error_msg,
    )


def _default_config_path() -> Path:
    """Installed share config, falling back to the source-tree copy."""
    try:
        share_dir = get_package_share_directory('tk_vision_specialized')
    except PackageNotFoundError:
        share_dir = None
    if share_dir is not None:
        candidate = Path(share_dir) / 'config' / 'waving_bench.yaml'
        if candidate.is_file():
            return candidate
    return Path(__file__).resolve().parent.parent / 'config' / 'waving_bench.yaml'


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='waving_bench',
        description=(
            'Interactive scenario bench for /detect_waving_persons '
            '(tinker_vision_msgs_26/srv/DetectWaving).'
        ),
    )
    parser.add_argument(
        '--scenario', action='append', metavar='NAME', dest='scenario',
        help='Run the named scenario. Repeatable.')
    parser.add_argument(
        '--all', action='store_true',
        help='Run every scenario in the suite.')
    parser.add_argument(
        '--config', metavar='PATH', default=None,
        help='Suite YAML path (default: installed share config, falling '
             'back to the source-tree copy).')
    parser.add_argument(
        '--out-dir', metavar='PATH', default=None,
        help='Output directory (default: /tmp/waving_bench_<YYYYmmdd_HHMMSS>).')
    parser.add_argument(
        '--service', default=DEFAULT_SERVICE,
        help="DetectWaving service name (default: '%(default)s').")
    parser.add_argument(
        '--calls', type=int, default=None, metavar='N',
        help='Override calls_per_case for every case (quick smoke).')
    parser.add_argument(
        '--yes', action='store_true',
        help='Non-interactive: skip operator prompts (still runs all selected cases).')
    return parser


def _write_json(fp, record: dict) -> None:
    fp.write(json.dumps(record) + '\n')


def _format_call_line(call_index: int, n_calls: int, record: CallRecord,
                       distances: list, verdict) -> str:
    mark = '✓' if verdict.passed else '✗'
    dist_str = ', '.join(f'{d:.2f}' for d in distances)
    line = (f'    call {call_index + 1}/{n_calls}: {mark} '
            f'status={record.status} n={len(record.points)} distances=[{dist_str}]')
    if not verdict.passed:
        line += f' reasons={verdict.reasons}'
    return line


def _run_case(node, client, scenario_name: str, case, jsonl) -> tuple:
    """Fire calls for one case, print + log per-call lines, return CallRecords."""
    calls = []
    for call_index in range(case.calls_per_case):
        request = DetectWaving.Request(
            threshold_meters=float(case.request['threshold_meters']),
            target_frame=str(case.request['target_frame']),
            min_waving_persons=int(case.request['min_waving_persons']),
        )
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future)
        record = record_from_response(future.result())
        calls.append(record)

        verdict = evaluate_call(case, record)
        distances = [distance_of(pt, case.request['target_frame']) for pt in record.points]
        print(_format_call_line(call_index, case.calls_per_case, record, distances, verdict))

        _write_json(jsonl, {
            'ts': datetime.now().astimezone().isoformat(),
            'scenario': scenario_name,
            'case_index': case.index,
            'prompt': case.prompt,
            'request': case.request,
            'status': record.status,
            'error_msg': record.error_msg,
            'points': [
                {'x': p[0], 'y': p[1], 'z': p[2], 'frame_id': fid}
                for p, fid in zip(record.points, record.frame_ids)
            ],
            'passed': verdict.passed,
            'reasons': verdict.reasons,
        })

        if call_index < case.calls_per_case - 1:
            time.sleep(case.interval_sec)

    return calls


def run(ns: argparse.Namespace, suite: dict, selected_names: list,
        results_path: Path) -> int:
    rclpy.init()
    node = Node('waving_bench')
    client = node.create_client(DetectWaving, ns.service)
    try:
        if not client.wait_for_service(timeout_sec=SERVICE_WAIT_TIMEOUT_SEC):
            print(
                f"error: service '{ns.service}' not available after "
                f"{SERVICE_WAIT_TIMEOUT_SEC:.1f}s wait "
                "(is waving_person_server running?)",
                file=sys.stderr,
            )
            return 1

        case_results = []
        cases_run = 0
        cases_skipped = 0
        quit_requested = False

        with results_path.open('w', buffering=1) as jsonl:
            for scenario_name in selected_names:
                for case in suite[scenario_name]:
                    if ns.calls is not None:
                        case = replace(case, calls_per_case=ns.calls)

                    print(f'[{scenario_name}/{case.index}] {case.prompt}')

                    if not ns.yes:
                        answer = input(
                            'Position the scene, then press Enter '
                            '(s=skip, q=quit): ').strip().lower()
                        if answer == 's':
                            cases_skipped += 1
                            print('  skipped')
                            _write_json(jsonl, {
                                'type': 'case_summary',
                                'scenario': scenario_name,
                                'case_index': case.index,
                                'n_passed': 0,
                                'n_calls': 0,
                                'passed': False,
                                'best_effort': case.best_effort,
                                'skipped': True,
                            })
                            continue
                        if answer == 'q':
                            quit_requested = True
                            break

                    calls = _run_case(node, client, scenario_name, case, jsonl)
                    result = evaluate_case(case, calls)
                    cases_run += 1
                    case_results.append(result)

                    print(f'  case summary: {result.n_passed}/{result.n_calls} passed '
                          f"-> {'PASS' if result.passed else 'FAIL'}"
                          f"{' (best-effort)' if case.best_effort else ''}")
                    _write_json(jsonl, {
                        'type': 'case_summary',
                        'scenario': scenario_name,
                        'case_index': case.index,
                        'n_passed': result.n_passed,
                        'n_calls': result.n_calls,
                        'passed': result.passed,
                        'best_effort': case.best_effort,
                        'skipped': False,
                    })

                if quit_requested:
                    break

            overall_passed = suite_passed(case_results)
            _write_json(jsonl, {
                'type': 'suite_summary',
                'passed': overall_passed,
                'cases_run': cases_run,
                'cases_skipped': cases_skipped,
            })

        print(f"\nSuite {'PASSED' if overall_passed else 'FAILED'} "
              f'({cases_run} run, {cases_skipped} skipped). Results: {results_path}')
        return 0 if overall_passed else 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main(args=None) -> None:
    parser = build_arg_parser()
    ns = parser.parse_args(args=args)

    config_path = Path(ns.config) if ns.config else _default_config_path()
    try:
        with config_path.open() as f:
            raw_config = yaml.safe_load(f) or {}
    except OSError as exc:
        print(f'error: could not read suite config {config_path}: {exc}', file=sys.stderr)
        sys.exit(1)

    try:
        suite = load_suite(raw_config)
    except ValueError as exc:
        print(f'error: invalid suite config {config_path}: {exc}', file=sys.stderr)
        sys.exit(1)

    if not ns.scenario and not ns.all:
        print('No scenario selected. Pass --scenario NAME (repeatable) or --all.')
        print('Available scenarios:')
        for name, cases in suite.items():
            plural = '' if len(cases) == 1 else 's'
            print(f'  {name} ({len(cases)} case{plural})')
        sys.exit(2)

    if ns.all:
        selected_names = list(suite.keys())
    else:
        unknown = [name for name in ns.scenario if name not in suite]
        if unknown:
            print(f'error: unknown scenario(s) {unknown}. Available: {list(suite.keys())}',
                  file=sys.stderr)
            sys.exit(2)
        selected_names = ns.scenario

    out_dir = Path(ns.out_dir) if ns.out_dir else Path(
        f"/tmp/waving_bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / 'results.jsonl'

    sys.exit(run(ns, suite, selected_names, results_path))


if __name__ == '__main__':
    main()
