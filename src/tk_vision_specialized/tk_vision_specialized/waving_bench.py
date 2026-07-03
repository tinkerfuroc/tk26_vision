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
from rclpy.signals import SignalHandlerOptions

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
# Per-call ceiling. Must exceed the server's 20 s VLM-fallback budget so a
# slow-but-alive keyed run is never misread as a dead/stalled server.
CALL_TIMEOUT_SEC = 30.0
NO_RESPONSE_REASON = 'no response from service (timeout or server death)'


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
        rclpy.spin_until_future_complete(node, future, timeout_sec=CALL_TIMEOUT_SEC)

        response, call_error = None, None
        if not future.done():
            future.cancel()
            call_error = NO_RESPONSE_REASON
        else:
            try:
                response = future.result()
            except Exception as exc:  # server died/restarted mid-call
                call_error = f'{NO_RESPONSE_REASON}: {exc}'
        if response is None and call_error is None:
            call_error = NO_RESPONSE_REASON

        if call_error is not None:
            # Sentinel record mirrors the server's own status=-1 error
            # convention; empty points make evaluate_case count this as a
            # failed call for every count/status expectation.
            record = CallRecord(status=-1, points=[], frame_ids=[], error_msg=call_error)
            passed, reasons = False, [call_error]
            print(f'    call {call_index + 1}/{case.calls_per_case}: ✗ {call_error}')
        else:
            record = record_from_response(response)
            verdict = evaluate_call(case, record)
            passed, reasons = verdict.passed, verdict.reasons
            distances = [distance_of(pt, case.request['target_frame'])
                         for pt in record.points]
            print(_format_call_line(call_index, case.calls_per_case, record,
                                    distances, verdict))
        calls.append(record)

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
            'passed': passed,
            'reasons': reasons,
        })

        if call_index < case.calls_per_case - 1:
            time.sleep(case.interval_sec)

    return calls


def run(ns: argparse.Namespace, suite: dict, selected_names: list,
        results_path: Path) -> int:
    # Keep Python's default SIGINT handling: rclpy's own handler would swallow
    # Ctrl-C at the input() prompt (no KeyboardInterrupt -> unkillable prompt).
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
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
        aborted = False

        with results_path.open('w', buffering=1) as jsonl:
            try:
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
            except (KeyboardInterrupt, EOFError):
                # Ctrl-C anywhere, or stdin closing at the operator prompt.
                aborted = True
                print('\naborted by operator')

            overall_passed = suite_passed(case_results)
            _write_json(jsonl, {
                'type': 'suite_summary',
                'passed': overall_passed,
                'cases_run': cases_run,
                'cases_skipped': cases_skipped,
            })

        outcome = 'ABORTED' if aborted else ('PASSED' if overall_passed else 'FAILED')
        print(f'\nSuite {outcome} '
              f'({cases_run} run, {cases_skipped} skipped). Results: {results_path}')
        if aborted:
            return 1
        return 0 if overall_passed else 1
    finally:
        node.destroy_node()
        if rclpy.ok():
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
    except yaml.YAMLError as exc:
        one_line = ' '.join(str(exc).split())
        print(f'error: could not parse suite config {config_path}: {one_line}',
              file=sys.stderr)
        sys.exit(1)

    try:
        suite = load_suite(raw_config)
    except ValueError as exc:
        print(f'error: invalid suite config {config_path}: {exc}', file=sys.stderr)
        sys.exit(1)

    # Exit-code split: 2 = usage/selection error (no/unknown scenario),
    # 1 = runtime/environment failure (bad config, service down, failed suite).
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

    try:
        sys.exit(run(ns, suite, selected_names, results_path))
    except (KeyboardInterrupt, EOFError):
        # Backstop for interrupts outside the case loop (e.g. during the
        # initial wait_for_service); the in-loop handler writes the summary.
        print('\naborted by operator', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
