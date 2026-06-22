"""Bench bringup for the track_web dashboard.

Starts the person tracker (with the track_web debug telemetry ENABLED), the
waving server, and the dashboard in one command:

    ros2 launch vision_track track_web_bench.launch.py
    ros2 launch vision_track track_web_bench.launch.py port:=9000 with_waving:=false

Cameras are deliberately NOT included — start them first per
src/tk26_vision/CAMERA_BRINGUP.md (the vendored launches alone drop to ~3 Hz).

Idempotent restart guard
-------------------------
Across a day of bench restarts, killing a launch ungracefully (terminal closed,
SIGKILL) reparents the tracker/dashboard/waving nodes to PID 1. Each orphaned
``person_track_server`` keeps squatting ~700 MiB of GPU and growing host RAM
until the box swap-thrashes — ssh drops and the Orbbec depth-engine SIGSEGVs.
To make a restart never leave duplicates, ``kill_stale`` (default ``true``)
runs a cleanup ExecuteProcess FIRST that SIGTERMs any stale instances of the
three bench executables. The patterns are scoped to the installed
``lib/<pkg>/`` exec paths so a developer's editor, a ``grep``, or this very
launch's ``ros2 launch`` parent are never matched. The nodes start only after
the cleanup exits (via OnProcessExit); with ``kill_stale:=false`` they start
directly. ``generate_launch_description`` stays side-effect-free — the pkills
fire only when launch actually executes the ExecuteProcess, never at
description-build time (so the structural unit test never spawns a subprocess).
The same three narrow patterns live in ``scripts/kill_stale_bench.sh`` for
manual cleanup.
"""
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            RegisterEventHandler)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import (AndSubstitution, LaunchConfiguration,
                                  NotSubstitution, PathJoinSubstitution)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

# Narrow, lib/<pkg>/-scoped patterns for the three bench executables. The
# lib/<pkg>/ segment is load-bearing: it scopes pkill to the INSTALLED exec
# paths so an editor buffer, a grep, or the parent `ros2 launch` process are
# never matched. Keep this list identical to scripts/kill_stale_bench.sh.
STALE_PATTERNS = (
    'lib/vision_track/person_track_server',
    'lib/vision_track/track_web',
    'lib/tk_vision_specialized/waving_person_server',
)


def _bench_nodes(node_condition, waving_condition):
    """Build the three bench Node actions with the given conditions.

    A LaunchDescription cannot contain the same action instance twice and each
    action carries at most one condition, so the kill_stale=true branch (nodes
    nested in OnProcessExit.on_exit) and the kill_stale=false branch (nodes at
    top level) each get their own freshly-built set with their own conditions.
    ``node_condition`` gates the tracker + dashboard; ``waving_condition`` gates
    the waving server (which additionally honours with_waving). Either may be
    ``None`` (no condition).
    """
    bind = LaunchConfiguration('bind')
    port = LaunchConfiguration('port')
    perf_logging = LaunchConfiguration('perf_logging')

    # Resolved at LAUNCH time (substitutions), keeping generate_launch_description
    # pure and unit-testable without a sourced workspace.
    tracker_params = PathJoinSubstitution(
        [FindPackageShare('vision_track'), 'config', 'default.yaml'])

    return [
        # Tracker: production config first, bench telemetry overrides after —
        # the three debug flags are default-OFF in code/yaml; the bench is the
        # one place they are deliberately ON.
        Node(
            package='vision_track', executable='person_track_server',
            output='screen', condition=node_condition,
            parameters=[
                tracker_params,
                {'debug_state_enabled': True,
                 'gallery_keep_crops': True,
                 'debug_image_enabled': True,
                 'perf_logging_enabled': ParameterValue(
                     perf_logging, value_type=bool)},
            ]),
        Node(
            package='tk_vision_specialized', executable='waving_person_server',
            output='screen', condition=waving_condition),
        Node(
            package='vision_track', executable='track_web', output='screen',
            condition=node_condition,
            parameters=[{
                'bind': ParameterValue(bind, value_type=str),
                'port': ParameterValue(port, value_type=int),
            }]),
    ]


def generate_launch_description():
    with_waving = LaunchConfiguration('with_waving')
    kill_stale = LaunchConfiguration('kill_stale')
    not_kill_stale = NotSubstitution(kill_stale)

    # kill_stale=true: nodes start AFTER the cleanup exits. They run only because
    # the OnProcessExit handler (itself gated on kill_stale) fires, so the
    # tracker/dashboard need no extra condition; the waving server still honours
    # with_waving.
    nodes_after_cleanup = _bench_nodes(
        node_condition=None,
        waving_condition=IfCondition(with_waving))

    # kill_stale=false: nodes start directly at top level, gated UNLESS
    # kill_stale; the waving server additionally honours with_waving.
    nodes_direct = _bench_nodes(
        node_condition=UnlessCondition(kill_stale),
        waving_condition=IfCondition(
            AndSubstitution(not_kill_stale, with_waving)))

    # Idempotent pre-launch cleanup. SIGTERM (default, NOT -9) lets each stale
    # node release the camera/GPU cleanly. Best-effort: pkill returns non-zero
    # when nothing matches, so `|| true` swallows it. Echo the patterns so the
    # operator sees what is being cleaned. Gated on kill_stale; built here but
    # only executed at launch time (no subprocess at description-build time).
    pkill_cmds = ' ; '.join(
        f"echo '[kill_stale] SIGTERM stale: {p}' ; pkill -f '{p}' || true"
        for p in STALE_PATTERNS)
    cleanup = ExecuteProcess(
        cmd=['bash', '-lc', pkill_cmds],
        output='screen',
        condition=IfCondition(kill_stale),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'bind', default_value='0.0.0.0',
            description='dashboard bind address (0.0.0.0 = operator laptop on LAN)'),
        DeclareLaunchArgument(
            'port', default_value='8766', description='dashboard HTTP port'),
        DeclareLaunchArgument(
            'with_waving', default_value='true',
            description='also start waving_person_server (the 👋 button backend)'),
        DeclareLaunchArgument(
            'perf_logging', default_value='false',
            description='log per-frame [perf] track/post/loop timings on the tracker'),
        DeclareLaunchArgument(
            'kill_stale', default_value='true',
            description='SIGTERM stale bench procs (narrow lib/<pkg>/ patterns) '
                        'before starting, so a restart never leaves duplicates'),
        # kill_stale=true: clean up first, start nodes only after cleanup exits.
        cleanup,
        RegisterEventHandler(
            OnProcessExit(
                target_action=cleanup, on_exit=list(nodes_after_cleanup)),
            condition=IfCondition(kill_stale)),
        # kill_stale=false: start the nodes directly (no event handler).
        *nodes_direct,
    ])
