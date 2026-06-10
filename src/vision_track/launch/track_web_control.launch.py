"""Control-surface bringup for the upgraded track_web dashboard (follow demo).

Brings up the person tracker (with the track_web debug telemetry ENABLED) plus
the upgraded track_web dashboard in one command, as a focused "follow-demo
control surface":

    ros2 launch vision_track track_web_control.launch.py
    ros2 launch vision_track track_web_control.launch.py launch_tracker:=false
    ros2 launch vision_track track_web_control.launch.py port:=9000 with_waving:=true

The upgraded dashboard carries a new **Bringup panel** with per-component
start/stop toggles for the follow demo's supporting stack — audio
(``ros2 launch audio_pakage audio.launch.py``), dummy_nav
(``ros2 run behavior_tree dummy-nav``), and the follow-person behaviour tree
(``ros2 run behavior_tree follow-person``). Those are spawned on demand AT
RUNTIME by the dashboard's fixed-allowlist ProcessManager — they are
deliberately NOT part of this launch. For those spawns to resolve, build
``behavior_tree`` + ``audio_pakage`` into ``tk25_ws/install`` first via
``tkbuild tk25_decision`` + ``tkbuild tk_24_audio``.

Cameras are deliberately NOT included — start them first per
src/tk26_vision/CAMERA_BRINGUP.md (the vendored launches alone drop to ~3 Hz).

Arguments
---------
- ``launch_tracker`` (default ``true``) — gate the ``person_track_server`` Node.
  Set ``false`` to point the dashboard at an already-running tracker; the
  ``track_web`` dashboard Node always starts regardless.
- ``with_waving`` (default ``false``) — also start
  ``tk_vision_specialized/waving_person_server`` (the 👋 button backend). Off by
  default since the follow demo doesn't need it.
- ``kill_stale`` (default ``true``) — SIGTERM stale tracker/dashboard procs
  before starting (see below).
- ``bind`` (default ``0.0.0.0``) — dashboard bind address.
- ``port`` (default ``8766``) — dashboard HTTP port.
- ``perf_logging`` (default ``false``) — log per-frame tracker timings.

Why telemetry is forced ON
--------------------------
The dashboard renders the tracker's state/gallery/video only when the tracker
publishes them, and the three ``debug_*`` flags default OFF in code/yaml. This
launch forces ``debug_state_enabled`` / ``gallery_keep_crops`` /
``debug_image_enabled`` ON so the webui isn't blank — exactly like the bench.

Idempotent restart guard
-------------------------
Killing a launch ungracefully (terminal closed, SIGKILL) reparents the
tracker/dashboard nodes to PID 1. Each orphaned ``person_track_server`` keeps
squatting ~700 MiB of GPU and growing host RAM until the box swap-thrashes. To
make a restart never leave duplicates, ``kill_stale`` (default ``true``) runs a
cleanup ExecuteProcess FIRST that SIGTERMs any stale instances of the two
vision_track executables. The patterns are scoped to the installed
``lib/<pkg>/`` exec paths so a developer's editor, a ``grep``, or this very
launch's ``ros2 launch`` parent are never matched. The nodes start only after
the cleanup exits (via OnProcessExit); with ``kill_stale:=false`` they start
directly. ``generate_launch_description`` stays side-effect-free — the pkills
fire only when launch actually executes the ExecuteProcess, never at
description-build time (so the structural unit test never spawns a subprocess).

NOTE on the waving server + kill_stale: ``STALE_PATTERNS`` deliberately covers
ONLY the two vision_track execs (the tracker + dashboard). The waving server is
off by default (``with_waving:=false``), so a stale ``waving_person_server`` is
NOT auto-killed on restart — this keeps the cleanup minimal and only ever
SIGTERMs what this control surface owns by default. If you run with
``with_waving:=true`` repeatedly, clean a stale waving server manually.
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

# Narrow, lib/<pkg>/-scoped patterns for the two vision_track executables this
# control surface owns. The lib/<pkg>/ segment is load-bearing: it scopes pkill
# to the INSTALLED exec paths so an editor buffer, a grep, or the parent
# `ros2 launch` process are never matched. The waving server is intentionally
# absent (off by default; see the module docstring).
STALE_PATTERNS = (
    'lib/vision_track/person_track_server',
    'lib/vision_track/track_web',
)


def _control_nodes(tracker_condition, dashboard_condition, waving_condition):
    """Build the control-surface Node actions with the given conditions.

    A LaunchDescription cannot contain the same action instance twice and each
    action carries at most one condition, so the kill_stale=true branch (nodes
    nested in OnProcessExit.on_exit) and the kill_stale=false branch (nodes at
    top level) each get their own freshly-built set with their own conditions.
    ``tracker_condition`` gates the person tracker; ``dashboard_condition``
    gates the dashboard; ``waving_condition`` gates the waving server. Any may
    be ``None`` (no condition).
    """
    bind = LaunchConfiguration('bind')
    port = LaunchConfiguration('port')
    perf_logging = LaunchConfiguration('perf_logging')

    # Resolved at LAUNCH time (substitutions), keeping generate_launch_description
    # pure and unit-testable without a sourced workspace.
    tracker_params = PathJoinSubstitution(
        [FindPackageShare('vision_track'), 'config', 'default.yaml'])

    return [
        # Tracker: production config first, control-surface telemetry overrides
        # after — the three debug flags are default-OFF in code/yaml; this
        # launch (like the bench) is one place they are deliberately ON.
        Node(
            package='vision_track', executable='person_track_server',
            output='screen', condition=tracker_condition,
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
            condition=dashboard_condition,
            parameters=[{
                'bind': ParameterValue(bind, value_type=str),
                'port': ParameterValue(port, value_type=int),
            }]),
    ]


def generate_launch_description():
    launch_tracker = LaunchConfiguration('launch_tracker')
    with_waving = LaunchConfiguration('with_waving')
    kill_stale = LaunchConfiguration('kill_stale')
    not_kill_stale = NotSubstitution(kill_stale)

    # kill_stale=true: nodes start AFTER the cleanup exits. They run only because
    # the OnProcessExit handler (itself gated on kill_stale) fires, so the
    # ordering vs cleanup is already handled — the tracker just needs its
    # launch_tracker gate, the dashboard needs no extra condition, and the
    # waving server still honours with_waving.
    nodes_after_cleanup = _control_nodes(
        tracker_condition=IfCondition(launch_tracker),
        dashboard_condition=None,
        waving_condition=IfCondition(with_waving))

    # kill_stale=false: nodes start directly at top level, gated UNLESS
    # kill_stale. The tracker additionally honours launch_tracker (two gates
    # stacked via AndSubstitution); the dashboard only needs the kill_stale
    # gate; the waving server additionally honours with_waving.
    nodes_direct = _control_nodes(
        tracker_condition=IfCondition(
            AndSubstitution(not_kill_stale, launch_tracker)),
        dashboard_condition=UnlessCondition(kill_stale),
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
            'launch_tracker', default_value='true',
            description='start person_track_server (false = point the dashboard '
                        'at an already-running tracker; dashboard still starts)'),
        DeclareLaunchArgument(
            'with_waving', default_value='false',
            description='also start waving_person_server (the 👋 button backend); '
                        'off by default — the follow demo does not need it'),
        DeclareLaunchArgument(
            'bind', default_value='0.0.0.0',
            description='dashboard bind address (0.0.0.0 = operator laptop on LAN)'),
        DeclareLaunchArgument(
            'port', default_value='8766', description='dashboard HTTP port'),
        DeclareLaunchArgument(
            'perf_logging', default_value='false',
            description='log per-frame [perf] track/post/loop timings on the tracker'),
        DeclareLaunchArgument(
            'kill_stale', default_value='true',
            description='SIGTERM stale tracker/dashboard procs (narrow '
                        'lib/<pkg>/ patterns) before starting, so a restart '
                        'never leaves duplicates'),
        # kill_stale=true: clean up first, start nodes only after cleanup exits.
        cleanup,
        RegisterEventHandler(
            OnProcessExit(
                target_action=cleanup, on_exit=list(nodes_after_cleanup)),
            condition=IfCondition(kill_stale)),
        # kill_stale=false: start the nodes directly (no event handler).
        *nodes_direct,
    ])
