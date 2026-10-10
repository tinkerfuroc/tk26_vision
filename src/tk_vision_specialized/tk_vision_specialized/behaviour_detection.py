"""Entry point for RGB-only person behaviour observation."""
from .rgb_scene_node import run_node


def main(args=None):
    run_node('behaviour_detection', 'behaviour', args)
