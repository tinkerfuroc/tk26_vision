"""Entry point for VLM person behaviours with registered-depth localization."""
from .rgb_scene_node import run_node


def main(args=None):
    run_node('behaviour_detection', 'behaviour', args)
