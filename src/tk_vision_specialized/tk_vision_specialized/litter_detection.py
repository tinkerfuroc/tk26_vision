"""Entry point for Orbbec RGB-only floor litter detection."""
from .rgb_scene_node import run_node


def main(args=None):
    run_node('litter_detection', 'litter', args)
