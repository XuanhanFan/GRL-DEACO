#!/usr/bin/env python3
"""Preferred CLI entry point for trained GRL-DEACO policy evaluation.

``test_model.py`` is kept as a compatibility entry point because earlier
experiments and documentation referenced it. New public reproduction commands
should use this file name instead.
"""

from test_model import main


if __name__ == "__main__":
    main()
