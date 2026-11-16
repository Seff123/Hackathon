"""Shared test helpers."""

import numpy as np


def truth_types():
    from floodsim.flood import FloodTimeline, Storm, LocalInertialSolver
    from floodsim.people import Placement
    from floodsim.terrain import Terrain
    from floodsim.world import World

    types = [FloodTimeline, Storm, LocalInertialSolver, Placement, Terrain, World]
    try:
        from floodsim.weather import Weather

        types.append(Weather)
    except ImportError:
        pass
    try:
        from floodsim.people import People

        types.append(People)
    except ImportError:
        pass
    try:
        from floodsim.glider import GliderState

        types.append(GliderState)
    except ImportError:
        pass
    return tuple(types)


def reachable_truth(obj, max_depth=12):
    """Return a path to the first ground-truth object reachable from obj, else None."""
    bad = truth_types()
    seen = set()
    stack = [(obj, "root", 0)]
    while stack:
        o, path, d = stack.pop()
        if id(o) in seen or d > max_depth:
            continue
        seen.add(id(o))
        if isinstance(o, bad):
            return path
        if isinstance(o, (np.ndarray, str, bytes, int, float, bool, type(None), np.generic)):
            continue
        if callable(o) and not hasattr(o, "__dict__"):
            continue
        if isinstance(o, dict):
            for k, v in o.items():
                stack.append((v, f"{path}[{k!r}]", d + 1))
        elif isinstance(o, (list, tuple, set, frozenset)):
            for i, v in enumerate(o):
                stack.append((v, f"{path}[{i}]", d + 1))
        else:
            slf = getattr(o, "__self__", None)  # bound methods
            if slf is not None:
                stack.append((slf, f"{path}.__self__", d + 1))
            if hasattr(o, "__dict__"):
                for k, v in vars(o).items():
                    stack.append((v, f"{path}.{k}", d + 1))
            for k in getattr(type(o), "__slots__", ()):
                if hasattr(o, k):
                    stack.append((getattr(o, k), f"{path}.{k}", d + 1))
    return None
