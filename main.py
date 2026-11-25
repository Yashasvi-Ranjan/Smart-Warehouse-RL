"""Smart Warehouse RL dashboard.  ``python main.py`` opens the app; ``python main.py --smoke`` runs a headless self-test."""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def smoke() -> int:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("MPLBACKEND", "Agg")
    from dashboard.app import App
    app = App()
    app.render_all_tabs()
    sim = app.sim
    for algo in ("random", "astar"):
        if not sim.set_agent(algo):
            print(f"smoke: {algo} unavailable: {sim.agent_error}")
            return 1
        sim.reset(new_task=True)
        stats = sim.run_episode()
        if stats is None or "reward" not in stats:
            print(f"smoke: {algo} produced no episode stats")
            return 1
        print(f"smoke: {algo}: success={stats['success']} steps={stats['steps']} reward={stats['reward']:.1f}")
    app.draw()
    import pygame
    pygame.quit()
    print("smoke: ok")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Smart Warehouse RL dashboard")
    ap.add_argument("--smoke", action="store_true", help="headless self-test, exits 0 on success")
    ap.add_argument("--size", default="1280x800")
    args = ap.parse_args(argv)
    if args.smoke:
        return smoke()
    from dashboard.app import App
    w, h = (int(x) for x in args.size.lower().split("x"))
    App((w, h)).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
