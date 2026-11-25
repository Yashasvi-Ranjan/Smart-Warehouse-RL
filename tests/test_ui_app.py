"""Headless tests of the pygame app: building, rendering every tab, synthetic input, background jobs."""
import os
import subprocess
import sys
import time

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MPLBACKEND", "Agg")

import pygame  # noqa: E402

from dashboard.app import App  # noqa: E402
from dashboard.controller import EvaluationJob, TrainingJob, build_env_config  # noqa: E402
from dashboard.widgets import Button, Checkbox, Dropdown, Slider, TextField  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def app():
    a = App((1280, 800))
    yield a
    pygame.quit()


def click(app, pos):
    for t in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
        app.handle_event(pygame.event.Event(t, pos=pos, button=1))


def key(app, k, unicode="", mod=0):
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=k, unicode=unicode, mod=mod))


def test_each_tab_renders(app):
    app.render_all_tabs()
    for i in range(4):
        app.switch_to(i)
        app.view.update(0.1)
        app.draw()
    assert app.screen.get_width() >= 1100


def test_resize_min_size(app):
    pygame.display.set_mode((900, 600), pygame.RESIZABLE)
    app.relayout()
    app.draw()
    assert app.screen.get_width() >= 1100 and app.screen.get_height() >= 700
    pygame.display.set_mode((1280, 800), pygame.RESIZABLE)
    app.relayout()


def test_tab_key_cycles(app):
    app.switch_to(0)
    key(app, pygame.K_TAB)
    assert app.current == 1
    key(app, pygame.K_TAB, mod=pygame.KMOD_SHIFT)
    assert app.current == 0


def test_tab_click(app):
    click(app, (112 * 2 + 8 + 40, 10))
    assert app.current == 2
    app.switch_to(0)


def test_sim_buttons_and_shortcuts(app):
    app.switch_to(0)
    sv = app.views[0]
    sv.set_agent = None
    app.sim.set_agent("astar")
    sv.sync_from_controller()
    sv.reset()
    key(app, pygame.K_s, "s")
    assert app.sim.steps == 1
    click(app, sv.btn_step.rect.center)
    assert app.sim.steps == 2
    key(app, pygame.K_SPACE, " ")
    assert sv.running
    for _ in range(300):
        sv.speed = 61
        sv.update(0.05)
        if app.sim.done:
            break
    assert app.sim.done and app.sim.result == "delivered"
    assert not sv.running
    app.draw()
    click(app, sv.btn_reset.rect.center)
    assert app.sim.steps == 0 and not app.sim.done
    before = sv.sl_speed.value
    key(app, pygame.K_EQUALS, "=")
    assert sv.sl_speed.value > before
    key(app, pygame.K_MINUS, "-")


def test_sim_dropdown_and_checkbox(app):
    sv = app.views[0]
    click(app, sv.dd_agent.rect.center)
    assert sv.dd_agent.open
    opt = pygame.Rect(sv.dd_agent.rect.x, sv.dd_agent.rect.bottom, sv.dd_agent.rect.w, 24)
    click(app, (opt.centerx, opt.y + 24 + 12))  # second option: astar
    assert not sv.dd_agent.open and app.sim.agent_key == "astar"
    n_before = app.sim.battery
    click(app, sv.chk_batt.rect.center)
    assert app.sim.battery != n_before
    click(app, sv.chk_batt.rect.center)
    assert app.sim.battery == n_before


def test_layout_seed_field(app):
    sv = app.views[0]
    click(app, sv.tf_seed.rect.center)
    assert sv.tf_seed.focused
    for _ in range(4):
        key(app, pygame.K_BACKSPACE)
    for ch in "17":
        key(app, ord(ch), ch)
    key(app, pygame.K_RETURN)
    assert app.sim.layout_seed == 17 and app.sim.env.layout_seed == 17
    click(app, sv.btn_next.rect.center)
    assert app.sim.layout_seed == 18


def test_textfield_validation():
    tf = TextField("12", "int", minv=0)
    assert tf.valid and tf.value == 12
    tf.set_text("abc")
    assert not tf.valid
    tf.set_text("-3")
    assert not tf.valid
    f = TextField("1.5", "float")
    assert f.value == 1.5
    f.focused = True
    f.handle_event(pygame.event.Event(pygame.KEYDOWN, key=ord("x"), unicode="x", mod=0))
    assert f.text == "1.5"  # non numeric char rejected


def test_slider_and_checkbox_widgets():
    got = []
    s = Slider(1, 61, 8, got.append, rect=(0, 0, 112, 24))
    s.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(106, 10), button=1))
    s.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(106, 10), button=1))
    assert s.value == 61 and got == [61]
    c = Checkbox("x", False, rect=(0, 0, 60, 24))
    c.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(5, 5), button=1))
    assert c.value


def test_model_browser(app):
    app.switch_to(0)
    sv = app.views[0]
    click(app, sv.btn_load.rect.center)
    assert sv.modal
    app.draw()
    if sv.m_models:
        sv.fl.selected = 0
        key(app, pygame.K_RETURN)
        assert app.sim.agent_key == sv.m_models[0]["algo"]
    else:
        key(app, pygame.K_ESCAPE)
    assert not sv.modal


def test_training_collect_and_validation(app):
    app.switch_to(1)
    tv = app.views[1]
    env_cfg, hp, seed, name = tv.collect()
    import yaml
    default_eps = yaml.safe_load(open("config.yaml"))["qlearning"]["episodes"]
    assert hp["episodes"] == default_eps and env_cfg.battery_enabled
    for f in tv.fields["qlearning"]:
        if f.key == "alpha":
            f.widget.set_text("zzz")
    tv.start()
    assert "alpha" in tv.msg and not tv.running
    for f in tv.fields["qlearning"]:
        if f.key == "alpha":
            f.widget.set_text("0.2")


def test_training_thread_real_qlearning(tmp_path):
    cfg = build_env_config(layout_seeds=[0])
    job = TrainingJob("qlearning", cfg, {"episodes": 300, "epsilon_decay_episodes": 200}, 0, "ui_test_run",
                      models_dir=str(tmp_path / "models"), results_dir=str(tmp_path / "res"))
    job.start()
    job.join(60)
    assert job.status == "done", job.error
    hist, latest = job.snapshot()
    assert hist and latest["episode"] == 300
    assert os.path.exists(job.result["model_path"]) and os.path.exists(job.result["log_path"])


def test_training_stop(tmp_path):
    cfg = build_env_config(layout_seeds=[0])
    job = TrainingJob("qlearning", cfg, {"episodes": 5_000_000}, 0, "ui_stop_run",
                      models_dir=str(tmp_path / "m"), results_dir=str(tmp_path / "r"))
    job.start()
    t0 = time.time()
    while time.time() - t0 < 20 and not job.snapshot()[0]:
        time.sleep(0.05)
    assert job.running
    job.stop()
    job.join(30)
    assert job.status == "done" and not job.thread.is_alive()
    assert os.path.exists(job.result["model_path"])


def test_training_error_surface(tmp_path):
    cfg = build_env_config()
    job = TrainingJob("qlearning", cfg, {"bogus": 1}, 0, "ui_err", models_dir=str(tmp_path), results_dir=str(tmp_path))
    job.start()
    job.join(10)
    assert job.status == "error" and "bogus" in job.error


def test_training_view_runs_job_and_renders(app, tmp_path, monkeypatch):
    app.switch_to(1)
    tv = app.views[1]
    for f in tv.fields["qlearning"]:
        if f.key == "episodes":
            f.widget.set_text("200")
    tv.tf_name.set_text("ui_view_test")
    import dashboard.training_view as tvm
    monkeypatch.setattr(tvm, "MODELS_DIR", str(tmp_path / "m"))
    monkeypatch.setattr(tvm, "RESULTS_DIR", str(tmp_path / "r"))
    click(app, tv.btn_start.rect.center)
    assert tv.job is not None
    tv.job.join(60)
    tv.update(0.1)
    app.draw()
    assert tv.job.status == "done" and tv.btn_use.visible
    n = app.sim.episode_no
    click(app, tv.btn_use.rect.center)
    assert app.current == 0 and app.sim.agent_key == "qlearning"
    assert "ui_view_test" in app.sim.agent_label


def test_evaluation_job_identical_seeds():
    from agents import make_agent
    cfg = build_env_config(layout_seeds=[0, 1])
    job = EvaluationJob({"a": make_agent("astar"), "b": make_agent("astar")}, cfg, [100000, 100001, 100002], [0, 1])
    job.start()
    job.join(60)
    assert job.status == "done", job.error
    df = job.long_df
    a, b = df[df.label == "a"], df[df.label == "b"]
    assert list(a.seed) == list(b.seed) and list(a.layout_seed) == list(b.layout_seed)
    assert list(a.reward) == list(b.reward)
    assert job.progress == pytest.approx(1.0)
    assert set(job.summary.index) == {"a", "b"}


def test_evaluation_view_flow_and_replay(app):
    app.switch_to(3)
    ev = app.views[3]
    ev.agents.checked = {"random", "astar"}
    ev.tf_eps.set_text("4")
    click(app, ev.btn_run.rect.center)
    ev.job.join(60)
    ev.update(0.1)
    ev.update(0.1)
    app.draw()
    assert len(ev.table.rows) == 2 and ev.chart is not None
    ev.table.selected = 1
    ev.tf_seed.set_text("100001")
    click(app, ev.btn_replay.rect.center)
    assert app.current == 0
    assert app.sim.task_seed == 100001 and app.sim.layout_seeds_set is not None


def test_analytics_view_modes(app, tmp_path, monkeypatch):
    import dashboard.analytics_view as avm
    monkeypatch.setattr(avm, "EXPORT_DIR", str(tmp_path))
    app.switch_to(2)
    av = app.views[2]
    for mode in ("curves", "compare", "general"):
        av.dd_mode.value = mode
        av._mode_changed(mode)
        av._render()
        av.update(0.5)
        app.draw()
    av.dd_mode.value = "curves"
    av._mode_changed("curves")
    if av.runs.items:
        av.runs.checked = {av.runs.items[0][0]}
        av._render()
        assert av.surface is not None
        av.export()
        assert "Exported" in av.msg


def test_smoke_entrypoint():
    env = dict(os.environ, SDL_VIDEODRIVER="dummy", MPLBACKEND="Agg")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "main.py"), "--smoke"], capture_output=True, text=True,
                       timeout=120, env=env, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "smoke: ok" in r.stdout
