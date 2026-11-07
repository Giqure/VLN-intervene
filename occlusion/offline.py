"""Render archived poses with habitat_sim, without constructing a policy."""
import copy
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np

from . import runner
from .agents.base import Adapter
from .core import store
from .core.geom import quat_from_yaw, yaw_from_quat
from .core.rig import Pose
from .select.common import observation_pose, primary


def native_cameras(model, native):
    """Return the actual native sensor configs, including original resolutions."""
    if model == 'simplevln':
        cfg = native[1].habitat
        sensors = cfg.simulator.agents.main_agent.sim_sensors
        return cfg, list(sensors.values()), True
    cfg = native.TASK_CONFIG
    return cfg, [getattr(cfg.SIMULATOR, key) for key in cfg.SIMULATOR.AGENT_0.SENSORS], False


def _get(sensor, key, modern, default=None):
    return getattr(sensor, key if modern else key.upper(), default)


def poses(model, rows):
    if model == 'etpnav':
        for k, row in enumerate((r for r in rows if r.get('event') == 'decision'), 1):
            yaw = row.get('yaw')
            if yaw is None:
                yaw = yaw_from_quat(row['rotation'], 'xyzw')
            yield k, Pose(np.asarray(store._position(row), np.float64), float(yaw))
    else:
        count = sum(r.get('event') == 'step' for r in rows)
        for k in range(1, count + 1):
            position, yaw = observation_pose(rows, k)
            yield k, Pose(np.asarray(position, np.float64), yaw)


class Renderer(object):
    """Use the model's Habitat wrapper, not a hand-built sensor approximation.

    This preserves native near/far planes, sensor subtype/noise, scene dataset,
    simulator options and depth postprocessing across old/new Habitat versions.
    Only scene and (for the baseline) independent tracker sensors change.
    """
    def __init__(self, scene, simulator, modern, tracker_keys, augmented):
        from habitat.sims import make_sim
        if modern:
            from omegaconf import OmegaConf, open_dict
            cfg = copy.deepcopy(simulator)
            OmegaConf.set_readonly(cfg, False)
            with open_dict(cfg):
                cfg.scene = str(scene)
            if not augmented:
                for agent in cfg.agents.values():
                    sensors = agent.sim_sensors
                    with open_dict(sensors):
                        for key in list(sensors):
                            sensor = sensors[key]
                            uuid = _get(sensor, 'uuid', True) or key
                            if uuid in tracker_keys:
                                del sensors[key]
        else:
            cfg = simulator.clone()
            cfg.defrost()
            cfg.SCENE = str(scene)
            if not augmented:
                cfg.AGENT_0.SENSORS = [key for key in cfg.AGENT_0.SENSORS
                                       if _get(getattr(cfg, key), 'uuid', False) not in tracker_keys]
            cfg.freeze()
        self.sim = make_sim(id_sim=_get(cfg, 'type', modern), config=cfg)

    def at(self, pose):
        obs = self.sim.get_observations_at(position=pose.position.tolist(),
                                           rotation=quat_from_yaw(pose.yaw, 'xyzw'),
                                           keep_agent_at_new_pose=True)
        if obs is None:
            raise RuntimeError('native simulator could not render archived pose')
        return obs

    def close(self):
        self.sim.close()


def _rgb_hash(obs, keys):
    return {key: hashlib.sha256(np.ascontiguousarray(obs[key][..., :3]).tobytes()).hexdigest()
            for key in keys}


def render(model, batch, episodes=None, limit=None, check_rgb=False):
    with runner.exclusive(runner.batch_dir(model, batch) / '.offline.lock'):
        return _render(model, batch, episodes, limit, check_rgb)


def _render(model, batch, episodes=None, limit=None, check_rgb=False):
    params, candidates = runner.load_batch(model, batch)
    if episodes:
        wanted = set(map(str, episodes))
        candidates = [c for c in candidates if c['ep'] in wanted]
    if limit is not None:
        candidates = candidates[:limit]
    module = importlib.import_module('.agents.' + model, __package__)
    native, maps = module.configuration(params, attach=True)
    if maps[0].to_json() != params['rig']:
        raise ValueError('offline rig differs from immutable planned rig')
    cfg, cameras, modern = native_cameras(model, native)
    import habitat
    dataset_cfg = cfg.dataset if modern else cfg.DATASET
    dataset = habitat.datasets.make_dataset(id_dataset=(dataset_cfg.type if modern else dataset_cfg.TYPE),
                                             config=dataset_cfg)
    episode_map = {str(ep.episode_id): ep for ep in dataset.episodes}
    simulator = cfg.simulator if modern else cfg.SIMULATOR
    complete = []
    for cand in candidates:
        root = runner.batch_dir(model, batch) / 'offline' / cand['ep']
        output = root / 'render.json'
        if output.exists():
            old = json.loads(output.read_text())
            if old['params'] != params or old['cand'] != cand:
                raise ValueError('offline candidate/params mismatch')
            if not check_rgb or old['checks'].get('rgb_equal') is True:
                complete.append(dict(ep=cand['ep'], resumed=True))
                continue
        scene = Path(episode_map[cand['ep']].scene_id).resolve()
        if not scene.is_file():
            raise FileNotFoundError('native episode scene is unavailable: %s' % scene)
        trajectory = list(poses(model, primary(model, cand['split'], cand['ep'])))
        if not any(k == cand['step'] for k, _ in trajectory):
            raise ValueError('archived trajectory does not reach intervention start')
        baseline = {}
        if check_rgb:
            renderer = Renderer(scene, simulator, modern, set(maps[3].values()), False)
            try:
                for k, pose in trajectory:
                    baseline[k] = _rgb_hash(renderer.at(pose), maps[1].values())
            finally:
                renderer.close()
        bridge = Adapter(params, native, maps, root)
        bridge.params = dict(params, save_images=-1)
        bridge.begin(cand)
        renderer = Renderer(scene, simulator, modern, set(maps[3].values()), True)
        try:
            for k, pose in trajectory:
                obs = renderer.at(pose)
                if check_rgb and _rgb_hash(obs, maps[1].values()) != baseline[k]:
                    raise RuntimeError('adding tracker depth changed RGB at step %s' % k)
                bridge.observation_step = k
                bridge.pose = lambda: pose
                bridge.observe(obs)
        finally:
            renderer.close()
        record = dict(params=params, cand=cand, scene=str(scene), batch=batch,
                      trajectory=[dict(step=k, position=p.position.tolist(), yaw=p.yaw) for k, p in trajectory],
                      blackout=bridge.intervention.log() if bridge.intervention else [], images=bridge.images,
                      checks=dict(rgb_equal=True if check_rgb else None, rgb_poses_checked=len(baseline)),
                      at=runner.now())
        store.write_json_atomic(output, record)
        complete.append(dict(ep=cand['ep'], images=len(bridge.images), rgb_equal=record['checks']['rgb_equal']))
    return dict(batch=batch, rendered=complete)
