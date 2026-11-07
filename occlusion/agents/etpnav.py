"""ETPNav native evaluation, single environment, without split-depth forwarding.

The wrapped VectorEnv changes observations BEFORE native batch_obs. Both
waypoint and planner therefore receive the same blacked depth in rgbd runs.
"""
import copy
import os
import random
import sys
from pathlib import Path

import numpy as np

from .base import Adapter, assert_pose, native_rig
from ..core.geom import yaw_from_quat
from ..core.rig import Pose
from ..select.common import OPEN
from ..select.etpnav import source_logs

V13 = 'data/datasets/R2R_VLNCE_v1-3_preprocessed'
OPTS = ['SIMULATOR_GPU_IDS', '[0]', 'TORCH_GPU_IDS', '[0]', 'GPU_NUMBERS', '1',
        'NUM_ENVIRONMENTS', '1', 'TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING', 'True',
        'MODEL.pretrained_path', 'data/pretrained/ETP/mlm.sap_r2r/ckpts/model_step_82500.pt',
        'EVAL.CKPT_PATH_DIR', 'data/logs/checkpoints/release_r2r/ckpt.iter12000.pth',
        'EVAL.EPISODE_COUNT', '-1', 'IL.back_algo', 'control', 'EVAL.SAVE_TRAJ', 'True',
        'TASK_CONFIG.DATASET.DATA_PATH', V13 + '_BERTidx/{split}/{split}_bertidx.json.gz',
        'TASK_CONFIG.TASK.NDTW.GT_PATH', V13 + '/{split}/{split}_gt.json.gz',
        'TASK_CONFIG.TASK.SDTW.GT_PATH', V13 + '/{split}/{split}_gt.json.gz']


def _config(params):
    root = Path(params['adapter'].get('root', OPEN / 'ETPNav')).resolve()
    os.chdir(str(root))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import habitat_extensions  # noqa: F401 -- registry
    import vlnce_baselines  # noqa: F401 -- registry
    from vlnce_baselines.config.default import get_config
    cfg = get_config(str(root / params['adapter'].get('config', 'run_r2r/iter_train.yaml')),
                     OPTS + params['adapter'].get('opts', []))
    cfg.defrost()
    cfg.EVAL.SPLIT = params['split']
    cfg.TASK_CONFIG.DATASET.SPLIT = params['split']
    cfg.local_rank = 0
    cfg.freeze()
    return cfg


def _maps(params, cfg, attach):
    enabled = list(cfg.RL.POLICY.OBS_TRANSFORMS.ENABLED_TRANSFORMS)
    # The native v13 eval has identity CenterCropper only. A changed crop/FOV
    # must be represented explicitly rather than silently masking wrong rays.
    if any(x != 'CenterCropperPerSensor' for x in enabled):
        raise ValueError('ETPNav rig requires native identity CenterCropper transforms')
    if enabled:
        crops = dict(cfg.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS)
        sim = cfg.TASK_CONFIG.SIMULATOR
        for key in sim.AGENT_0.SENSORS:
            s = getattr(sim, key)
            if s.TYPE in ('HabitatSimRGBSensor', 'HabitatSimDepthSensor'):
                uuid = getattr(s, 'UUID', 'rgb' if s.TYPE == 'HabitatSimRGBSensor' else 'depth')
                if tuple(crops[uuid]) != (s.HEIGHT, s.WIDTH):
                    raise ValueError('nonidentity crop for %s is not the native rig' % uuid)
    return native_rig('etpnav', cfg, params['depth_bias'], params['depth_max'], attach=attach)


def configuration(params, attach=True):
    """Inspect exactly the panorama that BaseILTrainer.eval creates."""
    cfg = _config(params)
    from vlnce_baselines.utils import get_camera_orientations12
    cfg.defrost()
    for kind in ('RGB', 'DEPTH'):
        sim = cfg.TASK_CONFIG.SIMULATOR
        for angle, ori in get_camera_orientations12().items():
            key = '%s_%s' % (kind, angle)
            sensor = copy.deepcopy(getattr(sim, kind + '_SENSOR'))
            sensor.ORIENTATION = ori
            sensor.UUID = key.lower()
            setattr(sim, key, sensor)
            sim.AGENT_0.SENSORS.append(key)
            for block, field in [('RESIZER_PER_SENSOR', 'SIZES'),
                                 ('CENTER_CROPPER_PER_SENSOR', 'SENSOR_CROPS')]:
                obj = getattr(cfg.RL.POLICY.OBS_TRANSFORMS, block)
                values = list(getattr(obj, field))
                values.append((key.lower(), dict(values)[kind.lower()]))
                setattr(obj, field, values)
    maps = _maps(params, cfg, attach)
    cfg.freeze()
    return cfg, maps


def same_choice(a, b):
    return (a['act'] == b['act'] and
            np.allclose(a['target_pos'], b['target_pos'], atol=1e-6, rtol=0.))


def reproduction(reference, actual, orig, branch, start):
    a, b = reference['decisions'], actual['decisions']
    prefix = len(b) >= start - 1 and all(same_choice(x, y) for x, y in zip(a[:start - 1], b[:start - 1]))
    choices = len(a) == len(b) and all(same_choice(x, y) for x, y in zip(a, b))
    p, q = np.asarray(reference['positions']), np.asarray(actual['positions'])
    positions = p.shape == q.shape and np.allclose(p, q, atol=1e-6, rtol=0.)
    metrics = all(k in branch and np.allclose(branch[k], v, atol=1e-6, rtol=0.)
                  for k, v in orig.items() if isinstance(v, (int, float)))
    return dict(prefix_same=bool(prefix), choices_same=bool(choices),
                positions_same=bool(positions), metrics_same=bool(metrics),
                full_same=bool(choices and positions and metrics))


def termination(decisions, max_len):
    """Native act=0 also includes forced graph/length termination."""
    last = decisions[-1]
    if last['model_stop']:
        reason, truncated = 'native_stop', False
    elif last['no_vp_left']:
        reason, truncated = 'no_vp_left', True
    elif len(decisions) >= max_len:
        reason, truncated = 'max_len', True
    else:
        reason, truncated = 'environment_limit', True
    return dict(reason=reason, truncated=truncated)


def planning_trajectory(traj):
    """ETP steps are planning observations; retain low-level positions separately.

    No orientation is interpolated or fabricated for native position-only logs.
    The terminal orientation is recorded by the native environment itself.
    """
    rows = [dict(step=k, position=d['position'], rotation=d['rotation'],
                 yaw=float(yaw_from_quat(d['rotation'], 'xyzw')))
            for k, d in enumerate(traj['decisions'], 1)]
    rotation = traj['decisions'][-1]['end_rotation']
    rows.append(dict(position=traj['positions'][-1], rotation=rotation,
                     yaw=float(yaw_from_quat(rotation, 'xyzw')), terminal=True))
    return rows


class PlanningEnv(object):
    def __init__(self, envs, trainer, adapter, reference):
        if envs.num_envs != 1:
            raise ValueError('ETPNav adapter requires exactly one environment')
        self._envs, self.trainer, self.adapter, self.reference = envs, trainer, adapter, reference
        self.provenance, self.pose_checked = [], 0
        self.closed = False

    def __getattr__(self, name):
        return getattr(self._envs, name)

    def close(self):
        if not self.closed:
            self.closed = True
            self._envs.close()

    def reset(self):
        obs = self._envs.reset()
        if str(self._envs.current_episodes()[0].episode_id) != self.adapter.cand['ep']:
            raise RuntimeError('native environment selected a different episode')
        self.adapter.observation_step = 1
        obs[0] = self._apply(obs[0])
        return obs

    def _apply(self, obs):
        pos, ori = self._envs.call(['get_pos_ori'])[0]
        self.adapter.current_pose = Pose(np.asarray(pos, np.float64), yaw_from_quat(ori, 'xyzw'))
        k = self.adapter.observation_step
        if k <= self.adapter.cand['step'] or self.adapter.params['arm'] == 'none':
            if k > len(self.reference['decisions']):
                raise RuntimeError('none/prefix has more planning decisions than reference')
            ref = self.reference['decisions'][k - 1]
            assert_pose(self.adapter.current_pose, (ref['position'], yaw_from_quat(ref['rotation'], 'xyzw')))
            self.pose_checked += 1
        return self.adapter.observe(obs)

    def step(self, actions):
        act, gmap = actions[0]['action'], self.trainer.gmaps[0]
        g = getattr(gmap, '_c6002_consumed', None)
        self.provenance.append(copy.deepcopy(g) if act['act'] == 4 and g and g['vp'] == act['ghost_vp'] else None)
        gmap._c6002_consumed = None
        outputs = self._envs.step(actions)
        self.adapter.observation_step += 1
        if not outputs[0][2]:
            item = list(outputs[0])
            item[0] = self._apply(item[0])
            outputs[0] = tuple(item)
        return outputs


class ETPNavAdapter(Adapter):
    def __init__(self, params, output=None):
        # Heavy policy initialization remains inside native eval().
        self.cfg = _config(params)
        self.params, self.output = params, Path(output) if output else None
        self.reference, self.orig_stats = source_logs(params.get('selection'))
        self.trainer = None

    def pose(self):
        return self.current_pose

    def episodes(self):
        return self.reference.keys()

    def run(self, cand):
        import torch
        from habitat_baselines.common.baseline_registry import baseline_registry
        from vlnce_baselines.models.graph_utils import GraphMap
        cfg = self.cfg.clone()
        cfg.defrost()
        scratch = Path(self.params['adapter']['scratch']).resolve() / cand['ep']
        scratch.mkdir(parents=True, exist_ok=True)
        cfg.RESULTS_DIR = str(scratch)
        cfg.TENSORBOARD_DIR = ''
        cfg.VIDEO_OPTION = []
        cfg.LOG_FILE = str(scratch / 'native.log')
        cfg.EVAL.EPISODE_COUNT = 1
        cfg.EVAL.fast_eval = False
        cfg.use_pbar = True
        cfg.freeze()
        random.seed(cfg.TASK_CONFIG.SEED)
        np.random.seed(cfg.TASK_CONFIG.SEED)
        torch.manual_seed(cfg.TASK_CONFIG.SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        if torch.cuda.is_available():
            torch.set_num_threads(1)
        base = baseline_registry.get_trainer(cfg.TRAINER_NAME)
        adapter = self
        reference = self.reference[cand['ep']]
        class Trainer(base):
            def collect_val_traj(self):
                base.collect_val_traj(self)
                return [cand['ep']]

            def _eval_checkpoint(self, checkpoint_path, writer, checkpoint_index=0):
                self.config.defrost()
                maps = _maps(adapter.params, self.config, True)
                self.config.SENSORS = self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS
                self.config.freeze()
                Adapter.__init__(adapter, adapter.params, self.config, maps, adapter.output)
                adapter.begin(cand)
                base._eval_checkpoint(self, checkpoint_path, writer, checkpoint_index)

            def rollout(self, mode, ml_weight=None, sample_ratio=None):
                if not isinstance(self.envs, PlanningEnv):
                    self.envs = PlanningEnv(self.envs, self, adapter, reference)
                return base.rollout(self, mode, ml_weight, sample_ratio)
        original_delete = GraphMap.delete_ghost
        def consume(gmap, vp):
            gmap._c6002_consumed = {'vp': vp, 'fronts': [int(gmap.node_stepId[f]) for f in gmap.ghost_fronts[vp]],
                                      'pred_pos': [np.asarray(p, np.float64).tolist() for p in gmap.ghost_pos[vp]],
                                      'n': int(gmap.ghost_embeds[vp][1])}
            return original_delete(gmap, vp)
        GraphMap.delete_ghost = consume
        try:
            self.trainer = Trainer(cfg)
            self.trainer.eval()
            key = next(k for k in self.trainer.stat_eps if str(k) == cand['ep'])
            branch, traj = self.trainer.stat_eps[key], self.trainer.traj_eps[key]
            checks = reproduction(reference, traj, self.orig_stats[cand['ep']], branch, cand['step'])
            checks['start_reached'] = len(traj['decisions']) >= cand['step']
            checks['pose_checked'] = self.trainer.envs.pose_checked
            if not checks['prefix_same']:
                raise RuntimeError('ETPNav pre-intervention choices do not reproduce reference')
            if self.params['arm'] == 'none' and not checks['full_same']:
                raise RuntimeError('ETPNav none run does not fully reproduce original')
            if len(traj['decisions']) < cand['step']:
                raise RuntimeError('episode ended before intervention start')
            return dict(branch=branch, trajectory=planning_trajectory(traj),
                        decisions=traj['decisions'], positions=traj['positions'], checks=checks,
                        provenance=self.trainer.envs.provenance, images=self.images,
                        blackout=self.intervention.log() if self.intervention else [],
                        termination=termination(traj['decisions'], self.trainer.max_len))
        finally:
            GraphMap.delete_ghost = original_delete
            self.close()

    def close(self):
        if self.trainer is not None and hasattr(self.trainer, 'envs'):
            self.trainer.envs.close()
        self.trainer = None         # release the policy and waypoint predictor before the next candidate loads them
