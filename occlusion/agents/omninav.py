"""Native OmniNav closed-loop adapter (run in the omninav environment)."""
import os
import sys
from pathlib import Path

import numpy as np

from .base import Adapter, assert_pose, literal, native_rig
from ..select.common import OPEN, observation_pose, original_metrics, primary


def configuration(params, attach=True):
    options = params['adapter']
    root = Path(options.get('root', OPEN / 'OmniNav/infer_r2r_rxr')).resolve()
    os.chdir(str(root))
    sys.path.insert(0, str(root))
    from VLN_CE.vlnce_baselines.config.default import get_config
    cfg = get_config(str(root / options.get('config', 'data/r2r_pano.yaml')), options.get('opts', []))
    cfg.defrost()
    cfg.TASK_CONFIG.DATASET.SPLIT = params['split']
    size = literal(root / 'agent/waypoint_agent.py', 'INPUT_IMG_SIZE')
    sizes = {getattr(cfg.TASK_CONFIG.SIMULATOR, key).UUID: (size[1], size[0])
             for key in cfg.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS
             if getattr(cfg.TASK_CONFIG.SIMULATOR, key).TYPE == 'HabitatSimRGBSensor'}
    maps = native_rig('omninav', cfg, params['depth_bias'], params['depth_max'], sizes, attach)
    cfg.freeze()
    return cfg, maps


def check_action(expected, actual, arrive):
    """Archive actions/r/theta are rounded to four decimals by native rollout."""
    name = 'stop' if actual['action'] == 'STOP' else 'go_toward_point'
    if expected['action'] != name or int(expected['arrive']) != int(arrive):
        raise RuntimeError('OmniNav prefix action/arrive mismatch')
    if name != 'stop':
        for key in ('r', 'theta'):
            if not np.isclose(float(expected[key]), actual['action_args'][key], atol=5.1e-5, rtol=0.):
                raise RuntimeError('OmniNav prefix %s mismatch' % key)


class OmniNavAdapter(Adapter):
    def __init__(self, params, output=None):
        cfg, maps = configuration(params)
        from habitat import Env
        from habitat.datasets import make_dataset
        from agent.waypoint_agent import Waypoint_Agent
        dataset = make_dataset(id_dataset=cfg.TASK_CONFIG.DATASET.TYPE, config=cfg.TASK_CONFIG.DATASET)
        self.episode_map = {str(e.episode_id): e for e in dataset.episodes}
        self.env = Env(cfg.TASK_CONFIG, dataset)
        Adapter.__init__(self, params, cfg, maps, output)
        scratch = Path(params['adapter']['scratch']).resolve()
        scratch.mkdir(parents=True, exist_ok=True)
        self.agent = Waypoint_Agent(params['adapter']['model_path'], str(scratch))

    HISTORY_ORDER = ('left', 'right', 'front')    # the order Waypoint_Agent.act passes the current views to add_frame

    def keep_clean_history(self):
        """Put the real current views back into the model's history after `act` has stored the blacked ones."""
        if self.intervention is None or not self.intervention.active(self.observation_step):
            return
        from PIL import Image
        n = len(self.HISTORY_ORDER)
        if len(self.agent.rgb_list) < n:
            raise RuntimeError('agent history has fewer than %d frames' % n)
        self.agent.rgb_list[-n:] = [Image.fromarray(self.clean_rgb[v]) for v in self.HISTORY_ORDER]

    def reset(self, ep):
        # Env.reset() takes next(episode_iterator) and ignores an assigned env.current_episode.
        self.env.episode_iterator = iter([self.episode_map[str(ep)]])
        self.obs = self.env.reset()
        if str(self.env.current_episode.episode_id) != str(ep):
            raise RuntimeError('environment reset to episode %s, expected %s' %
                               (self.env.current_episode.episode_id, ep))
        self.agent.reset()
        self.last_dtg, self.still = 999, 0
        return self.obs

    def step(self):
        import torch
        info = self.metrics()
        if info['distance_to_goal'] != self.last_dtg:
            self.last_dtg, self.still = info['distance_to_goal'], 0
        else:
            self.still += 1
        obs = self.observe(self.obs)
        q = self.env._sim.get_agent_state().rotation
        obs['pose'] = {'position': self.pose().position.tolist(), 'rotation': [q.w, q.x, q.y, q.z]}
        with torch.no_grad():
            pred = self.agent.act(obs, info, self.cand['ep'])
        if self.params['adapter']['history'] == 'clean':
            self.keep_clean_history()
        action = ({'action': 'STOP'} if pred['arrive_pred'] > 0 else
                  {'action': 'GO_TOWARD_POINT', 'action_args':
                   {'theta': float(pred['recover_angle'][0]), 'r': float(np.linalg.norm(pred['action'][0]))}})
        if self.still > self.native.EVAL.EARLY_STOP_ROTATION or self.observation_step - 1 > self.native.EVAL.EARLY_STOP_STEPS:
            action = {'action': 'STOP'}
        if self.observation_step < self.cand['step'] or self.params['arm'] == 'none':
            if self.observation_step > len(self.original_steps):
                raise RuntimeError('OmniNav has more prefix/control actions than reference')
            check_action(self.original_steps[self.observation_step - 1], action, pred['arrive_pred'])
            self.action_checked += 1
        self.stopped = ('early_stop' if action['action'] == 'STOP' and pred['arrive_pred'] <= 0
                        else 'arrive' if action['action'] == 'STOP' else 'episode_over')
        p = self.pose()
        self.trajectory.append({'step': self.observation_step, 'position': p.position.tolist(),
                                'yaw': p.yaw, 'action': action})
        self.obs = self.env.step(action)
        return action

    def run(self, cand):
        self.begin(cand)
        rows = primary('omninav', cand['split'], cand['ep'])
        self.reset(cand['ep'])
        self.original_steps = [r for r in rows if r.get('event') == 'step']
        self.action_checked = 0
        self.stopped = 'episode_over'
        checks = {'pose_checked': 0, 'start_reached': False}
        while not self.env.episode_over:
            self.observation_step += 1
            if self.observation_step <= cand['step'] or self.params['arm'] == 'none':
                # Original OmniNav poses were archived as pos=4dp, yaw=2dp.
                assert_pose(self.pose(), observation_pose(rows, self.observation_step),
                            tolerance=1e-4, angle_tolerance=.0051)
                checks['pose_checked'] += 1
            if self.observation_step == cand['step']:
                checks['start_reached'] = True
            self.step()
        if not checks['start_reached']:
            raise RuntimeError('episode ended before intervention start')
        p = self.pose()
        if self.params['arm'] == 'none':
            if self.observation_step != len(self.original_steps):
                raise RuntimeError('OmniNav control action count mismatch')
            # STOP has no archived pos; its observation pose is the terminal pose.
            terminal = self.original_steps[-1]
            expected = ((terminal['pos'], terminal['yaw']) if 'pos' in terminal else
                        observation_pose(rows, self.observation_step))
            assert_pose(p, expected, tolerance=1e-4, angle_tolerance=.0051)
            orig, branch = original_metrics(rows), self.metrics()
            for key, value in orig.items():
                if isinstance(value, (int, float)) and (key not in branch or not
                        np.isclose(value, branch[key], atol=1e-6, rtol=0.)):
                    raise RuntimeError('OmniNav control native metric mismatch: %s' % key)
            checks['full_same'] = True
        checks['action_checked'] = self.action_checked
        self.trajectory.append({'position': p.position.tolist(), 'yaw': p.yaw, 'terminal': True})
        stopped = self.trajectory[-2]['action']['action'] == 'STOP'
        return {'branch': self.metrics(), 'trajectory': self.trajectory,
                'blackout': self.intervention.log() if self.intervention else [],
                'images': self.images, 'checks': checks,
                'termination': {'reason': self.stopped, 'truncated': not stopped or self.stopped == 'early_stop'}}
