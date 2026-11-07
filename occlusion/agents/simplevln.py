"""SimpleVLN: replay archived actions without sampling, then generate once."""
import os
import sys
from pathlib import Path

from .base import Adapter, assert_pose, load_module, native_rig
from ..select.common import OPEN, observation_pose, primary
from ..select.simplevln import ACTIONS


def patch_ndtw(measure):
    """SimpleVLN's NDTW looks the GT up as gt_json[episode.episode_id]. R2R episode ids are ints while the GT json
    keys are strings, so env.reset() fails with KeyError. Accept either form; nothing else changes."""
    if getattr(measure, '_c6002_str_keys', False):
        return measure

    def reset_metric(self, *args, episode, **kwargs):
        key = episode.episode_id
        self.locations = []
        self.gt_locations = self.gt_json[key if key in self.gt_json else str(key)]['locations']
        self.update_metric()

    measure.reset_metric = reset_metric
    measure._c6002_str_keys = True
    return measure


def configuration(params, attach=True):
    options = params['adapter']
    root = Path(options.get('root', OPEN / 'SimpleVLN/eval')).resolve()
    os.chdir(str(root))
    os.environ.setdefault('VLN_DATA_ROOT', str(root / 'data/vln_data'))
    sys.path.insert(0, str(root / 'vln'))
    module = sys.modules.get('c6002_simplevln_eval') or load_module(root / 'vln/eval.py', 'c6002_simplevln_eval')
    from habitat_extensions.measures import NDTW
    patch_ndtw(NDTW)
    cfg = module.get_config(str(root / options.get('config', 'config/vln_r2r.yaml')))
    from omegaconf import open_dict, read_write
    with read_write(cfg), open_dict(cfg):
        cfg.habitat.dataset.split = params['split']
        # Use the model's own registered nDTW measure, not a substitute metric.
        if 'ndtw' not in cfg.habitat.task.measurements:
            from habitat_extensions.measures import NDTWMeasurementConfig
            if not options.get('gt_path'):
                raise ValueError('SimpleVLN adapter.gt_path is required for native nDTW')
            cfg.habitat.task.measurements.ndtw = NDTWMeasurementConfig(gt_path=options['gt_path'])
        maps = native_rig('simplevln', cfg, params['depth_bias'], params['depth_max'], attach=attach)
    return (module, cfg), maps


class SimpleVLNAdapter(Adapter):
    def __init__(self, params, output=None):
        native, maps = configuration(params)
        module, cfg = native
        module.seed_all()
        dataset = module.habitat.datasets.make_dataset(id_dataset=cfg.habitat.dataset.type, config=cfg.habitat.dataset)
        dataset.config = cfg.habitat.dataset
        self.dataset = dataset
        self.episode_map = {str(e.episode_id): e for e in dataset.episodes}
        self.env = module.Env(cfg.habitat, dataset)
        Adapter.__init__(self, params, native, maps, output)
        options = params['adapter']
        self.agent = module.SimpleVLN_Agent(options['base_url'], options['model_name'],
                                           str(Path(options['scratch']).resolve()),
                                           float(cfg.habitat.simulator.forward_step_size) * 100.,
                                           float(cfg.habitat.simulator.turn_angle),
                                           int(options['max_history']), 1.0, save_video=False)

    def original_metrics(self, cand):
        """Run the model's native measure on the recorded original positions."""
        from types import SimpleNamespace
        from habitat_extensions.measures import NDTW
        from ..select.common import original_metrics
        from ..core import store
        rows = primary('simplevln', cand['split'], cand['ep'])
        result = original_metrics(rows)
        if 'ndtw' not in result:
            positions = store.trajectory_from('simplevln', cand['split'], cand['ep'], 1)
            if not len(positions):
                raise ValueError('original trajectory is empty')
            state = SimpleNamespace(position=positions[0])
            sim = SimpleNamespace(get_agent_state=lambda: state)
            measure = NDTW(sim=sim, config=self.native[1].habitat.task.measurements.ndtw,
                           dataset=self.dataset)
            measure.reset_metric(episode=self.episode_map[str(cand['ep'])])
            for position in positions[1:]:
                state.position = position
                measure.update_metric()
            result['ndtw'] = float(measure.get_metric())
            result['ndtw_source'] = 'native habitat_extensions.measures.NDTW / primary rollout'
        return result

    def reset(self, ep):
        self.env.current_episode = self.episode_map[str(ep)]
        self.obs = self.env.reset()
        if str(self.env.current_episode.episode_id) != str(ep):
            raise RuntimeError('environment reset to episode %s, expected %s' %
                               (self.env.current_episode.episode_id, ep))
        self.agent.reset()
        self.last_dtg, self.still = 999, 0
        return self.obs

    def step(self):
        from PIL import Image
        info = self.metrics()
        if info['distance_to_goal'] != self.last_dtg:
            self.last_dtg, self.still = info['distance_to_goal'], 0
        else:
            self.still += 1
        k = self.observation_step
        clean = Image.fromarray(self.obs['rgb'].astype('uint8')).convert('RGB')
        if k < self.cand['step']:
            # No act(), API call, random fallback or original pending plan before start.
            self.agent.rgb_list.append(clean)
            self.agent.rgb_list = self.agent.rgb_list[-self.agent.max_action_history:]
            action = {'action': ACTIONS.get(self.original_steps[k - 1]['action'], self.original_steps[k - 1]['action'])}
            source = 'replay'
        else:
            if k == self.cand['step']:
                assert_pose(self.pose(), observation_pose(self.original_rows, k))
                self.agent.pending_action_list = []
            action = self.agent.act(self.observe(self.obs), info, self.cand['ep'])
            source = 'model' if self.agent.last_call is not None else 'pending'
            # Keep actual clean observations as history, including after blacked calls.
            self.agent.rgb_list[-1] = clean
            # Same stop conditions as native evaluate_agent (zero-based iter_step).
            if self.still > 25 or k - 1 > 400:
                action, source = {'action': 0}, 'forced_stop'
        p = self.pose()
        row = {'step': k, 'position': p.position.tolist(), 'yaw': p.yaw,
               'action': action['action'], 'source': source}
        if source == 'model':
            row['call'] = self.agent.last_call
        self.trajectory.append(row)
        self.obs = self.env.step(action)
        return action

    def run(self, cand):
        self.begin(cand)
        self.original_rows = primary('simplevln', cand['split'], cand['ep'])
        self.original_steps = [r for r in self.original_rows if r.get('event') == 'step']
        self.reset(cand['ep'])
        reached = False
        while not self.env.episode_over:
            self.observation_step += 1
            if self.observation_step <= cand['step']:
                assert_pose(self.pose(), observation_pose(self.original_rows, self.observation_step))
            if self.observation_step == cand['step']:
                reached = True
            self.step()
        if not reached:
            raise RuntimeError('episode ended before intervention start')
        p = self.pose()
        self.trajectory.append({'position': p.position.tolist(), 'yaw': p.yaw, 'terminal': True})
        return {'branch': self.metrics(), 'trajectory': self.trajectory,
                'blackout': self.intervention.log() if self.intervention else [], 'images': self.images,
                'checks': {'replay_pose_same': True, 'start_reached': True, 'sample_repetitions': 1},
                'termination': {'reason': self.trajectory[-2]['source'],
                                'truncated': (self.trajectory[-2]['action'] != 0 or
                                              self.trajectory[-2]['source'] == 'forced_stop')}}
