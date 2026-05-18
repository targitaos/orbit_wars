import math

import gymnasium as gym
import numpy as np
from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from orbital_agents.agent_methods import fleet_speed, trajectory_calculation, trajectory_crosses_sun

SOLAR_X = 50
SOLAR_Y = 50
COMET_MARKER = -99

MAX_PLANETS = 30  # orbit_wars can spawn up to ~26 planets; 30 gives headroom
_PLANET_FEATURES = 7  # [is_mine, is_enemy, ships, production, x, y, dist_from_sun]
_GLOBAL_FEATURES = 6  # [step, my_ships, enemy_ships, my_count, enemy_count, neutral_count]
OBS_SIZE = _GLOBAL_FEATURES + MAX_PLANETS * _PLANET_FEATURES

# Action indices
W_DIST = 0
W_ALLEGIANCE = 1
W_SHIPS = 2
W_PRODUCTION = 3
SEND_RATIO = 4


class OrbitWarsEnv(gym.Env):
    """Gym wrapper around the kaggle orbit_wars environment.

    Action space (Box, 5 dims):
        [w_dist, w_allegiance, w_ships, w_production, send_ratio]
        - w_dist: how much to prefer closer targets (higher = prefer near)
        - w_allegiance: how much to prefer enemy-owned targets vs neutral
        - w_ships: how much to prefer lightly defended targets (higher = prefer weak)
        - w_production: how much to prefer high-production targets (higher = prefer rich)
        - send_ratio: fraction of a planet's ships to commit per attack (0.1-0.9)

    Observation space (Box, OBS_SIZE dims):
        Global game state + per-planet features, all normalized to [0, 1].
    """

    metadata = {'render_modes': ()}

    def __init__(self, opponent_agent=None, episode_steps: int = 30):
        super().__init__()
        self.episode_steps = episode_steps
        self.opponent_agent = opponent_agent  # callable(obs) -> moves, or None for no-op
        self._fleet_ledger: dict[tuple[int, int], int] = {}
        self._player = 0
        self._kaggle_env = None
        self._current_obs = None

        self.action_space = gym.spaces.Box(
            low=np.array([0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32),
            high=np.array([5.0, 5.0, 5.0, 5.0, 0.9], dtype=np.float32),
        )  # fmt: skip
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBS_SIZE,),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._fleet_ledger.clear()
        self._kaggle_env = make(
            'orbit_wars',
            debug=False,
            configuration={'episodeSteps': self.episode_steps},
        )
        self._kaggle_env.reset()
        self._current_obs = self._kaggle_env.state[self._player].observation
        return self._extract_obs(self._current_obs), {}

    def step(self, action: np.ndarray):
        w_dist, w_allegiance, w_ships, w_prod, send_ratio = action.tolist()

        p0_moves = self._compute_moves(
            self._current_obs, w_dist, w_allegiance, w_ships, w_prod, send_ratio
        )

        opp_obs = self._kaggle_env.state[1].observation
        opp_obs.step = self._current_obs.step  # kaggle only injects step into player 0's obs
        p1_moves = self.opponent_agent(opp_obs) if self.opponent_agent is not None else []

        self._kaggle_env.step([p0_moves, p1_moves])
        self._current_obs = self._kaggle_env.state[self._player].observation

        obs = self._extract_obs(self._current_obs)
        reward = self._compute_reward()
        done = self._kaggle_env.done
        return obs, reward, done, False, {}

    # ------------------------------------------------------------------
    # Agent logic (parameterised heuristic)
    # ------------------------------------------------------------------

    def _compute_moves(self, obs, w_dist, w_allegiance, w_ships, w_prod, send_ratio) -> list:
        step = obs.step
        player = obs.player if hasattr(obs, 'player') else self._player

        if step == 0:
            self._fleet_ledger.clear()
        else:
            stale = [k for k, cutoff in self._fleet_ledger.items() if cutoff <= step]
            for k in stale:
                del self._fleet_ledger[k]

        planets = [Planet(*p) for p in obs.planets]
        my_planets = [p for p in planets if p.owner == player]

        comet_ids = {
            planets[i].id
            for i, ip in enumerate(obs.initial_planets)
            if ip[2] == COMET_MARKER and ip[3] == COMET_MARKER
        }
        targets = [p for p in planets if p.owner != player and p.id not in comet_ids]
        if not targets:
            return []

        # Ships needed to capture a target (neutral: just outnumber; enemy: min 15).
        def required(t: Planet) -> int:
            return t.ships + 1 if t.owner == -1 else max(t.ships + 1, 15)

        committed: dict[int, int] = {}  # target id -> ships allocated by this turn's fleets

        moves = []
        for sender in my_planets:
            # Prefer targets this turn's other fleets haven't already covered, so
            # planets spread out. A target stays selectable while still
            # under-committed, which keeps deliberate combined attacks possible.
            available = [t for t in targets if committed.get(t.id, 0) < required(t)]
            if not available:
                available = targets

            best = max(
                available,
                key=lambda t: self._score(sender, t, w_dist, w_allegiance, w_ships, w_prod),
            )

            if step < self._fleet_ledger.get((sender.id, best.id), 0):
                continue

            needed = max(0, required(best) - committed.get(best.id, 0))
            ships = max(needed, int(sender.ships * send_ratio))

            if sender.ships < ships:
                continue

            resolved = self._resolve_launch(sender, best, ships, obs.angular_velocity)
            if resolved is None:
                continue
            angle, delta_t = resolved

            self._fleet_ledger[(sender.id, best.id)] = step + math.ceil(delta_t)
            committed[best.id] = committed.get(best.id, 0) + ships
            moves.append([sender.id, angle, ships])

        return moves

    def _resolve_launch(
        self,
        sender: Planet,
        target: Planet,
        ships: int,
        angular_velocity: float,
    ) -> tuple[float, float] | None:
        """Return (angle, delta_t) for a valid fleet, or None if it can't be launched."""
        v = fleet_speed(ships)
        angle, delta_t = trajectory_calculation(
            sender,
            target,
            v=v,
            angular_speed=angular_velocity,
        )
        if angle is None:
            return None

        ix = sender.x + v * delta_t * math.cos(angle)
        iy = sender.y + v * delta_t * math.sin(angle)
        if trajectory_crosses_sun(sender.x, sender.y, ix, iy):
            return None

        return angle, delta_t

    def _score(
        self, sender: Planet, target: Planet, w_dist, w_allegiance, w_ships, w_prod
    ) -> float:
        dist = math.hypot(sender.x - target.x, sender.y - target.y)
        dist_factor = 1.0 / (1.0 + dist * w_dist * 0.1)
        # Neutral targets are the baseline; w_allegiance scales how much
        # enemy-owned targets are preferred (>1) or avoided (<1) relative to them.
        allegiance_factor = 1.0 if target.owner == -1 else w_allegiance
        ship_factor = 1.0 / (1.0 + target.ships * w_ships * 0.01)
        prod_factor = 1.0 + target.production * w_prod * 0.1
        return dist_factor * ship_factor * prod_factor * allegiance_factor

    # ------------------------------------------------------------------
    # Observation & reward
    # ------------------------------------------------------------------

    def _extract_obs(self, obs) -> np.ndarray:
        planets = [Planet(*p) for p in obs.planets]
        player = obs.player if hasattr(obs, 'player') else self._player

        my_ships = sum(p.ships for p in planets if p.owner == player)
        enemy_ships = sum(p.ships for p in planets if p.owner not in {-1, player})
        my_count = sum(1 for p in planets if p.owner == player)
        enemy_count = sum(1 for p in planets if p.owner not in {-1, player})
        neutral_count = sum(1 for p in planets if p.owner == -1)

        global_feat = np.array(
            [
                obs.step / self.episode_steps,
                min(my_ships / 500.0, 1.0),
                min(enemy_ships / 500.0, 1.0),
                min(my_count / MAX_PLANETS, 1.0),
                min(enemy_count / MAX_PLANETS, 1.0),
                min(neutral_count / MAX_PLANETS, 1.0),
            ],
            dtype=np.float32,
        )

        planet_feat = np.zeros(MAX_PLANETS * _PLANET_FEATURES, dtype=np.float32)
        for i, p in enumerate(planets[:MAX_PLANETS]):
            base = i * _PLANET_FEATURES
            planet_feat[base + 0] = 1.0 if p.owner == player else 0.0
            planet_feat[base + 1] = 1.0 if p.owner not in {-1, player} else 0.0
            planet_feat[base + 2] = min(p.ships / 200.0, 1.0)
            planet_feat[base + 3] = min(p.production / 10.0, 1.0)
            planet_feat[base + 4] = p.x / 100.0
            planet_feat[base + 5] = p.y / 100.0
            planet_feat[base + 6] = min(math.hypot(p.x - SOLAR_X, p.y - SOLAR_Y) / 100.0, 1.0)

        return np.concatenate([global_feat, planet_feat])

    def _compute_reward(self) -> float:
        player_state = self._kaggle_env.state[self._player]

        if self._kaggle_env.done:
            # Final reward: +1 win, -1 loss, 0 draw
            r = player_state.reward
            if r == 1:
                return 1.0
            if r == 0:
                return -1.0
            return 0.0

        # Intermediate shaping: relative ship advantage, small scale
        obs = player_state.observation
        planets = [Planet(*p) for p in obs.planets]
        player = self._player
        my_ships = sum(p.ships for p in planets if p.owner == player)
        enemy_ships = sum(p.ships for p in planets if p.owner not in {-1, player})
        total = my_ships + enemy_ships + 1
        return (my_ships - enemy_ships) / total * 0.01
