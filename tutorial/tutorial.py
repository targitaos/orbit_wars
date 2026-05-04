import math
import os
from pathlib import Path

import kagglehub
from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet


def nearest_planet_sniper(obs: dict) -> list:
    moves = []
    player = obs.get('player', 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get('planets', []) if isinstance(obs, dict) else obs.planets
    planets = [Planet(*p) for p in raw_planets]

    # Separate our planets from targets
    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]

    if not targets:
        return moves

    for mine in my_planets:
        # Find the nearest planet we don't own
        nearest = None
        min_dist = float('inf')
        for t in targets:
            dist = math.sqrt((mine.x - t.x) ** 2 + (mine.y - t.y) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest = t

        if nearest is None:
            continue

        # How many ships do we need? Target's garrison + 1
        ships_needed = max(nearest.ships + 1, 20)

        # Only send if we have enough
        if mine.ships >= ships_needed:
            # Calculate angle from our planet to the target
            angle = math.atan2(nearest.y - mine.y, nearest.x - mine.x)
            moves.append([mine.id, angle, ships_needed])

    return moves


if __name__ == '__main__':
    # kagglehub.login(api_token=os.environ['KAGGLE_API_TOKEN'])

    # Download latest version
    # path = kagglehub.competition_download('orbit-wars')

    # print('Path to competition files:', path)

    # env = make('orbit_wars', debug=True)
    # print('----------------------')
    # print(f'Environment: {env.name} v{env.version}')
    # print('----------------------')
    # print(f'Players: {env.specification.agents}')
    # print('----------------------')
    # print(f'Max steps: {env.configuration.episodeSteps}')

    # Run a quick game to see what the observation looks like
    # env = make('orbit_wars', debug=True)
    # env.run(['random', 'random'])

    # # Peek at the initial observation

    # obs = env.steps[1][0].observation  # step 1 = first action step
    # planets = [Planet(*p) for p in obs.planets]

    # print(f'Player: {obs.player}')
    # print(f'Angular velocity: {obs.angular_velocity:.4f} rad/turn')
    # print(f'\nPlanets ({len(planets)}):')

    # for p in planets[:6]:
    #     owner_str = f'Player {p.owner}' if p.owner >= 0 else 'Neutral'
    #     print(
    #         f'  id={p.id} owner={owner_str:10s} pos=({p.x:.1f}, {p.y:.1f}) r={p.radius:.1f} ships={p.ships} prod={p.production}'
    #     )

    # Test it against the random agent
    env = make('orbit_wars', debug=True)
    env.run([nearest_planet_sniper, 'random'])

    final = env.steps[-1]
    for i, s in enumerate(final):
        print(f'Player {i}: reward={s.reward}, status={s.status}')

    html = env.render(mode='html', width=800, height=600)
    with Path.open('replay.html', 'w') as f:
        f.write(html)
