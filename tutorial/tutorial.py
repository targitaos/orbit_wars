import math
import os
from pathlib import Path

import kagglehub
import numpy as np
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
        print(f'--- MY PLANET: ({mine.x:.2f}, {mine.y:.2f}), ships: {mine.ships}  ---')
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
            print(f'Player {player}: Sending fleet from {mine.id} to {nearest.id}')
            print(f'  Ships: {ships_needed}, Angle: {(angle / np.pi):.2f}π')
            print(
                f'  Mine pos: ({mine.x:.2f}, {mine.y:.2f}), Target pos: ({nearest.x:.2f}, {nearest.y:.2f})'
            )

    return moves


if __name__ == '__main__':
    # Test it against the random agent
    env = make('orbit_wars', debug=True)
    env.run([nearest_planet_sniper, 'random'])

    final = env.steps[-1]
    for i, s in enumerate(final):
        print(f'Player {i}: reward={s.reward}, status={s.status}')

    html = env.render(mode='html', width=800, height=600)
    with Path.open('replay.html', 'w') as f:
        f.write(html)
