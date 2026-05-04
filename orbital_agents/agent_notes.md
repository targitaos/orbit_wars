Here are thoughts and notes for the agents

<!-- Changed to 30 steps:
changed to seed 11

  "agents": [2, 4],
  "configuration": {
    "episodeSteps": 30,
    "actTimeout": 1,

C:\Users\th-om\Projekter\orbit_wars\.venv\Lib\site-packages\kaggle_environments\envs\orbit_wars\orbit_wars.json

 -->







# Implement
* Deterministic trajectories, never miss.
* Features: Avoid sun; center at (50,50), radius 10. If trajectory intersects sun, adjust angle to be tangent to sun.
* Weigh planet by distance AND ship strengh

# Implement later
* If no attack vector, send reinforcements
* Measure incident fleets - enemy and friendly

# Learnable:
Economies of fleets; what proportion to send from planet
Order of planets to command
Weighting of target planets

# Open questions:
Does planet speed impact velocity of fleet? Assumption asof 4/5: NO
Can I see ships enroute?

# Agent ideas
Zerg rush
Networked support
