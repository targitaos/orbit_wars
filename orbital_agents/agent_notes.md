Here are thoughts and notes for the agents

<!-- Changed to 30 steps:
changed to seed 11

  "agents": [2, 4],
  "configuration": {
    "episodeSteps": 30,
    "actTimeout": 1,

C:\Users\th-om\Projekter\orbit_wars\.venv\Lib\site-packages\kaggle_environments\envs\orbit_wars\orbit_wars.json

Changed OvereageTime to 600s

 -->


# Implement
* DONE: Deterministic trajectories, never miss.
* DONE: Features: Avoid sun; center at (50,50), radius 10. If trajectory intersects sun, adjust angle to be tangent to sun.
* Implement Fleet ledger.
* Weigh planet by distance AND ship strengh
* Multiple attack vectors from a single planet

# Implement later
* If no attack vector, send reinforcements
* Measure incident fleets - enemy and friendly
* Weighting made dependent on my planet fleet size
* In main.py implement generall "def agent" method + specific agent class.
* Threat + Support : Send lower/higher proportion of fleet

# Learnable:
Economies of fleets; what proportion to send from planet
Order of planets to command
Weighting of target planets
Simple weightings: Ships, production, dist, and owner.

# Open questions:
When do I go from Neutral Planets to Enemy Planets?
Does planet speed impact velocity of fleet? Assumption asof 4/5: NO, asnwer 5/5: No
Can I see ships enroute? 6/5 YES

# Ideas
Zerg rush
Networked support - empire/clique building
Phase changes in games
