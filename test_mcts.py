from mcts import SokoMCTS
from gym_sokoban.envs.sokoban_env_fast import SokobanEnvFast
from PIL import Image

env = SokobanEnvFast(dim_room=(10, 10), num_boxes=4, seed=3)

# Show room
env.reset()
Image.fromarray(env.render(mode="rgb_array")).show()

mcts = SokoMCTS(env=env, c=1, max_depth=5000, number_of_simulations=4)
mcts.run(passes=1000, verbose=1)

graph = mcts.get_graph()
graph.view(cleanup=True)
