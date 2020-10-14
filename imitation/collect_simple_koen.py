import pommerman
from pommerman import agents
import numpy as np
import argparse
from copy import deepcopy

NUM_AGENTS = 4

def featurize(obs):
    board = obs['board']

    # convert board items into bitmaps
    maps = [board == i for i in range(10)] # several bitplanes True/False for different items on board (0...10)
    maps.append(obs['bomb_blast_strength'])
    maps.append(obs['bomb_life'])

    # duplicate ammo, blast_strength and can_kick over entire map
    # np.full(shape, fill_value): Return a new array of given shape and type, filled with `fill_value`.
    maps.append(np.full(board.shape, obs['ammo']))
    maps.append(np.full(board.shape, obs['blast_strength']))
    maps.append(np.full(board.shape, obs['can_kick']))

    # add my position as bitmap
    position = np.zeros(board.shape)
    position[obs['position']] = 1
    maps.append(position)

    # add teammate
    if obs['teammate'] is not None:
        maps.append(board == obs['teammate'].value)
    else:
        maps.append(np.zeros(board.shape))
    
    # add enemies
    enemies = [board == e.value for e in obs['enemies']]
    maps.append(np.any(enemies, axis=0))

    return np.stack(maps, axis=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int, default=2)
    parser.add_argument('--render', action="store_true", default=False)
    parser.add_argument('--out_file', default='temp.file')
    args = parser.parse_args()

    # Create a set of NUM_AGENTS agents
    agent_list = [agents.SimpleAgent() for _ in range(NUM_AGENTS)]

    # Make the 'Free-For-All' environment using the agent list
    env = pommerman.make('PommeFFACompetition-v0', agent_list)

    observations = [[] for _ in range(NUM_AGENTS)]
    actions      = [[] for _ in range(NUM_AGENTS)]
    rewards      = [[] for _ in range(NUM_AGENTS)]

    # Run the episodes, just like OpenAI Gym
    for i in range(args.num_episodes):
        obs = env.reset()
        done = False
        reward = [0, 0, 0, 0]
        t = 0
        while not done:
            if args.render:
                env.render()
            action = env.act(obs)
            new_obs, new_reward, done, info = env.step(action)
            for j in range(NUM_AGENTS):
                if reward[j] == 0:
                    observations[j].append(featurize(obs[j]))
                    actions[j].append(action[j])
                    rewards[j].append(new_reward[j])
            obs = deepcopy(new_obs)
            reward = deepcopy(new_reward)
            t += 1
        print(f"Episode {i+1}: length: {t}, rewards: {reward}")
    env.close()

    np.savez(args.out_file,
             observations=sum(observations, []), # flattens the list
             actions=sum(actions, []),
             rewards=sum(rewards, []))