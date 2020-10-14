import argparse

import multiprocessing
from queue import Empty
import numpy as np
import time

import pommerman
from pommerman.agents import BaseAgent, SimpleAgent
from pommerman import constants

NUM_AGENTS   = 4
NUM_ACTIONS  = len(constants.Action)
NUM_CHANNELS = 18 # ??

def argmax_tiebreaking(Q):
    """Find the best action with random tiebreaking"""
    idx = np.flatnonzero(Q == np.max(Q))
    assert len(idx) > 0, str(Q)
    return np.random.choice(idx)

class MCTSNode:
    def __init__(self, p):
        "p: initial probability distribution over actions"
        # values for 6 actions
        self.Q = np.zeros(NUM_ACTIONS)
        self.W = np.zeros(NUM_ACTIONS)
        self.N = np.zeros(NUM_ACTIONS, dtype=np.uint32)
        assert p.shape == (NUM_ACTIONS,)
        self.P = p
    
    def action(self):
        U = args.mcts_c_puct * self.P * np.sqrt(np.sum(self.N)) / (1 + self.N) # update used in AlphaGo Zero
        return argmax_tiebreaking(self.Q + U)

    def update(self, action, reward):
        self.W[action] += reward
        self.N[action] += 1
        self.Q[action] += self.W[action] / self.N[action]
    
    def probs(self, temperature):
        if temperature == 0: # return deterministic policy
            p = np.zeros(NUM_ACTIONS)
            p[argmax_tiebreaking(self.N)] = 1
            return p
        else:
            Nt = self.N ** (1.0 / temperature)
            return Nt / np.sum(Nt)

class MCTSAgent(BaseAgent):
    def __init__(self, agent_id):
        super().__init__()
        self.agent_id = agent_id
        self.env = self.make_env()
        self.reset_tree()
    
    def make_env(self):
        agents = []
        for agent_id in range(NUM_AGENTS):
            if agent_id == self.agent_id:
                agents.append(self)
            else:
                agents.append(SimpleAgent())
        return pommerman.make('PommeFFACompetition-v0', agents)

    def reset_tree(self):
        self.tree = {}
    
    def search(self, root, num_iters, temperature=1.0):
        """Does `num_iters` complete tree traversals + rollouts.
        rollouts are implicits; they are done with uniform rollout policy
        without updating trace.
        """
        # remember current game state
        # root = game_state_file JSON format
        self.env._init_game_state = root
        root = str(root)

        for _ in  range(num_iters):
            # restore game state to root node
            obs = self.env.reset()
            # serialize game state
            state = str(self.env.get_json_info()) # returns complete state information

            trace = []
            done = False
            while not done:
                if state in self.tree: # node is part of constructed game tree
                    node = self.tree[state]
                    # choose actions based on Q + U
                    action = node.action()
                    trace.append((node, action))
                else: # state represents leaf node
                    # initially: uniform distribution for probs
                    probs = np.ones(NUM_ACTIONS) / NUM_ACTIONS

                    # use current rewards for values
                    rewards = self.env._get_rewards()
                    reward = rewards[self.agent_id]

                    # add new node to tree
                    self.tree[state] = MCTSNode(probs) # expansion is implicit through probs generation

                    # stop tree traversal at leaf node
                    break

                # ensure we're not called recursively
                assert self.env.training_agent == self.agent_id
                # make other agents act
                actions = self.env.act(obs)
                # add my action to the list of actions
                actions.insert(self.agent_id, action)
                # step environment forward
                obs, rewards, done, info = self.env.step(actions)
                reward = rewards[self.agent_id]

                # fetch next state
                state = str(self.env.get_json_info())
            
            # update tree nodes with rollout results
            for node, action in trace:
                node.update(action, reward)
                reward *= args.discount
            
        # after all iteration, reset env back where we were
        self.env.set_json_info() # Sets the game state as the init_game_state.
        self.env.init_game_state = None
        # return action probabilities
        return self.tree[root].probs(temperature)
    
    def rollout(self):
        # reset search tree in the beginning of each rollout
        self.reset_tree()

        # guarantees that we are not called recursively
        # and episode ends when agent dies
        self.env.training_agent = self.agent_id
        obs = self.env.reset()

        length = 0
        done = False
        while not done:
            if args.render:
                self.env.render()

            root = self.env.get_json_info()
            # perform Monte-Carlo Tree Search
            pi = self.search(root, args.mcts_iters, args.temperature)
            # sample action from probabilities
            action = np.random.choice(NUM_ACTIONS, p=pi)

            # ensure we're not called recursively
            assert self.env.training_agent == self.agent_id
            # make other agents act
            actions = self.env.act(obs)
            # insert own action
            actions.insert(self.agent_id, action)
            # step environment
            obs, rewards, done, _ = self.env.step(actions)
            assert self == self.env._agents[self.agent_id]
            length += 1
            #print(f"Agent: {self.agent_id}, Step: {length}, Actions: {[constants.Action(a).name for a in actions]}, Probs: {[round(p, 2) for p in pi]}, Rewards: {rewards}, Done {done}")
            print(f"Tree size = {len(self.tree)}")

        reward = rewards[self.agent_id]
        return length, reward, rewards
    
    def act(self, obs, action_space):
        # TODO
        assert False

def runner(id, num_episodes, fifo, _args):
    # make args accessible to MCTSAgent
    global args
    args = _args
    # make sure agents play at all positions
    agent_id = id % NUM_AGENTS
    agent = MCTSAgent(agent_id=agent_id)

    for _ in range(num_episodes):
        # do rollout
        start_time = time.time()
        length, reward, rewards = agent.rollout()
        elapsed_time = time.time() - start_time
        # add data samples to log
        fifo.put((length, reward, rewards, agent_id, elapsed_time))

def profile_runner(id, num_episodes, fifo, _args):
    import cProfile
    command = """runner(id, num_episodes, fifo, _args)"""
    cProfile.runctx(command, globals(), locals(), filename=_args.profile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile')
    parser.add_argument('--render', action="store_true", default=False)
    # runner params
    parser.add_argument('--num_episodes', type=int, default=400)
    parser.add_argument('--num_runners', type=int, default=1)
    # MCTS params
    parser.add_argument('--mcts_iters', type=int, default=10)
    parser.add_argument('--mcts_c_puct', type=float, default=1.0)
    # RL params
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--temperature', type=float, default=0)
    args = parser.parse_args()

    assert args.num_episodes % args.num_runners == 0, "The number of episodes should be divisible by number of runners"

    # use spawn method for starting subprocesses
    ctx = multiprocessing.get_context('spawn')      

    # create fifo and processes for all runners
    fifo = ctx.Queue()
    for i in range(args.num_runners):
        process = ctx.Process(target=profile_runner if args.profile else runner,
                              args=(i, args.num_episodes // args.num_runners, fifo, args))
        process.start()

    # do logging in the main case
    all_rewards = []                              
    all_lengths = []
    all_elapsed = []
    for i in range(args.num_episodes):
        # wait for new trajectory
        length, reward, rewards, agent_id, elapsed = fifo.get()
        print(f"Episode {i}: reward={reward}, length={length}, agent={agent_id}, time per step = {elapsed/length}")
        all_rewards.append(reward)
        all_lengths.append(length)        
        all_elapsed.append(elapsed)
    
    print("Average reward:", np.mean(all_rewards))
    print("Average length:", np.mean(all_lengths))
    print("Time per timestep:", np.sum(all_elapsed) / np.sum(all_lengths))