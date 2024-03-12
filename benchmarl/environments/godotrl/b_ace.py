#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from typing import Callable, Dict, List, Optional

from benchmarl.environments.common import Task
from benchmarl.utils import DEVICE_TYPING

from tensordict import TensorDictBase
from torchrl.data import CompositeSpec
from torchrl.envs import EnvBase
from torchrl.data import CompositeSpec, BoundedTensorSpec, DiscreteTensorSpec
import torch
# from torchrl.envs.libs import YourTorchRLEnvConstructor

# from pettingzoo.classic import tictactoe_v3
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.envs.utils import MarlGroupMapType
from GodotRLPettingZooWrapper import GodotRLPettingZooWrapper
import random
                                 

class B_ACE(Task):
    # Your task names.
    # Their config will be loaded from benchmarl/conf/task/customenv

    b_ace = None  # Loaded automatically from benchmarl/conf/task/customenv/task_1
    #TASK_2 = None  # Loaded automatically from benchmarl/conf/task/customenv/task_2    

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
         return lambda: PettingZooWrapper(
                                env=GodotRLPettingZooWrapper(
                                    #env_path=self.config.pop("BVR_AirCombat/bin/BVR_1x1_FullView.exe", "BVR_AirCombat/bin/BVR_1x1_FullView.exe"), 
                                    #num_agents = 1, 
                                    #show_window=True, 
                                    #seed = seed,
                                    #port = GodotRLPettingZooWrapper.DEFAULT_PORT + random.randint(0,3000),
                                    #framerate = None,
                                    #action_repeat = 20,
                                    #action_type = "Low_Level_Continuous",#"Low_Level_Continuous"
                                    #speedup  = 100,
                                    convert_action_space = False,
                                    **self.config), 
                                    
                                    # scenario=self.name.lower(),
                                    # num_envs=2,#num_envs,  # Number of vectorized envs (do not use this param if the env is not vectorized)
                                    # num_envs=2,#num_envs,  # Number of vectorized envs (do not use this param if the env is not vectorized)
                                    # continuous_actions=continuous_actions,#continuous_actions,  # Ignore this param if your env does not have this choice                                    
                                    use_mask=True, # Must use it since one player plays at a time
                                    # seed=seed,
                                    # device=device,
                                    #categorical_actions=True,  # If your env has discrete actions, they need to be categorical (TorchRL can help with this)
                        ) 
     
    def supports_continuous_actions(self) -> bool:
        # Does the environment support continuous actions?
        return False

    def supports_discrete_actions(self) -> bool:
        # Does the environment support discrete actions?
        return True

    def has_render(self, env: EnvBase) -> bool:
        # Does the env have a env.render(mode="rgb_array") or env.render() function?
        return False

    def max_steps(self, env: EnvBase) -> int:
        # Maximum number of steps for a rollout during evaluation
        return 1000

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        # The group map mapping group names to agent names
        # The data in the tensordict will havebe presented this way        
        
        return {"agent" : [agent for agent in env.agents]}
        
        # return {agent : [agent] for agent in env.agents}

    # def observation_spec(self, env: EnvBase) -> CompositeSpec:
    #     # A spec for the observation.
    #     # Must be a CompositeSpec with one (group_name, "observation") entry per group.
    #     return env.full_observation_spec

    # def action_spec(self, env: EnvBase) -> CompositeSpec:
    #     # A spec for the action.
    #     # If provided, must be a CompositeSpec with one (group_name, "action") entry per group.
    #     return env.full_action_spec

    def state_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        # A spec for the state.
        # If provided, must be a CompositeSpec with one "state" entry
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        # A spec for the action mask.
        # If provided, must be a CompositeSpec with one (group_name, "action_mask") entry per group.
        return None

    def info_spec(self, env: EnvBase) -> Optional[CompositeSpec]:
        # A spec for the info.
        # If provided, must be a CompositeSpec with one (group_name, "info") entry per group (this entry can be composite).
        return None

    # def observation_spec(self, env: EnvBase) -> CompositeSpec:
    #     # Dynamically create an observation spec for each agent, naming the group after the agent
    #     observation_specs = {}
    #     for agent_name in env.agents:
    #         observation_specs[agent_name] = CompositeSpec({
    #             'observation': BoundedTensorSpec(
    #                 shape=torch.Size([1, 15]),  # Assuming this is the shape of each agent's observation
    #                 low=torch.full((1, 15), -float('inf'), dtype=torch.float32),
    #                 high=torch.full((1, 15), float('inf'), dtype=torch.float32),
    #                 dtype=torch.float32,
    #                 device='cuda',
    #             ),
    #             # Add 'mask' or other specs as needed, depending on your environment's requirements
    #         })

    #     return CompositeSpec(observation_specs)



    # def action_spec(self, env: EnvBase) -> CompositeSpec:
    #     action_specs = {}
    #     for agent_name in env.agents:
    #         # Assuming each action has a lower bound of -1 and an upper bound of 1
    #         # Adjust the bounds according to your environment's requirements
    #         low = torch.full((4,), -1.0, dtype=torch.float32, device='cuda')
    #         high = torch.full((4,), 1.0, dtype=torch.float32, device='cuda')

    #         action_specs[agent_name] = CompositeSpec({
    #             'action': BoundedTensorSpec(
    #                 shape=torch.Size([4]),  # 4 continuous actions
    #                 low=low,
    #                 high=high,
    #                 dtype=torch.float32,  # Continuous actions are typically represented using floating-point numbers
    #                 device='cuda',  # Assuming you want to place the actions on a CUDA device
    #             ),
    #             # If your environment uses action masks to indicate valid actions, include them here
    #             # 'action_mask': ...
    #         })

    #     return CompositeSpec(action_specs)


    @staticmethod
    def env_name() -> str:
        # The name of the environment in the benchmarl/conf/task folder
        return "b_ace"

    # def log_info(self, batch: TensorDictBase) -> Dict[str, float]:
    #     # Optionally return a str->float dict with extra things to log
    #     # This function has access to the collected batch and is optional
    #     return {}


    def observation_spec(self, env: EnvBase) -> CompositeSpec:
        num_agents = len(env.agents)  # Dynamically determine the number of agents
        # Create a spec for aggregated observations under a single group "agent_1"
        obs_len = 19
        observation_spec = CompositeSpec({
            'agent': CompositeSpec({
                'observation': BoundedTensorSpec(
                    shape=torch.Size([num_agents, obs_len]),  # Adjust shape based on aggregation
                    low=torch.full((num_agents, obs_len), -float('inf'), dtype=torch.float32, device='cuda'),
                    high=torch.full((num_agents, obs_len), float('inf'), dtype=torch.float32, device='cuda'),                    
                    dtype=torch.float32,
                    device='cuda',
                ),
                # If you have a mask or other specs per agent, adjust accordingly
            })
        })
        return observation_spec



    def action_spec(self, env: EnvBase) -> CompositeSpec:
        
        action_type = env.action_type
        
        if action_type == "Low_Level_Continuous":
            # Assuming continuous actions for all agents, with 4 actions each
            num_actions = 4  # Number of continuous actions
            num_agents = len(env.agents) 
            action_spec = CompositeSpec({
                'agent': CompositeSpec({
                    'action': BoundedTensorSpec(
                        shape=torch.Size([num_agents, num_actions]),  # Assuming actions are aggregated, adjust shape as needed
                        low=torch.full((num_agents, num_actions), -1.0, dtype=torch.float32, device='cuda'),  # Adjust bounds as needed
                        high=torch.full((num_agents, num_actions), 1.0, dtype=torch.float32, device='cuda'),  # Adjust bounds as needed
                        dtype=torch.float32,
                        device='cuda',
                    ),
                    # Include additional specs like 'action_mask' if applicable
                })
            })
            
        elif action_type == "Low_Level_Discrete":
            # Define the action spec for discrete actions
            total_actions = 5 * 5 * 2  # Total combinations
            action_spec = CompositeSpec({
                'agent': CompositeSpec({
                    'action': DiscreteTensorSpec( 
                            total_actions ,
                            shape=torch.Size([env.num_agents]),
                            dtype=torch.float32,
                            device='cuda')
                })
            })
        
        return action_spec

        
    # def encode_action(turn_input, level_input, fire_input):
    #     # Example encoding, adjust based on your action space size
    #     return turn_input + (level_input * 5) + (fire_input * 25)


    # def decode_action(self, encoded_action):
    #     # Decode back to the original action tuple
    #     turn_input = encoded_action % 5
    #     level_input = (encoded_action // 5) % 5
    #     fire_input = (encoded_action // 25) % 2
    #     return turn_input, level_input, fire_input

  
       

    

    