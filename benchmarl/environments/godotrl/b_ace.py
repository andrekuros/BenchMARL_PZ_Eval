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
                                    
                                    #num_envs=10,
                                    #num_agents = 1, 
                                    #seed = seed,
                                    #port = GodotRLPettingZooWrapper.DEFAULT_PORT + random.randint(0,3000),
                                    convert_action_space = False,
                                    **self.config), 
                                    
                                    # scenario=self.name.lower(),
                                    #num_envs=10,#num_envs,  # Number of vectorized envs (do not use this param if the env is not vectorized)
                                    # continuous_actions=continuous_actions,#continuous_actions,  # Ignore this param if your env does not have this choice                                    
                                    use_mask=True, # Must use it since one player plays at a time                                    
                                    #device=device,
                                    #categorical_actions=True,  # If your env has discrete actions, they need to be categorical (TorchRL can help with this)
                        ) 
     
    def supports_continuous_actions(self) -> bool:
        # Does the environment support continuous actions?
        return True

    def supports_discrete_actions(self) -> bool:
        # Does the environment support discrete actions?
        return True

    def has_render(self, env: EnvBase) -> bool:
        # Does the env have a env.render(mode="rgb_array") or env.render() function?
        return False

    def max_steps(self, env: EnvBase) -> int:
        # Maximum number of steps for a rollout during evaluation
        return 1500

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        # The group map mapping group names to agent names
        # The data in the tensordict will havebe presented this way        
        
        return {"agent" : [agent for agent in env.agents]}
        
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
        obs_len = env.observation_space("agent_0").shape[0]
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
                            dtype=torch.int64,
                            device='cuda')
                })
            })
        
        return action_spec
  
       

    

    