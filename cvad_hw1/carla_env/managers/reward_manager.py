class RewardManager():
    """Computes and returns rewards based on states and actions."""
    def __init__(self):
        pass

    def get_reward(self, state, action):
        """Returns the reward as a dictionary. You can include different sub-rewards in the
        dictionary for plotting/logging purposes, but only the 'reward' key is used for the
        actual RL algorithm, which is generated from the sum of all other rewards."""
        reward_dict = {}
        # Your code here
        # I realized in the stae dictionary, we have more keys than that are mentioned in the homework pdf
        # I decided to use some of them since they seem useful
        
        # here is the list of state keys:
        # speed
        # waypoint_dist
        # waypoint_angle
        # command
        # route_dist
        # route_angle
        # lane_dist
        # lane_angle
        # is_junction
        # hazard
        # hazard_dist
        # hazard_coords
        # tl_state
        # tl_dist
        # optimal_speed
        # measurements
        # rgb
        # gps
        # imu

        # we need to speed up
        if state["optimal_speed"] > state["speed"]:
            reward_dict["throttle"] = action["throttle"]

        # we need to slow down
        if state["optimal_speed"] < state["speed"]:
            reward_dict["brake"] = action["brake"]

        # keep close to the optimal speed
        speed_diff = abs(state["optimal_speed"] - state["speed"])/100
        reward_dict["speed_diff"] = 1.0 - speed_diff


        # some rewards for steering based on the command,
        if state["command"] == 0:
            steering_reward = -1.0 * action["steer"] # -1 steering is left, so this will be positive if the steering is to the left
        elif state["command"] == 1:
            steering_reward = action["steer"] # +1 steering is right, so this will be positive if the steering is to the right
        elif state["command"] == 2:
            steering_reward = 1.0 - abs(action["steer"])
        elif state["command"] == 3:
            steering_reward = -(state["lane_angle"] - action["steer"])**2
        reward_dict["steering"] = steering_reward

        # keep the steering according to the waypoint angle
        reward_dict["waypoint"] = -1.0 * abs(state["waypoint_angle"]- action["steer"])
        
        # don't diverge from the route
        reward_dict["route_dist"] = (1.0 - state["route_dist"])

        # if traffic light, penalize throttle
        if state["tl_state"] == 1:
            reward_dict["tl"] = 1.0 - action["throttle"]

        # penalize the collision
        reward_dict["collision"] = state["collision"] * -20.0

        # Your code here
        reward = 0.0
        for val in reward_dict.values():
            reward += val
        reward_dict["reward"] = reward
        return reward_dict
