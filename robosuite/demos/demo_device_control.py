"""Teleoperate robot with keyboard or SpaceMouse.

***Choose user input option with the --device argument***

Keyboard:
    We use the keyboard to control the end-effector of the robot.
    The keyboard provides 6-DoF control commands through various keys.
    The commands are mapped to joint velocities through an inverse kinematics
    solver from Bullet physics.

    Note:
        To run this script with macOS, you must run it with root access.

SpaceMouse:

    We use the SpaceMouse 3D mouse to control the end-effector of the robot.
    The mouse provides 6-DoF control commands. The commands are mapped to joint
    velocities through an inverse kinematics solver from Bullet physics.

    The two side buttons of SpaceMouse are used for controlling the grippers.

    SpaceMouse Wireless from 3Dconnexion: https://www.3dconnexion.com/spacemouse_wireless/en/
    We used the SpaceMouse Wireless in our experiments. The paper below used the same device
    to collect human demonstrations for imitation learning.

    Reinforcement and Imitation Learning for Diverse Visuomotor Skills
    Yuke Zhu, Ziyu Wang, Josh Merel, Andrei Rusu, Tom Erez, Serkan Cabi, Saran Tunyasuvunakool,
    János Kramár, Raia Hadsell, Nando de Freitas, Nicolas Heess
    RSS 2018

    Note:
        This current implementation only supports macOS (Linux support can be added).
        Download and install the driver before running the script:
            https://www.3dconnexion.com/service/drivers.html

Additionally, --pos_sensitivity and --rot_sensitivity provide relative gains for increasing / decreasing the user input
device sensitivity


***Choose controller with the --controller argument***

Choice of using either inverse kinematics controller (ik) or operational space controller (osc):
Main difference is that user inputs with ik's rotations are always taken relative to eef coordinate frame, whereas
    user inputs with osc's rotations are taken relative to global frame (i.e.: static / camera frame of reference).

    Notes:
        OSC also tends to be more computationally efficient since IK relies on the backend pybullet IK solver.


***Choose environment specifics with the following arguments***

    --environment: Task to perform, e.g.: "Lift", "TwoArmPegInHole", "NutAssembly", etc.

    --robots: Robot(s) with which to perform the task. Can be any in
        {"Panda", "Sawyer", "IIWA", "Jaco", "Kinova3", "UR5e", "Baxter"}. Note that the environments include sanity
        checks, such that a "TwoArm..." environment will only accept either a 2-tuple of robot names or a single
        bimanual robot name, according to the specified configuration (see below), and all other environments will
        only accept a single single-armed robot name

    --config: Exclusively applicable and only should be specified for "TwoArm..." environments. Specifies the robot
        configuration desired for the task. Options are {"bimanual", "single-arm-parallel", and "single-arm-opposed"}

            -"bimanual": Sets up the environment for a single bimanual robot. Expects a single bimanual robot name to
                be specified in the --robots argument

            -"single-arm-parallel": Sets up the environment such that two single-armed robots are stationed next to
                each other facing the same direction. Expects a 2-tuple of single-armed robot names to be specified
                in the --robots argument.

            -"single-arm-opposed": Sets up the environment such that two single-armed robots are stationed opposed from
                each other, facing each other from opposite directions. Expects a 2-tuple of single-armed robot names
                to be specified in the --robots argument.

    --arm: Exclusively applicable and only should be specified for "TwoArm..." environments. Specifies which of the
        multiple arm eef's to control. The other (passive) arm will remain stationary. Options are {"right", "left"}
        (from the point of view of the robot(s) facing against the viewer direction)

    --switch-on-grasp: Exclusively applicable and only should be specified for "TwoArm..." environments. If enabled,
        will switch the current arm being controlled every time the gripper input is pressed

    --toggle-camera-on-grasp: If enabled, gripper input presses will cycle through the available camera angles

Examples:

    For normal single-arm environment:
        $ python demo_device_control.py --environment PickPlaceCan --robots Sawyer --controller osc

    For two-arm bimanual environment:
        $ python demo_device_control.py --environment TwoArmLift --robots Baxter --config bimanual --arm left --controller osc

    For two-arm multi single-arm robot environment:
        $ python demo_device_control.py --environment TwoArmLift --robots Sawyer Sawyer --config single-arm-parallel --controller osc


"""

import argparse
from pathlib import Path

import numpy as np

import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import GymWrapper, VisualizationWrapper

class TeleOpPolicy:
    def __init__(self, action_dim, device, active_robot, active_arm, env_configuration, blocking=True, env=None):
        self.last_grasp = 0
        self.action_dim = action_dim
        self.device = device
        self.active_robot = active_robot
        self.active_arm = active_arm
        self.env_configuration = env_configuration
        self.blocking = blocking
        self.env = env  # hold env pointer to optionally render


    def __call__(self, obs, state=None):
        self.env.render()
        if self.blocking:
            action = np.zeros(self.action_dim, dtype=np.float32)
            while np.allclose(action[:-1], np.zeros_like(action[:-1])):
                action, self.last_grasp = input2action(device=self.device, robot=self.active_robot, active_arm=self.active_arm, env_configuration=self.env_configuration)
                action = self._pad_action(action)
        else:
            action, self.last_grasp = input2action(device=self.device, robot=self.active_robot, active_arm=self.active_arm, env_configuration=self.env_configuration)
            action = self._pad_action(action)

        return {'action': np.array([action])}, state

    def _pad_action(self, action):
        # Fill out the rest of the action space if necessary
        rem_action_dim = self.action_dim - action.size
        if rem_action_dim > 0:
            # Initialize remaining action space
            rem_action = np.zeros(rem_action_dim)
            # This is a multi-arm setting, choose which arm to control and fill the rest with zeros
            if self.active_arm == "right":
                action = np.concatenate([action, rem_action])
            elif self.active_arm == "left":
                action = np.concatenate([rem_action, action])
        elif rem_action_dim < 0:
            # We're in an environment with no gripper action space, so trim the action space to be the action dim
            action = action[: self.action_dim]
        
        return action
    
class EpisodeReset:
    def __init__(self):
        self._ep_cntr = embodied.Counter()

    def __call__(self, ep, worker):
        self._ep_cntr.increment()
        print(f'Episode {int(self._ep_cntr)} complete.')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument(
        "--config", type=str, default="single-arm-opposed", help="Specified environment configuration if necessary"
    )
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--switch-on-grasp", action="store_true", help="Switch gripper control on gripper action")
    parser.add_argument("--toggle-camera-on-grasp", action="store_true", help="Switch camera angle on gripper action")
    parser.add_argument("--controller", type=str, default="osc", help="Choice of controller. Can be 'ik' or 'osc'")
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    args = parser.parse_args()

    # Import controller config for EE IK or OSC (pos/ori)
    if args.controller == "ik":
        controller_name = "IK_POSE"
    elif args.controller == "osc":
        controller_name = "OSC_POSE"
    else:
        print("Error: Unsupported controller specified. Must be either 'ik' or 'osc'!")
        raise ValueError

    # Get controller config
    controller_config = load_controller_config(default_controller=controller_name)

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config
    else:
        args.config = None

    # Create environment
    root_env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        hard_reset=False,
    )

    # Wrap this environment in a visualization wrapper
    root_env = VisualizationWrapper(root_env, indicator_configs=None)
    root_env = GymWrapper(root_env)

    # Setup printing options for numbers
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
        root_env.viewer.add_keypress_callback(device.on_press)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    else:
        raise Exception("Invalid device choice: choose either 'keyboard' or 'spacemouse'.")


    # ----- Initialize ----- #
    # Setup rendering
    obs = root_env.reset()
    cam_id = 0
    num_cam = len(root_env.sim.model.camera_names)
    root_env.render()
    last_grasp = 0  # Initialize variables that should the maintained between resets
    device.start_control()  # Initialize device control
    active_robot = root_env.robots[0] if args.config == "bimanual" else root_env.robots[args.arm == "left"]  # Set active robot

    policy = TeleOpPolicy(root_env.action_dim, device=device, active_robot=active_robot, active_arm=args.arm, env_configuration=args.config, env=root_env)
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    print(f'Observation Space: {root_env.observation_space}')

    import dreamerv3.embodied as embodied
    from dreamerv3.embodied.replay.log_replay_wrapper import FromGymnasiumLogReplayDriver
    step = embodied.Counter()
    logger = embodied.Logger(step, [
        embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
        embodied.logger.JSONLOutput(logdir, 'scores.jsonl', 'episode/score'),
        embodied.logger.TensorBoardOutput(logdir)
    ])

    driver = FromGymnasiumLogReplayDriver(root_env, logdir / 'replay', logger=logger, chunks=256, batch_length=64, replay_size=1e6, on_episode_fns=[EpisodeReset()], is_gym=False)
    driver.run(policy, steps=0, episodes=args.episodes)

