from isaaclab.utils import configclass


@configclass
class Go2RoughRunnerCfg:
    """Legacy rsl_rl-compatible config for the plain Go2 PPO task."""

    seed = 1

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 2
        num_mini_batches = 1
        learning_rate = 1.0e-3
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner:
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 8
        max_iterations = 2
        save_interval = 100
        experiment_name = "go2_rough_lab"
        run_name = "PPO"
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
