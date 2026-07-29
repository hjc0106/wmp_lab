from isaaclab.utils import configclass


_MOTION_FILES = [
    "datasets/mocap_motions/hop1.txt",
    "datasets/mocap_motions/hop2.txt",
    "datasets/mocap_motions/trot1.txt",
    "datasets/mocap_motions/trot2.txt",
]


@configclass
class Go2AmpWMPRunnerCfg:
    """Legacy WMPRunner-compatible config for Go2 AMP/WMP.

    The sizes below mirror the original IsaacGym Go2 AMP config. For a fast
    smoke test pass ``--max_iterations 1`` to the launch script; for full
    training increase ``amp_num_preload_transitions`` (orig. 2_000_000) and
    ``max_iterations`` (orig. 20_000).
    """

    seed = 1
    amp_motion_files = list(_MOTION_FILES)

    class policy:
        init_noise_std = 1.0
        encoder_hidden_dims = [256, 128]
        wm_encoder_hidden_dims = [64, 64]
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [512, 256, 128]
        latent_dim = 32 + 3
        wm_latent_dim = 32
        activation = "elu"

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.0e-3
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        vel_predict_coef = 1.0
        amp_replay_buffer_size = 1_000_000

    class runner:
        policy_class_name = "ActorCritic"
        algorithm_class_name = "AMPPPO"
        num_steps_per_env = 24
        max_iterations = 1
        save_interval = 500
        keep_last_n = 5
        experiment_name = "go2_amp_lab"
        run_name = "WMP"
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
        amp_reward_coef = 0.5 * 0.02
        amp_motion_files = list(_MOTION_FILES)
        amp_num_preload_transitions = 200_000
        amp_task_reward_lerp = 0.3
        amp_discr_hidden_dims = [1024, 512]
        min_normalized_std = [0.05, 0.02, 0.05] * 4

    class depth_predictor:
        lr = 3e-4
        weight_decay = 1e-4
        training_interval = 10
        training_iters = 1000
        batch_size = 1024
        loss_scale = 100