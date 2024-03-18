import diffuser.utils as utils
from ml_logger import logger, RUN
import torch
import numpy as np
import gym
from config.diffuser import Config 
from diffuser.utils.arrays import to_torch, to_np, to_device
from d4rl.ope import normalize

def evaluate(**deps):
    RUN._update(deps)
    Config._update(deps)

    logger.remove('*.pkl')
    logger.remove("traceback.err")
    logger.log_params(Config=vars(Config), RUN=vars(RUN))

    Config.device = 'cuda'

    # Load configs
    torch.backends.cudnn.benchmark = True
    utils.set_seed(Config.seed)

    dataset_config = utils.Config(
        Config.loader,
        savepath='dataset_config.pkl',
        env=Config.dataset,
        horizon=Config.horizon,
        normalizer=Config.normalizer,
        preprocess_fns=Config.preprocess_fns,
        use_padding=Config.use_padding,
        max_path_length=Config.max_path_length,
        include_returns=Config.include_returns,
        returns_scale=Config.returns_scale,
    )

    render_config = utils.Config(
        Config.renderer,
        savepath='render_config.pkl',
        env=Config.dataset,
    )

    dataset = dataset_config()
    renderer = render_config()

    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim

    if Config.diffusion == 'models.GaussianInvDynDiffusion':
        transition_dim = observation_dim
    else:
        transition_dim = observation_dim + action_dim

    model_config = utils.Config(
        Config.model,
        savepath='model_config.pkl',
        horizon=Config.horizon,
        transition_dim=transition_dim,
        cond_dim=observation_dim,
        dim_mults=Config.dim_mults,
        dim=Config.dim,
        returns_condition=Config.returns_condition,
        device=Config.device,
    )

    diffusion_config = utils.Config(
        Config.diffusion,
        savepath='diffusion_config.pkl',
        horizon=Config.horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        n_timesteps=Config.n_diffusion_steps,
        loss_type=Config.loss_type,
        clip_denoised=Config.clip_denoised,
        predict_epsilon=Config.predict_epsilon,
        hidden_dim=Config.hidden_dim,
        ar_inv=Config.ar_inv,
        train_only_inv=Config.train_only_inv,
        action_dropout=Config.action_dropout,
        ## loss weighting
        action_weight=Config.action_weight,
        loss_weights=Config.loss_weights,
        loss_discount=Config.loss_discount,
        returns_condition=Config.returns_condition,
        condition_guidance_w=Config.condition_guidance_w,
        device=Config.device,
    )

    trainer_config = utils.Config(
        utils.Trainer,
        savepath='trainer_config.pkl',
        train_batch_size=Config.batch_size,
        train_lr=Config.learning_rate,
        gradient_accumulate_every=Config.gradient_accumulate_every,
        ema_decay=Config.ema_decay,
        sample_freq=Config.sample_freq,
        save_freq=Config.save_freq,
        log_freq=Config.log_freq,
        label_freq=int(Config.n_train_steps // Config.n_saves),
        save_parallel=Config.save_parallel,
        bucket=Config.bucket,
        n_reference=Config.n_reference,
        train_device=Config.device,
    )

    loadpath = "/home/pjutrasd/depot_symlink/projects/decision-diffuser/out/diffuser/checkpoint/state.pt"
    state_dict = torch.load(loadpath, map_location=Config.device)
    model = model_config()
    diffusion = diffusion_config(model)
    trainer = trainer_config(diffusion, dataset, renderer)
    logger.print(utils.report_parameters(model), color='green')
    trainer.step = state_dict['step']
    trainer.model.load_state_dict(state_dict['model'])
    trainer.ema_model.load_state_dict(state_dict['ema'])

    a_diffusion_config = utils.Config(
        Config.diffusion,
        savepath='action_diffusion_config.pkl',
        horizon=Config.horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        n_timesteps=Config.n_diffusion_steps,
        loss_type=Config.loss_type,
        clip_denoised=Config.clip_denoised,
        predict_epsilon=Config.predict_epsilon,
        hidden_dim=256,
        ar_inv=Config.ar_inv,
        train_only_inv=Config.train_only_inv,
        action_dropout=Config.action_dropout,
        action_nll=False,
        ## loss weighting
        action_weight=Config.action_weight,
        loss_weights=Config.loss_weights,
        loss_discount=Config.loss_discount,
        returns_condition=Config.returns_condition,
        condition_guidance_w=Config.condition_guidance_w,
        device=Config.device,
    )

    loadpath_action = [f"/home/pjutrasd/depot_symlink/projects/decision-diffuser/out/action{i}/checkpoint/state.pt" for i in range(1, 6)]
    state_dict_action = [torch.load(loadpath, map_location=Config.device) for loadpath in loadpath_action]
    diffusions = [a_diffusion_config(model_config()) for _ in range(5)]
    for i in range(5):
        diffusions[i].load_state_dict(state_dict_action[i]['ema'])
    action_models = [diffusion.inv_model for diffusion in diffusions]

    delta = 0
    num_eval = 1
    uncertainty_threshold = delta
    device = Config.device

    env_list = [gym.make(Config.dataset) for _ in range(num_eval)]
    obs_list = [env.reset()[None] for env in env_list]
    returns = to_device(Config.test_ret * torch.ones(num_eval, 1), device)
    dones = [0 for _ in range(num_eval)]
    episode_rewards = [0 for _ in range(num_eval)]
    steps = [0 for _ in range(num_eval)]
    saved_steps = [0 for _ in range(num_eval)]
    stds = []
    rewards = []
    saved_nfe = []

    while sum(dones) <  num_eval:
        # Sample from the diffusion model
        obs = np.concatenate(obs_list, axis=0)
        obs = dataset.normalizer.normalize(obs, 'observations')
        conditions = {0: to_torch(obs, device=device)}
        samples = trainer.ema_model.conditional_sample(conditions, returns=returns)

        print("Step", steps, "Done", dones)

        # Sample first actions
        obs_comb = torch.cat([samples[:, 0, :], samples[:, 1, :]], dim=-1)
        obs_comb = obs_comb.reshape(-1, 2*observation_dim)
        actions, actions_std = ensemble_prediction(  # f(s_0, s_1)
                    action_models,
                    obs_comb
        )

        actions = to_np(actions)
        actions = dataset.normalizer.unnormalize(actions, 'actions')

        # For each epidode,
        for i in range(num_eval):
            if dones[i] == 1:
                continue

            stds.append(to_np(actions_std[i]).mean())

            # Take the first step
            action = actions[i]
            this_obs, this_reward, this_done, _ = env_list[i].step(action)
            obs_list[i] = this_obs[None]
            steps[i] += 1
            episode_rewards[i] += this_reward
            if this_done:
                dones[i] = 1
                if steps[i] == Config.max_path_length: logger.print(f"Episode ({i}) completed", color='green')
                else: logger.print(f"Episode ({i}) died at step {steps[i]}", color='red')
                continue

            # Sample next actions sequentially
            for j in range(1, Config.horizon - 1):
                this_obs = dataset.normalizer.normalize(np.array([this_obs]), 'observations')[0]

                this_obs = to_torch(this_obs, device=device)
                obs_comb = torch.cat([this_obs, samples[i, j+1, :]], dim=-1)
                obs_comb = obs_comb.reshape(-1, 2*observation_dim)

                # start_time = time.time()
                action, action_std = ensemble_prediction(
                    action_models,
                    obs_comb,
                )

                # Take the next step if the uncertainty is low
                uncertainty = to_np(action_std).mean()
                if uncertainty >= uncertainty_threshold:  # uncertainty >= 0 falls back to the base policy without adaptibe planning  
                    break
                saved_steps[i] += 1
                stds.append(uncertainty)

                action = to_np(action)
                action = dataset.normalizer.unnormalize(action, 'actions')
                action = action[0]
                this_obs, this_reward, this_done, _ = env_list[i].step(action)

                obs_list[i] = this_obs[None]
                steps[i] += 1
                episode_rewards[i] += this_reward
                if this_done:
                    dones[i] = 1
                    if steps[i] == Config.max_path_length: logger.print(f"Episode ({i}) completed", color='green')
                    else: logger.print(f"Episode ({i}) died at step {steps[i]}", color='red')
                    break


    episode_rewards = np.array(episode_rewards)
    policy_id = Config.dataset[:-3]
    normalize_rewards = np.array([normalize(policy_id, reward) for reward in episode_rewards])
    rewards.append(normalize_rewards)
    saved_nfe.append(np.sum(saved_steps) / np.sum(steps))

    # print("Steps time", np.mean(steps_time))
    rewards = np.array(rewards)
    rewards = rewards.mean(axis=1)
    saved_nfe = np.array(saved_nfe)




def mc_dropout_prediction(model, input_data, num_samples=10):
    """
    Make predictions using Monte Carlo Dropout

    :param model: The PyTorch model with dropout layers
    :param input_data: The input data for prediction
    :param num_samples: Number of samples to draw
    :return: Averaged prediction and std from the Monte Carlo samples
    """
    model.train()  # Enable dropout
    predictions = [model(input_data) for _ in range(num_samples)]
    predictions_stack = torch.stack(predictions)
    mean_prediction = torch.mean(predictions_stack, dim=0)
    std_deviation = torch.std(predictions_stack, dim=0)
    return mean_prediction, std_deviation

def ensemble_prediction(models, input_data):
    """
    Make predictions using ensemble of models

    :param models: A list of models
    :param input_data: The input data for prediction
    :return: Averaged prediction and std from the ensemble
    """
    if len(models) == 1:
        prediction = models[0](input_data)
        return prediction, torch.zeros_like(prediction)
    predictions = [model(input_data) for model in models]
    predictions_stack = torch.stack(predictions)
    mean_prediction = torch.mean(predictions_stack, dim=0)
    std_deviation = torch.std(predictions_stack, dim=0)
    return mean_prediction, std_deviation

def ensemble_prediction_nll(models, input_data):
    """
    Make predictions using an ensemble of models trained with NLL, each outputting
    a mean and variance for predictive uncertainty estimation.

    :param models: A list of models.
    :param input_data: The input data for prediction.
    :return: Averaged prediction, aleatoric uncertainty, and total predictive uncertainty.
    """
    if len(models) == 1:
        pred_mean, pred_variance = models[0](input_data)
        # Ensure variance is positive
        pred_variance += 1e-6
        return pred_mean, torch.zeros_like(pred_mean), pred_variance

    predictions = [model(input_data) for model in models]
    means = torch.stack([pred[0] for pred in predictions])
    variances = torch.stack([pred[1] + 1e-6 for pred in predictions])

    mean_prediction = torch.mean(means, dim=0)
    # Aleatoric uncertainty: average variance from each model's prediction
    aleatoric_uncertainty = torch.mean(variances, dim=0)
    # Epistemic uncertainty: variance of the means across the models
    epistemic_uncertainty = torch.var(means, dim=0)
    # Total predictive uncertainty is the sum of aleatoric and epistemic uncertainties
    total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty

    return mean_prediction, total_uncertainty