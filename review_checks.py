"""Small assertion checks for each implementation stage of lab07-bandits.

Checks restore NumPy's random state so running them does not change experiments.
"""

from contextlib import contextmanager
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np


@contextmanager
def _fixed_seed(seed=12345):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def _record(bandit):
    # Arm 1 has the largest total; arm 0 has the largest empirical mean.
    for arm, reward in ((0, 4.0), (1, 3.0), (1, 3.0), (2, -1.0)):
        bandit.add_result(arm, reward)
    return bandit


def _history(bandit):
    return [(record.arm_idx, record.reward) for record in bandit.hist]


def check_reward_sampling(pull_np, pull_gauss_np):
    with _fixed_seed():
        rewards = pull_np([0, 1, 0, 1], np.array([0.0, 1.0]))
        assert np.array_equal(rewards, [0, 1, 0, 1]), (
            "Bernoulli p=0 and p=1 must always return 0 and 1, respectively."
        )
        arms = np.zeros(30_000, dtype=int)
        rewards = pull_np(arms, np.array([0.3]))
        assert rewards.shape == arms.shape and np.isin(rewards, [0, 1]).all()
        assert abs(rewards.mean() - 0.3) < 0.02
        deterministic = pull_gauss_np([1, 0, 1], np.array([-2.0, 3.0]), variance=0)
        assert np.array_equal(deterministic, [3, -2, 3]), (
            "Gaussian rewards must use the selected arm's mean."
        )
        rewards = pull_gauss_np(arms, np.array([2.0]), variance=4.0)
        assert rewards.shape == arms.shape and np.isfinite(rewards).all()
        assert abs(rewards.mean() - 2.0) < 0.08
        assert abs(rewards.var() - 4.0) < 0.15, (
            "np.random.normal expects standard deviation, not variance."
        )
    print("Reward-sampling assertion checks passed.")


def check_base_bandit(HistoryRecord, BaseBandit, RandomBandit):
    bandit = BaseBandit(3)
    other = BaseBandit(3)
    assert np.array_equal(bandit.arms_counter, [0, 0, 0])
    assert np.array_equal(bandit.rewards, [0, 0, 0]) and bandit.hist == []
    for arm, reward in enumerate((0.5, -2.0, 1.5)):
        assert bandit.choose_arm() == arm, "Pull every arm once before using the policy."
        bandit.add_result(arm, reward)
    bandit.add_result(1, 0.75)
    assert np.array_equal(bandit.arms_counter, [1, 2, 1])
    assert np.allclose(bandit.rewards, [0.5, -1.25, 1.5])
    assert all(isinstance(record, HistoryRecord) for record in bandit.hist)
    assert _history(bandit) == [(0, 0.5), (1, -2.0), (2, 1.5), (1, 0.75)]
    assert other.hist == [] and not other.rewards.any(), "Instances must not share state."
    with patch.object(bandit, "next_arm_idx", return_value=2) as policy:
        assert bandit.choose_arm() == 2
        policy.assert_called_once_with()
    with _fixed_seed():
        random_bandit = _record(RandomBandit(3))
        choices = np.array([random_bandit.choose_arm() for _ in range(6000)])
        assert np.isin(choices, [0, 1, 2]).all()
        assert np.all(np.abs(np.bincount(choices, minlength=3) / len(choices) - 1 / 3) < 0.03)
    print("Base and random bandit assertion checks passed.")


def check_greedy_bandit(GreedyBandit):
    bandit = _record(GreedyBandit(3))
    assert bandit.choose_arm() == 0, "Choose the largest mean, not the largest total."
    bandit.add_result(0, -10.0)
    assert bandit.choose_arm() == 1, "Recompute empirical means after new feedback."
    print("Greedy bandit assertion checks passed.")


def check_epsilon_greedy(EpsGreedyBandit):
    for eps, draw, expected in ((0.0, 0.0, 0), (1.0, 0.999, 2), (0.2, 0.1, 2), (0.2, 0.9, 0)):
        bandit = _record(EpsGreedyBandit(3, eps=eps))
        with patch.object(np.random, "rand", return_value=draw), patch.object(
            np.random, "randint", return_value=2
        ) as explore:
            assert bandit.choose_arm() == expected, (
                f"Wrong exploration/exploitation branch for eps={eps}, draw={draw}."
            )
            assert explore.call_count == int(expected == 2)
    for eps in (-0.1, 1.1):
        try:
            EpsGreedyBandit(3, eps=eps)
        except ValueError:
            pass
        else:
            raise AssertionError("Reject epsilon outside [0, 1].")
    print("Epsilon-greedy assertion checks passed.")


def check_ucb1(UCB1Bandit):
    for c, expected in ((0.0, 0), (0.3, 0), (0.35, 1), (1.0, 1), (np.sqrt(2), 1)):
        bandit = UCB1Bandit(3, c=c)
        for arm, count, reward in ((0, 10, 0.6), (1, 1, 0.2), (2, 10, 0.1)):
            for _ in range(count):
                bandit.add_result(arm, reward)
        assert bandit.choose_arm() == expected, (
            "UCB must combine the empirical mean and c * sqrt(log(t) / N(a))."
        )
    print("UCB1 assertion checks passed.")


def check_thompson_sampling(ThompsonBandit):
    bandit = ThompsonBandit(3)
    assert np.array_equal(bandit.alpha, [1, 1, 1])
    assert np.array_equal(bandit.beta, [1, 1, 1])
    for arm, reward in ((0, 1), (1, 0), (2, 1), (0, 0), (2, 1)):
        bandit.add_result(arm, reward)
    assert np.array_equal(bandit.alpha, [2, 1, 3]), "Alpha counts successes plus the prior."
    assert np.array_equal(bandit.beta, [2, 2, 1]), "Beta counts failures plus the prior."
    assert np.array_equal(bandit.arms_counter, [2, 1, 2])
    assert np.array_equal(bandit.rewards, [1, 0, 2]) and len(bandit.hist) == 5
    with patch.object(np.random, "beta", return_value=np.array([0.2, 0.8, 0.4])) as sample:
        assert bandit.choose_arm() == 1, "Choose the maximum posterior sample."
        sample.assert_called_once()
        assert np.array_equal(sample.call_args.args[0], bandit.alpha)
        assert np.array_equal(sample.call_args.args[1], bandit.beta)
    before = (_history(bandit), bandit.alpha.copy(), bandit.beta.copy())
    try:
        bandit.add_result(0, 0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("Beta-Bernoulli Thompson sampling requires binary rewards.")
    assert _history(bandit) == before[0]
    assert np.array_equal(bandit.alpha, before[1]) and np.array_equal(bandit.beta, before[2])
    fig, ax = plt.subplots()
    try:
        bandit.plot_distributions(ax, step=3)
        assert len(ax.lines) == 3
        assert len(set(ax.get_legend_handles_labels()[1])) == 3
        # Plotting an earlier posterior must not modify the current posterior.
        assert _history(bandit) == before[0]
        assert np.array_equal(bandit.alpha, before[1]) and np.array_equal(bandit.beta, before[2])
    finally:
        plt.close(fig)
    print("Thompson sampling assertion checks passed.")


def check_experiments(RandomBandit, BanditSettings, run_bandits,
                      plot_single_type, conduct_experiment):
    existing_figures = set(plt.get_fignums())
    try:
        with _fixed_seed():
            bandits = [RandomBandit(3), RandomBandit(3)]
            run_bandits(bandits, np.array([0.1, 0.5, 0.9]), n_steps=30, seed=7)
            for bandit in bandits:
                assert len(bandit.hist) == 30 and bandit.arms_counter.sum() == 30
                assert [record.arm_idx for record in bandit.hist[:3]] == [0, 1, 2]
                assert np.allclose(bandit.rewards.sum(), sum(r.reward for r in bandit.hist))
            assert _history(bandits[0]) != _history(bandits[1]), (
                "Independent runs should not all receive the same random trajectory."
            )
            fig, axes = plot_single_type(bandits, opt_arm_idx=2, n_steps=30)
            for ax in axes:
                labels = ax.get_legend_handles_labels()[1]
                assert len(labels) == len(set(labels)) == 2, "Label each run separately."
                assert ax.get_xlabel() and ax.get_ylabel(), "Label both plot axes."
            assert np.allclose(axes[0].lines[0].get_ydata(),
                               np.cumsum([r.reward for r in bandits[0].hist]) / np.arange(1, 31))
            assert np.allclose(axes[1].lines[0].get_ydata(),
                               np.cumsum([r.arm_idx == 2 for r in bandits[0].hist]) / np.arange(1, 31))
            plt.close(fig)

            def experiment(seed):
                return conduct_experiment(
                    [BanditSettings(RandomBandit)], num_bandits_per_type=3,
                    num_steps=30, distribution_means=[0.1, 0.5, 0.9],
                    num_bandits_for_sanity_check=0, is_gauss=False, seed=seed,
                )

            first = experiment(42)
            axes = plt.gcf().axes
            expected_reward = np.mean([
                np.cumsum([r.reward for r in b.hist]) / np.arange(1, 31) for b in first[0]
            ], axis=0)
            expected_optimal = np.mean([
                np.cumsum([r.arm_idx == 2 for r in b.hist]) / np.arange(1, 31) for b in first[0]
            ], axis=0)
            assert np.allclose(axes[0].lines[0].get_ydata(), expected_reward)
            assert np.allclose(axes[1].lines[0].get_ydata(), expected_optimal)
            # An unrelated run must not affect replay of a seeded experiment.
            run_bandits([RandomBandit(3)], np.array([0.1, 0.5, 0.9]), n_steps=5)
            replay = experiment(42)
            different = experiment(43)
            assert [_history(b) for b in first[0]] == [_history(b) for b in replay[0]], (
                "The same seed and parameters must reproduce histories, regardless of cell order."
            )
            assert [_history(b) for b in first[0]] != [_history(b) for b in different[0]]
    finally:
        for number in set(plt.get_fignums()) - existing_figures:
            plt.close(number)
    print("Simulation, plotting and reproducibility assertion checks passed.")
