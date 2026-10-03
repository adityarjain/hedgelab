import torch

from hedgelab.hedging import Hedger, cvar, delta_hedge, leland_hedge, pnl, report, simulate, train


def test_delta_hedge_is_fair_and_error_shrinks_like_sqrt_dt():
    S30, S120 = simulate(50_000, steps=30, seed=1), simulate(50_000, steps=120, seed=1)
    p30, p120 = pnl(S30, delta_hedge(S30), 0.0), pnl(S120, delta_hedge(S120), 0.0)
    assert abs(p30.mean()) < 0.02  # premium = BS price, so a costless delta hedge breaks even
    assert 0.4 < (p120.std() / p30.std()).item() < 0.6  # 4x more rebalances -> ~half the error


def test_cvar_is_mean_of_worst_tail():
    assert cvar(torch.arange(100.0), 0.95).item() == 97.0  # mean of 95..99


def test_costs_only_ever_reduce_pnl():
    S = simulate(10_000, seed=2)
    h = delta_hedge(S)
    diff = pnl(S, h, 0.0) - pnl(S, h, 0.002)
    assert (diff >= 0).all() and diff.mean() > 0


def test_leland_equals_delta_without_costs():
    S = simulate(1_000, seed=3)
    assert torch.allclose(leland_hedge(S, 0.0), delta_hedge(S))


def test_hedger_does_not_look_ahead():
    """Scramble prices after step t: holdings up to t must not change."""
    S, t = simulate(500, seed=4), 12
    S2 = S.clone()
    S2[:, t + 1 :] *= torch.empty_like(S2[:, t + 1 :]).uniform_(0.5, 2.0)
    m = Hedger()
    with torch.no_grad():
        assert torch.equal(m(S)[:, : t + 1], m(S2)[:, : t + 1])


def test_short_training_beats_delta_under_costs():
    cost, S = 0.005, simulate(50_000, seed=5)
    model = train(cost, iters=200, batch=4096)
    with torch.no_grad():
        net = report(S, model(S), cost)
    assert net["CVaR95"] < report(S, delta_hedge(S), cost)["CVaR95"]
    assert net["turnover"] < report(S, delta_hedge(S), cost)["turnover"]  # it learns to trade less
