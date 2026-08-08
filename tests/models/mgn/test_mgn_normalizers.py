import torch

from structbench.models.mgn.normalizers import OnlineNormalizer


def test_identity_before_any_accumulation():
    n = OnlineNormalizer(size=3)
    x = torch.randn(5, 3)
    torch.testing.assert_close(n(x), x)
    torch.testing.assert_close(n.inverse(x), x)


def test_converges_to_moments_and_inverts():
    torch.manual_seed(0)
    n = OnlineNormalizer(size=2)
    data = torch.randn(1000, 2) * torch.tensor([3.0, 0.5]) + torch.tensor([1.0, -2.0])
    n(data, accumulate=True)
    out = n(data)
    assert abs(float(out.mean())) < 0.05
    assert abs(float(out.std()) - 1.0) < 0.05
    torch.testing.assert_close(n.inverse(n(data)), data, rtol=1e-4, atol=1e-4)


def test_accumulation_cap_counts_calls_not_samples():
    # cap = 2 CALLS; each call carries 10 samples. Third call must be ignored.
    n = OnlineNormalizer(size=1, max_accumulations=2)
    n(torch.zeros(10, 1), accumulate=True)  # call 1: counts
    n(torch.ones(10, 1), accumulate=True)  # call 2: counts (20 samples ok)
    assert int(n._count) == 20  # samples accumulated across 2 calls
    mean_before = (n._sum / n._count).clone()
    n(torch.full((10, 1), 100.0), accumulate=True)  # call 3: over cap, ignored
    assert int(n._count) == 20
    torch.testing.assert_close(n._sum / n._count, mean_before)


def test_constant_feature_column_stays_finite():
    n = OnlineNormalizer(size=2)
    x = torch.cat([torch.full((50, 1), 3.0), torch.randn(50, 1)], dim=1)
    n(x, accumulate=True)
    out = n(x)  # constant column: variance clamps to 0 -> std_epsilon, no NaN
    assert torch.isfinite(out).all()


def test_state_dict_roundtrip():
    n = OnlineNormalizer(size=2)
    n(torch.randn(50, 2) + 5.0, accumulate=True)
    m = OnlineNormalizer(size=2)
    m.load_state_dict(n.state_dict())
    x = torch.randn(4, 2)
    torch.testing.assert_close(n(x), m(x))


def test_module_dtype_cast_does_not_downcast_accumulators():
    n = OnlineNormalizer(size=2)
    n(torch.randn(50, 2) + 5.0, accumulate=True)
    n.float()  # module-wide cast must not degrade the accumulators
    assert n._sum.dtype == torch.float64
    assert n._sum_sq.dtype == torch.float64
    assert n._count.dtype == torch.float64
    assert n._n_accumulations.dtype == torch.int64
    out = n(torch.randn(4, 2))  # still normalizes correctly, correct output dtype
    assert out.dtype == torch.float32
    x = torch.randn(3, 2)
    torch.testing.assert_close(n.inverse(n(x)), x, rtol=1e-4, atol=1e-4)
