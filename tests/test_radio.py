import numpy as np

from floodsim.radio import BASE, Radio


def test_delivery_only_within_range():
    r = Radio(3, 3000.0, (0.0, 0.0))
    pos = np.array([[100.0, 0.0], [2900.0, 0.0], [9000.0, 0.0]])
    r.send(0.0, 0, "hello")
    r.deliver(pos, 0.0)
    assert len(r.take(1)) == 1 and len(r.take(2)) == 0 and len(r.inbox[BASE]) == 1


def test_signal_radius_edge():
    R = 3000.0
    r = Radio(2, R, (-1e6, -1e6))
    r.send(0.0, 0, "a")
    r.deliver(np.array([[0.0, 0.0], [0.99 * R, 0.0]]), 0.0)
    assert len(r.take(1)) == 1
    r.send(0.0, 0, "b")
    r.deliver(np.array([[0.0, 0.0], [1.01 * R, 0.0]]), 0.0)
    assert len(r.take(1)) == 0


def test_multi_hop_connectivity():
    r = Radio(3, 3000.0, (0.0, 0.0))
    pos = np.array([[2500.0, 0.0], [5000.0, 0.0], [20000.0, 0.0]])
    assert r.connected_to_base(pos).tolist() == [True, True, False]
