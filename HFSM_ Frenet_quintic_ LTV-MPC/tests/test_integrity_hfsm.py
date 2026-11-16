from l2_path_planner.messages import (
    Function, FunctionRequest, FunctionRequests, Header, RequestState, Side,
)
from l2_path_planner.planner import HFSM, IntegrityMonitor


def test_message_age_future_and_counter_checks():
    monitor = IntegrityMonitor(0.25, 0.01)
    assert monitor.valid("lane", Header(1_000_000, 1), 1_100_000)
    # A consumer faster than the sensor (planner 40 Hz, camera 10 Hz) re-reads
    # the same frame; that is a hold-last-value, not a fault, while it is fresh.
    assert monitor.valid("lane", Header(1_000_000, 1), 1_110_000)
    # Staleness is bounded by max_input_age_s, measured from the capture stamp.
    assert not monitor.valid("lane", Header(1_000_000, 1), 1_300_001)
    # A new stamp carrying a repeated counter is still a counter fault.
    assert not monitor.valid("lane", Header(1_050_000, 1), 1_060_000)
    assert not monitor.valid("future", Header(1_020_000, 1), 1_000_000)


def test_hfsm_priority_and_completed_latch():
    hfsm = HFSM()
    req = FunctionRequests(Header(0, 0), [
        FunctionRequest(Function.HWA, RequestState.REQUEST),
        FunctionRequest(Function.LKA, RequestState.REQUEST, Side.RIGHT),
        FunctionRequest(Function.ELK, RequestState.REQUEST, Side.RIGHT),
    ])
    assert hfsm.select(req, True, 0.0) == Function.ELK
    for t in (0.0, 0.25, 0.51):
        completed = hfsm.update_completion(t, 0.0, 0.0, 0.0, 0.0)
    assert completed
    hfsm.after_completion(req, 0.51)
    assert hfsm.select(req, True, 0.6) == Function.LKA


def test_hfsm_completion_requires_added_acceleration_to_settle():
    hfsm = HFSM()
    req = FunctionRequests(Header(0, 0), [
        FunctionRequest(Function.LKA, RequestState.REQUEST, Side.RIGHT),
    ])
    assert hfsm.select(req, True, 0.0) == Function.LKA
    for t in (0.0, 0.25, 0.75):
        assert not hfsm.update_completion(t, 0.0, 0.0, 0.0, 0.25, 0.2)
    for t in (1.0, 1.25):
        assert not hfsm.update_completion(t, 0.0, 0.0, 0.0, 0.10, 0.2)
    assert hfsm.update_completion(1.51, 0.0, 0.0, 0.0, 0.10, 0.2)

