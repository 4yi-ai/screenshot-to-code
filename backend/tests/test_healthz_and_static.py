def test_healthz_route_registered():
    from main import app
    paths = {getattr(r, "path", None) for r in app.routes}
    assert "/healthz" in paths


def test_home_get_root_removed():
    from main import app
    get_root = [
        r for r in app.routes
        if getattr(r, "path", None) == "/" and "GET" in (getattr(r, "methods", None) or set())
    ]
    assert get_root == []


def test_eval_routes_not_registered():
    # The evals router routes client-supplied model strings to native provider
    # SDKs (bypassing the 4yi gateway / per-org metering). It must not be
    # mounted in the v1 gateway container.
    from main import app
    eval_paths = [
        str(getattr(r, "path", ""))
        for r in app.routes
        if str(getattr(r, "path", "")).startswith("/run_evals")
        or str(getattr(r, "path", "")).startswith("/evals")
    ]
    assert eval_paths == []
