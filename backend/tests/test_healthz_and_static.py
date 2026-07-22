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
