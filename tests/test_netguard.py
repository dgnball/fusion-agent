import socket

import pytest

from fusion_agent import netguard


@pytest.fixture
def guard():
    netguard.install()
    yield
    netguard.uninstall()


def test_remote_name_lookup_is_blocked(guard):
    with pytest.raises(netguard.EgressBlocked):
        socket.getaddrinfo("example.com", 443)


def test_remote_connection_is_blocked(guard):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s, pytest.raises(netguard.EgressBlocked):
        s.connect(("93.184.215.14", 443))


def test_loopback_is_allowed(guard):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        with socket.create_connection(("localhost", server.getsockname()[1]), timeout=2):
            pass


def test_uninstall_restores_sockets(guard):
    netguard.uninstall()
    assert not netguard.installed()
    assert socket.socket.connect is not netguard._connect
