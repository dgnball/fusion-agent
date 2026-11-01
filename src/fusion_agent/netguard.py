"""Refuse every network connection that doesn't stay on this machine.

The agent is meant to run where there is no internet, so this is enforced rather than assumed. Once
`install()` has run, any attempt to resolve a remote host name or connect to a non-loopback address raises
`EgressBlocked`. That covers our own code and every library underneath it, such as model hubs that try to
check for updates.
"""

import ipaddress
import os
import socket

LOCAL_NAMES = {"localhost", "localhost.localdomain", ""}

_original = {}


class EgressBlocked(OSError):
    pass


def _is_local(host) -> bool:
    if host is None:
        return True
    host = host.decode() if isinstance(host, bytes) else str(host)
    host = host.strip("[]").lower()
    if host in LOCAL_NAMES:
        return True
    try:
        return ipaddress.ip_address(host.split("%")[0]).is_loopback
    except ValueError:
        return False


def _check(host) -> None:
    if not _is_local(host):
        raise EgressBlocked(f"Network egress blocked: {host} is not on this machine")


def _connect(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        _check(address[0])
    return _original["connect"](self, address)


def _connect_ex(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        _check(address[0])
    return _original["connect_ex"](self, address)


def _getaddrinfo(host, *args, **kwargs):
    _check(host)
    return _original["getaddrinfo"](host, *args, **kwargs)


def install() -> None:
    """Block remote network access for the rest of this process."""
    if _original:
        return
    _original.update(connect=socket.socket.connect, connect_ex=socket.socket.connect_ex, getaddrinfo=socket.getaddrinfo)
    socket.socket.connect = _connect
    socket.socket.connect_ex = _connect_ex
    socket.getaddrinfo = _getaddrinfo
    # Tell model libraries not to try in the first place.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def uninstall() -> None:
    if not _original:
        return
    socket.socket.connect = _original["connect"]
    socket.socket.connect_ex = _original["connect_ex"]
    socket.getaddrinfo = _original["getaddrinfo"]
    _original.clear()


def installed() -> bool:
    return bool(_original)
