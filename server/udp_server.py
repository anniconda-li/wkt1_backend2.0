"""WTK1 UDP server loop.

The UDP side is independent from FastAPI: it receives device packets, records
basic traffic, forwards audio between devices on the same channel, and echoes
single-device audio for local tests. Keeping it separate lets the HTTP app stay
focused on API state and request handling.
"""

from __future__ import annotations

import socket

from server.protocol import PKT_TYPES, make_server_echo, parse_packet


def run_udp(host: str, port: int, *, log_func=print) -> None:
    """Run the blocking WTK1 UDP loop."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((host, port))
    except OSError as exc:
        log_func(f"UDP 绑定失败 {host}:{port}: {exc}")
        return
    log_func(f"UDP WTK1 监听 {host}:{port}")

    devices: dict[str, tuple[str, int, int]] = {}

    while True:
        data, addr = sock.recvfrom(2048)
        packet = parse_packet(data)
        if packet is None:
            log_func(f"UDP 原始数据 from {addr[0]}:{addr[1]} len={len(data)} data={data!r}")
            continue

        type_name = PKT_TYPES.get(packet.packet_type, f"type_{packet.packet_type}")
        devices[packet.device] = (addr[0], addr[1], packet.channel)
        log_func(
            f"UDP {type_name} from {packet.device}@{addr[0]}:{addr[1]} "
            f"ch={packet.channel} seq={packet.seq} payload={len(packet.payload)}"
        )

        if packet.packet_type == 4 and packet.payload:
            targets = [
                (dev, dev_addr)
                for dev, (ip, port, channel) in devices.items()
                if dev != packet.device and channel == packet.channel
                for dev_addr in [(ip, port)]
            ]
            if targets:
                for dev, dev_addr in targets:
                    sock.sendto(data, dev_addr)
                    log_func(f"UDP 音频转发至 {dev}@{dev_addr[0]}:{dev_addr[1]}")
            else:
                # Single-device test mode: use a server device id so clients do not drop it.
                sock.sendto(make_server_echo(data), addr)
