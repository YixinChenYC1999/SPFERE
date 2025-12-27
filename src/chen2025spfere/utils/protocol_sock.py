# protocol_sock.py
# Unified message protocol with type codes and framed send/receive over TCP sockets

# === Message Type Definitions ===
MSG_TYPE_MODEL        = b'\x01'  # Model file (binary)
MSG_TYPE_INFO         = b'\x02'  # JSON info / text payload (e.g. client_info)

# Hard disconnect / terminate
MSG_TYPE_TERMINATE    = b'\xFF'  # Terminate connection (by server or client)

# === Utility ===
MSG_TYPE_NAMES = {
    MSG_TYPE_MODEL:        "MODEL",
    MSG_TYPE_INFO:         "INFO",
    MSG_TYPE_TERMINATE:    "TERMINATE"
}

# === Protocol send/recv implementation ===
def send_packet(sock, msg_type: bytes, payload: bytes, desc="Sending"):
    assert len(msg_type) == 1, "msg_type must be a single byte"
    length = len(payload).to_bytes(8, 'big')
    sock.sendall(msg_type + length)

    sent = 0
    while sent < len(payload):
        chunk = payload[sent:sent+4096]
        sock.sendall(chunk)
        sent += len(chunk)

def recv_packet(sock, desc="Recving"):
    header = sock.recv(9)
    if len(header) < 9:
        raise ConnectionError("Incomplete header received")
    msg_type = header[0:1]
    length = int.from_bytes(header[1:], 'big')

    data = b""
    while len(data) < length:
        chunk = sock.recv(min(4096, length - len(data)))
        if not chunk:
            raise ConnectionError("Connection lost during data reception")
        data += chunk
    return msg_type, data
