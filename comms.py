import socket

def run_comms(listval):

    s = socket.socket()

    s.bind(('0.0.0.0', 8000))
    s.listen(0)

    while True:
        client, addr = s.accept()
        print(f"Connection from {addr[0]}:{addr[1]}")
        client.send(','.join(listval).encode())  # Convert list of strings to a single string and encode to bytes
        client.close()

