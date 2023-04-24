#comms :  server 
import socket

s = socket.socket()

s.bind(('0.0.0.0', 8090))
s.listen(0)

while True:
    client, addr = s.accept()
    print(f"Connection from {addr[0]}:{addr[1]}")
    a=["hi","gimbul"]
    client.send(','.join(a).encode())  # Convert list of strings to a single string and encode to bytes
    client.close()
