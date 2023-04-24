import socket

from alpha6 import extract_features

s = socket.socket()

s.bind(('0.0.0.0', 8000))
s.listen(0)

testpath = "C:/Users/Rohan Mahesh Rao/Documents/PES1UG20EC156/Sem 6/ML/project/gtzan/Data/genres_original/jazz/jazz.00003.wav"
datain = extract_features(testpath)

def run_comms(listval):

    while True:
        client, addr = s.accept()
        print(f"Connection from {addr[0]}:{addr[1]}")
        client.send(','.join(listval).encode())  # Convert list of strings to a single string and encode to bytes
        client.close()

run_comms(datain)
