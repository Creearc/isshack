import zmq

def send(socket, tmp):
  tmp = tmp.copy()
  finished = False
  while not finished:
    try:
      print('send start')
      msg = pickle.dumps(tmp)
      socket.send(msg, zmq.NOBLOCK)
      socket.recv()     
      finished = True
    except:
      time.sleep(5.0)
  print('send finish')
