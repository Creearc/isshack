

def shift(data):
  out = []
  out.append(data)
  sec, people = data
  for i in range(1, len(people)):
    start = people[len(people) - i: ]
    end = people[ : len(people) - i]
    out.append([sec, start + end])
  
  return out

def pavel_padding(data, size=10):
  sec, people = data
  start_size = len(people)
  for i in range(size - start_size):
    people.append(people[i])
  return [sec, people]
  


if __name__ == '__main__':
  import pprint
  data = [10, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
  data = pavel_padding(data, size=10)
  data = shift(data)
  pprint.pprint(data)

  data = [1, [1, 2, 3]]
  data = pavel_padding(data, size=10)
  data = shift(data)
  pprint.pprint(data)

  
