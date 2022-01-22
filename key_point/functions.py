

def shift(people):
  out = []
  out.append(data)
  for i in range(1, len(people)):
    start = people[len(people) - i: ]
    end = people[ : len(people) - i]
    out.append(start + end)
  
  return out

def pavel_padding(people, size=10):
  start_size = len(people)
  for i in range(size - start_size):
    people.append(people[i])
  for i in range(start_size - size):
    people.pop(-1)
  return people
  


if __name__ == '__main__':
  import pprint
  data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  data = pavel_padding(data, size=10)
  data = shift(data)
  pprint.pprint(data)

  data = [1, 2, 3]
  data = pavel_padding(data, size=10)
  data = shift(data)
  pprint.pprint(data)

  data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
  data = pavel_padding(data, size=10)
  data = shift(data)
  pprint.pprint(data)

  data = [1]
  data = pavel_padding(data, size=10)
  data = shift(data)
  pprint.pprint(data)

  
