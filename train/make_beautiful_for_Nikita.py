f = open('fight_train.txt', 'r')
f_out = open('nikita_fighter_train.txt', 'w')
for s in f:
  tmp = s.split()
  for i in range(int(tmp[0]), int(tmp[1])):
    f_out.write('{} {}\n'.format(i, tmp[2]))

f.close()
f_out.close()
