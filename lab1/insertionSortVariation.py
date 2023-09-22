#L is a list of numbers
import random
def insertion_sort2(L):
  for i in range(1,len(L)):
    value = L[i]
    while i > 0:
      if value < L[i-1]:
        L[i] = L[i-1]
      else:
        L[i] = value
        break
      i = i - 1

    if i == 0:
      L[i] = value

  return L

#function from bad_sorts file
def create_random_list(length, max_value):
    return [random.randint(0, max_value) for _ in range(length)]

def main():
  sampleList = create_random_list(10,40)
  print("Unsorted List: {}".format(sampleList))
  print("Sorted List: {}".format(insertion_sort2(sampleList)))

main()