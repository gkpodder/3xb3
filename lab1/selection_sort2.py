'''
This code has a bug. Currently working on fixing it
'''


#L is a list
import random


def selection_sort2(L):
  low = 0
  high = len(L)-1
  while (low < high):
    minIndex = low
    maxIndex = high
    for i in range(low,high + 1):
      #In a given iteration, find min and max index
      if L[i] < L[minIndex]:
        minIndex = i

      if L[i] > L[maxIndex]:
        maxIndex = i

    #swapping logic
    minValue = L[minIndex]
    maxValue = L[maxIndex]

    temp1 = L[low]
    L[low] = minValue
    L[minIndex] = temp1 
      
    temp2 = L[high]
    L[high] = maxValue

    if L[minIndex] == maxValue:  
      L[minIndex] = temp2
 
    else:
      L[maxIndex] = temp2
    
    low += 1
    high -= 1

  return L

#function from bad_sorts file
def create_random_list(length, max_value):
    return [random.randint(0, max_value) for _ in range(length)]

def main():
  #sampleList = create_random_list(7,23)
  sampleList = [19,20,19,23,4,11,21]
  print("Unsorted List: {}".format(sampleList))
  print("Sorted List: {}".format(selection_sort2(sampleList)))

main()





def Bugselection_sort2(L):
  low = 0
  high = len(L)-1
  while (low < high):
    minIndex = low
    maxIndex = high
    for i in range(low,high + 1):
      #logic of adjusting min and max index
      if L[i] < L[minIndex]:
        minIndex = i

      if L[i] > L[maxIndex]:
        maxIndex = i
    
    #swapping logic
    temp1 = L[low]
    L[low] = L[minIndex]
    L[minIndex] = temp1 
      
    temp2 = L[high]
    L[high] = L[maxIndex]
    L[maxIndex] = temp2

    low += 1
    high -= 1

  return L