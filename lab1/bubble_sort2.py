def bubble_sort2(L):
    for i in range(len(L)):
        val = None

        for j in range(len(L) - 1 - i):
            print(*L)
            if val is None:
                if L[j] > L[j+1]:
                    val = L[j]
                    L[j] = L[j+1]
            
            else:
                if val > L[j+1]:
                    L[j] = L[j+1]
                
                else:
                    L[j] = val
                    val = None
        
        if val is not None: 
            L[len(L) - 1 - i] = val
