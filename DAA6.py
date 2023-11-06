def printjobscheduling(arr, t):
    n=len(arr)
    
    for i in  range (n):
        for j in range (n -1 -i ):
            if arr[j] [2] < arr [j + 1] [2]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                result=[false] * t
                job=['-1'] *t
                
                for i in range(len(arr)):
                    for j in range (min(t -1, arr[i][1] -1), -1, -1):
                        if result[j] is false:
                            result[j] = true
                            job[j]= arr[i][0]
                            break
                            
                    print(job)
if __name__ == '__main__':
    arr =[['a',2,100],['b',1,19],['c',2,27],['d',1,25],['e',3,15]]
    print("following is max profit seqence of jobs")
    printjobscheduling(arr,3)