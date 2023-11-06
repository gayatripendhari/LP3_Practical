def fibonacci(n):
    if(n<=1):
        return n
    else:
        return (fibonacci(n-1)+fibonacci(n-2))
        
n=int(input("Enter number of term print:"))
print("fibonacci sequence")
for i in range(n):
    print(fibonacci(i))