'''
def fibonnaci(n):
    if(n<=1):
        return(n)
    else:
        return(fibonnaci(n-1)+fibonnaci(n-2))
    
n = int(input("Enter any number to print fibonacci sequence: "))
print("Fibonacci sequence: ")
for i in range (n):
    print(fibonnaci(i))

'''
'''
def fib(n):
    if(n<=1):
        return(n)
    else:
        return(fib(n-1)+fib(n-2))
n=int(input("Enter any number to print fibnocci serirs: "))
print("finbocci serirs is: ")
for i in range(n):
    print(fib(i))
'''

def fib(n):
    if(n<=1):
        return(n)
    else:
        return(fib(n-1)+fib(n-2))
n=int(input("Enter any number: "))
print("fibnocci series till n is:")
for i in range(n):
    print(fib(i))