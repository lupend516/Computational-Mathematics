# Task 1

import numpy as np
import matplotlib.pyplot as plt

def newton_raphson(f,df,x0,tol):
    x=x0
    while abs(f(x))>tol:
        x=x-f(x)/df(x) 
    return x

def f(x):
    return x**3-(2*(x**2))-5

def df(x):
    return 3*x**2-4*x

root=newton_raphson(f,df,2.5,1e-6)
print(f"Root: {root}")

x=np.linspace(1,4,500) 
y=f(x)

plt.figure(figsize=(8,6))
plt.plot(x,y,label="f(x)=x^3-2x^2-5")
plt.axhline(0, color='red', linestyle='--', label='y=0')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Graphical method")
plt.legend()
plt.grid()
plt.show()

def calculate_error(true_value, approx_value):
    absolute_error=abs(true_value - approx_value) 
    return absolute_error

true_value=2.6906474992568943 
absolute_error=calculate_error(true_value, approx_value)

print(f"Absolute error: {absolute_error}")



# Task 2

import numpy as np
import matplotlib.pyplot as plt

def bisection_method(f,a,b,tol):
    if f(a)*f(b) >= 0:
        print("Invalid initial values. f(a) and f(b) must be of different signs.")
        return None

    iteration=0
    
    midpoint = (a+b)/2
    while abs(f(midpoint)) > tol:
        iteration += 1
        if f(a)*f(midpoint) < 0:
            b=midpoint
        else:
            a=midpoint
        midpoint=(a+b)/2
    print(f"Bisection iteration: {iteration}")
    return midpoint

def secant_method(f, x0, x1, tol):

    iteration=0
    
    while abs(f(x1)) > tol:
        iteration += 1
        x_temp = x1-f(x1)*(x1-x0)/(f(x1)-f(x0)) 
        x0, x1 = x1, x_temp 
    print(f"Secant iteration {iteration}")
    return x1
    
def f(x):
    return np.exp(x)-2*x-3

bisection_root=bisection_method(f,0,2,1e-6)
secant_root=secant_method(f,0,2,1e-6)
print(f"Approximate bisection root: {bisection_root}")
print(f"Approximate secant root: {secant_root}")

def calculate_errors(true_value, approx_value):
    absolute_error=abs(true_value-approx_value) 
    relative_error=absolute_error/true_value if true_value !=0 else float("inf") 
    return relative_error

exact_value=1.923938694754772 
bisection_root=1.9239387512207031 
secant_root=1.923938694754772 
bisection_iteration=18
secant_iteration=5
bisection_relative_error=calculate_errors(exact_value,bisection_root)
secant_relative_error=calculate_errors(exact_value,secant_root)

print(f"Bisection relative error {bisection_relative_error}")
print(f"Secant relative error {secant_relative_error}")

if bisection_iteration < secant_iteration:
    print("Bisection Method is more efficient in this case.")
elif secant_iteration < bisection_iteration:
    print("Secant Method is more efficient in this case.")
else:
    print("Both methods are equally efficient in terms of iterations.")



# Task 3

import numpy as np
import matplotlib.pyplot as plt

def newton_raphson(f, df, x0, tol):
    iterations=[]
    i=0
    while abs(f(x0))>tol:
        i+=1
        x=x0-f(x0)/df(x0) 
        absolute_error=abs(x-x0) 
        relative_error=absolute_error/x if x!=0 else float('inf')
        iterations.append((i, x, absolute_error, relative_error))
        x0=x
    return x,iterations

def f(x):
    return x**2-3*x+2
def df(x):
    return 2*x-3

root, iterations= newton_raphson(f, df, 2.5, 1e-6)

print("Iteration Table")
for i, x, abs_err, rel_err in iterations:
    print(f"Iteration {i}, Current Guess {x}, Absolute Error {abs_err}, Relative Error {rel_err}")

print(f"Root: {root}")

iteration_nums = [it[0] for it in iterations]
absolute_errors = [it[2] for it in iterations]

plt.figure(figsize=(8, 6))
plt.plot(iteration_nums, absolute_errors, marker='o', label='Absolute Error')
plt.xlabel("Iteration Number")
plt.ylabel("Absolute Error")
plt.title("Convergence of Newton-Raphson Method")
plt.yscale('log') 
plt.grid()
plt.legend()
plt.show()

x=np.linspace(0,3,500)
y=f(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y, label="f(x)=x^2-3x+2")
plt.axhline(0, color='red', linestyle='--', label="y = 0")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Graphical method for x^2-3x+2")
plt.legend()
plt.grid()
plt.show()



# task 4

import numpy as np
import matplotlib.pyplot as plt
import cmath 
def muller_method(f, x0, x1, x2, tol, max_iter=100):
    for _ in range(max_iter):
        h0 = x1 - x0
        h1 = x2 - x1
        delta0 = (f(x1) - f(x0)) / h0
        delta1 = (f(x2) - f(x1)) / h1
        a = (delta1 - delta0) / (h1 + h0)
        b = a * h1 + delta1
        c = f(x2)

        discriminant = cmath.sqrt(b**2 - 4*a*c)
        if abs(b + discriminant) > abs(b - discriminant):
            denominator = b + discriminant
        else:
            denominator = b - discriminant

        x3 = x2 - (2 * c) / denominator  

        if abs(x3 - x2) < tol:  
            return x3
        x0, x1, x2 = x1, x2, x3

    print("The method did not converge within the specified number of iterations.")
    return None

def f(x):
    return x**3 - x**2 + x + 1

root = muller_method(f, -1, 0, 1, 1e-6)
result_value=f(root)
absolute_error=abs(result_value) 
print(f"Root: {root}, Result: {result_value}, Absolute_error {absolute_error}")



# task 5

import numpy as np
import matplotlib.pyplot as plt

def false_position_method(f, a, b, tol):
    if f(a) * f(b) >= 0:
        print("Invalid initial values. f(a) and f(b) must be of different signs.")
        return None
    
    iterations = []  
    c = a           
    prev_c = None   
    i = 0           

    while True:
        i += 1
        c = b - f(b) * (b - a) / (f(b) - f(a))
        
        if prev_c is not None:
            abs_err = abs(c - prev_c)  
            rel_err = abs_err / abs(c) if c != 0 else np.inf  
        else:
            abs_err, rel_err = np.inf, np.inf  

        iterations.append((i, c, abs_err, rel_err))

        if abs(f(c)) <= tol:
            break

        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
        
        prev_c = c  

    return c, iterations

def f(x):
    return x**2-2**x  
def absolute_error(xapprox):
    return 

root,iterations = false_position_method(f, 0, 3, 1e-6)

print(f"Root: {root}")
print("Iteration Table")
for i, c, abs_err, rel_err in iterations:
    print(f"Iteration {i}, Current Guess {c}, Absolute Error {abs_err}, Relative Error {rel_err}")

abs_errors = [abs_err for _, _, abs_err, _ in iterations]
plt.plot(range(1, len(abs_errors) + 1), abs_errors, marker='o', linestyle='-')
plt.xlabel('Iteration Number')
plt.ylabel('Absolute Error')
plt.title('Absolute Error vs Iteration Number (False Position Method)')
plt.grid()
plt.show()



# task 6

import numpy as np
import matplotlib.pyplot as plt

def find_quadratic_roots(a, b, c):
    root1 = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
    root2 = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)
    return root1, root2

def fixed_point_iteration(g,x0,tol,max_iter, true_root):
    results=[]
    x=x0
    for i in range(max_iter):
        x_next = g(x)
        absolute_error=abs(true_root-x_next)
        results.append((i+1,x_next,absolute_error))
        if abs(x_next - x) < tol:  
            break
        x = x_next
    return results

def f(x):
    return x**2-6*x+5
def g(x):
    return (x**2+5)/6

root1,root2=find_quadratic_roots(1, -6, 5)

true_root=root2

results = fixed_point_iteration(g, 0.5, 1e-6, 10, true_root)
print(f"True root: {true_root}")
print(f"Results")
for i,x,abs_err in results:
    print(f"Iteration: {i}, Current guess: {x}, Absolute error: {abs_err}")
