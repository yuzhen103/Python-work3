#!/usr/bin/env python
# coding: utf-8

# In[1]:


lamret=lambda x: x**2
print(lamret(6))


# In[2]:


type(lamret)


# In[3]:


lambret2=lambda x, y, z: ((x-y)**z)/2
print(lambret2(3,1,2))


# In[4]:


lambret3=lambda x, y: x**2 if x>y else 1/x**2
print(lambret3(4,3))


# In[5]:


print(lambret3(4,5))


# In[6]:


def polyequ(x):
    return(x**2)+(4*x)+6
sample=4
result=polyequ(sample)
print('呼叫函數(x**2)+(4*x)+6', result)


# In[7]:


import sympy as sp
x = sp.Symbol('x')
print(sp.diff(3*x**2+1, x))

from scipy.misc import derivative
def f(x):
    return 3*x**2+1

print(derivative(f, 2))


# In[8]:


def d(x):
    return derivative(f,x)

print(d(2))

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np

y=np.linspace(-3,3)
plt.plot(y,f(y))
plt.plot(y,d(y))


# In[9]:


import numpy as np

dx=float(input('Please input dx= '))
area=0
y=0
for x in np.arange(2,5,dx):
    y=(x**2)+1
    area=area+(y*dx)
print('area=', area)


# In[10]:


import sympy as sp
x=sp.Symbol('x')
sp.integrate(3.0*x**2+1,x)


# In[11]:


from scipy.integrate import quad
def f(x):
    return 3*x**2+1
i=quad(f,0,2)
print(i[0])


# In[12]:


import sympy as sp
x=sp.Symbol('x')
sp.integrate(sp.sin(3.0*x),x)


# In[13]:


from scipy.integrate import quad
import numpy as np

def f(x):
    return np.exp(-x)*np.sin(3*x)

i=quad(f,0,2*np.pi)
print(i[0])


# In[14]:


import numpy as np


# In[15]:


A = np.array([[3,-9],[2,4]])


# In[16]:


print(A)


# In[17]:


B = np.array([-42,2])


# In[18]:


Z = np.linalg.solve(A, B)


# In[19]:


print(Z)


# In[20]:


M = np.array([[1,-2,-1],[2,2,-1],[-1,-1,2]])


# In[21]:


print(M)


# In[22]:


c = np.array([6,1,1])


# In[23]:


y = np.linalg.solve(M,c)


# In[24]:


print(y)


# In[ ]:




