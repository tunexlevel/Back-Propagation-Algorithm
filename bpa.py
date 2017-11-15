# Back-Propagation-Algorithm
This is an algorithm that implements BPA for a multiple layer neural network

import math
   
def getWy(g,w):
    v = 0
    
    for i in range(len(w)):
        if w[i][2] == g:
            
            v += float(w[i][0])*float(w[i][1])
    return v

def getWz(g,w,y):
    v = 0
    for i in range(len(w)):
        if w[i][2] == g:
            #index for the hidden layer
            k = w[i][1]
            j = int(k[1])-1
            
            v += float(w[i][0])*float(y[j])
    return v

def getB(g,b):
    v = 0;       
    for i in range(len(b)):
        if b[i][1] == g[0]: 
            v = float(b[i][0])
    return v

def setNetY(w,b,y):
    Y = []
    for i in range(len(y)):
        v = getWy(y[i],w) +  getB(y[i],b)
        Y.append(v)
    return Y

def setOutY(y):
    Oy = [];       
    for i in range(len(y)):
        v = 1/(1+math.exp(-(y[i])))
        Oy.append(v)
    return Oy

def setNetZ(w,b,oY,z):
    Z = [];       
    for i in range(len(z)):
        v = getWz(z[i],w,oY) +  getB(z[i],b)
        Z.append(v)
    return Z

def setOutY(y):
    Oy = [];       
    for i in range(len(y)):
        v = 1/(1+math.exp(-(y[i])))
        Oy.append(v)
    return Oy

def setOutZ(z):
    Oz = [];       
    for i in range(len(z)):
        v = 1/(1+math.exp(-(z[i])))
        Oz.append(v)
    return Oz

def Etot(Oz,t):
    Et = 0  
    for i in range(len(Oz)):
        Et += 0.5*(math.pow((t[i] - Oz[i]),2))
    return Et


def backWardPassY(oZ,oY,n,w,t,z,y,nW):
    for i in range(len(y)):
        for j in range(len(w)):
            if w[j][2] == y[i]:
                hY   = w[j][2]
                m    = int(hY[1])-1
                eTot = eTotYw(oY[m],oZ,nW,t,w[j][1],hY)
                nW[j][0] = float(w[j][0])-(float(n)*float(eTot))
    return nW

def eTotYw(oY,oZ,nW,t,x,hY):
    v = float(eTotYz(oZ,nW,t,hY))*float(oY*(1-oY))*float(x)
    return v

def eTotYz(oZ,nW,t,hY):
    v = 0
    for j in range(len(nW)):
        if nW[j][1] == hY:
            z = nW[j][2]
            n = int(z[1])-1
            v += float(oZ[n]-t[n])*float(oZ[n]*(1-oZ[n]))*float(nW[j][0])
    return v
    

def backWardPassZ(oZ,n,w,t,z,oY):
    nW=w
    for i in range(len(z)):
        for j in range(len(w)):
            if w[j][2] == z[i]:
                hY = w[j][1]
                m  = int(hY[1])-1
                v = float(w[j][0])-(float(n)*float(eTotZw(oZ[i],t[i],oY[m])));
                nW[j][0] = v 
        
    return nW
    
    
def eTotZw(oZ,t,oY):
    v = (oZ-t)*(oZ*(1-oZ))*oY
    return float(v)
    
def main():
    
    w   = [[0.14,0.06,'y1'],[0.19,0.11,'y1'],
           [0.24,0.06,'y2'],[0.2,0.11,'y2'],
           [0.39,'y1','z1'],[0.43,'y2','z1'],
           [0.5,'y1','z2'],[0.5,'y2','z2']]
    y   = ['y1','y2']
    z   = ['z1','z2']
    t   = [0.01,0.9]
    b   = [[0.4,'y'],[0.5,'z']]
    n   = 0.5
   
    for i in range(400):
        nY  = setNetY(w,b,y)
        oY  = setOutY(nY)
        nZ  = setNetZ(w,b,oY,z)
        oZ  = setOutZ(nZ)
        Et  = Etot(oZ,t)
        
        bZ  = backWardPassZ(oZ,n,w,t,z,oY);
        bY  = backWardPassY(oZ,oY,n,w,t,z,y,bZ)
        rEt = round(Et,2)
        print('Etot ===',round(Et,2))
        if rEt == 0.00:
            print('The network has been able to learn successfully at iteration-'+str(i+1))
            break
        else:
            w   = bY #all new weight generated
main()
