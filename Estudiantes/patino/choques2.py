import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pygame

pygame.mixer.init()
sonido = pygame.mixer.Sound(r"C:\Users\sebas\Downloads\choque.mp3")


#parametros iniciales:
m1, m2 = 1, 10000 
#distanci minima
size2= 30
dmin=2

n=len(str(m2))-1

if (m2/10000)>1: 
    v2= 2*(10**(n-3)) 
else:
   v2= 5*(10**(n-3)) 
dt= 5/m2

#Funciones para obtener las velocidades tras una colision
def w1(v1,v2):
  return ((m1-m2)*v1+2*m2*v2)/(m1+m2)
def w2(v2,v1):
  return ((m2-m1)*v2+2*m1*v1)/(m1+m2)

fig, ax = plt.subplots()
line1, = ax.plot([], [], 'ro', markersize= 15)  # línea para x1 (roja)
line2, = ax.plot([], [], 'bo', markersize=size2)  # línea para x2 (azul, más grande)

def init():
    ax.set_xlim(0, 30)
    ax.set_ylim(-1, 1)
    return line1, line2  # Devolver ambas líneas


data = {'x1': 15, 'x2': 20, 'v1': 0, 'v2': -v2, 'dt': dt}

contador = 0

#-----------------------------------------------------------------------------------------
text_colisiones = ax.text(0.95, 0.95, '', transform=ax.transAxes, ha='right', va='top') 

def update(frame, line1, line2, data):

    global contador  # Acceder a la variable global

    data['x1'] += data['v1'] * data['dt']
    data['x2'] += data['v2'] * data['dt']

    line1.set_data([data['x1']], [0])  # Actualizar posición de line1
    line2.set_data([data['x2']], [0])  # Actualizar posición de line2

    if ((abs(data['x1'] - data['x2'])) <= dmin or data['x1']==(data['x2'])) :  #condiion de colision para las dos masas
        data['v1'], data['v2'] = w1(data['v1'], data['v2']), w2(data['v2'], data['v1'])
        contador += 1  # Incrementar la cuenta
        sonido.play() #sonido tras colision

    # Detectar colisión con paredes para x1
    if data['x1'] <= .7:
        data['v1'] *= -1  # Invertir velocidad
        contador+= 1 #contar colision
        sonido.play()

    if data['x1'] > 20 and data['x2'] > 30:
        if data['x1'] < data['x2']:
            ani.event_source.stop()  # Detener la animación si se cumplen las condiciones

    text_colisiones.set_text(f'Colisiones: {contador}') 
    
    return line1, line2, text_colisiones # Devolver el objeto de texto también
#--------------------------------------ooo------------------------------------------------------

ani = FuncAnimation(fig, update, frames=range(400), 
                    fargs=(line1, line2, data),  # Incluir line2 en fargs
                    init_func=init, blit=True, interval=1)

plt.show()