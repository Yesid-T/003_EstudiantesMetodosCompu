Edad = int(input('Introduzca su edad'))
if Edad % 2 == 0:
    print('Es par')
    Total = 0
    Extremo = Edad * 10 + 1
    for Contador in range(10, Extremo):
        if Contador % 10 == 3:
            Total = Total + Contador
    print(Total)
else:
    print('No es par')
    Total = 0
    for Contador in range(Edad, Edad * 5):
        if Contador % 3 == 0:
            Total = Total + Contador
    print(Total)
