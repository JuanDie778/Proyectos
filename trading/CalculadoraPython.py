
def calculadora(precio_compra, precio_venta, inversion):
    porcentaje = (precio_venta - precio_compra)/(precio_compra/100)
    return {'Porcentaje de ganancia %':porcentaje, 'Ganancia/Perdida':(porcentaje/100)*inversion, 'total': inversion+ ((porcentaje/100)*inversion)}
#determina el capital que deberia tener invertido para generar ciertas ganancias con un porcentaje de crecimiento esperado
def cuantoGanariaSi(porcentaje, ganancia_esperado):
    capital = ganancia_esperado/(porcentaje/100)
    return {'Deberia tener': capital}
#Me dice el tiempo(numero de veces que tendria que reinvertir) para llegar hasta un determiando capital y bajo un porcentaje de crecimiento constante
def cuantoTiempoNecesitoParaTener(Capital_de_arranque, capital_esperado, porcentaje_de_crecimiento):
    tiempo = 0
    while Capital_de_arranque <= capital_esperado:
        Capital_de_arranque += capital_esperado*(porcentaje_de_crecimiento/100)
        tiempo+= 1
    return tiempo



print(calculadora(118988, 127000, 2237))
#print(cuantoGanariaSi(3, 100))
#print(cuantoTiempoNecesitoParaTener(2237, 10000, 3))

