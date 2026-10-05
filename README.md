# Tarea2-IEE2714
Repositorio de códigos e imágenes (y demás) usados para el desarrollo de la tarea 2.  

## Pregunta 1

El archivo `FiltroGaussiano.py` contiene todas las funciones necesarias para la correcta ejecución del código. Se desarrolla la construcción de la imagen, el añadido del ruido y la aplicación de los filtros global y localmente adaptado.

Posterior a la definición de las funciones se encuentran las lineas del código principal a ser ejecutadas. 

```ruby

(...)
rng = np.random.default_rng(23)


imagen_original = construir_imagen()
imagen_ruido = rng.poisson(imagen_original * 40)/40

kernel_global, kernel_referencia = kernel_gauss("Valor σ para filtro global"), kernel_gauss("Valor σ para estimación de μ̂")

imagen_filtro = convolucion(kernel_global, imagen_ruido)

sigmas = np.arange(0, 6.05, 0.1)
resultados = rmse(imagen_ruido, imagen_original, sigmas)

imagen_suavizada = convolucion(kernel_referencia, imagen_ruido)
puntos_control = np.array([(0.15, 1.8), (0.45, 1.4), (0.80, 1.5)])


mapa_sigma = interpolacion_sigma(imagen_suavizada, puntos_control)


imagen_filtro_adaptado = filtro_gaussiano_adaptativo(imagen_ruido, mapa_sigma)
```
Se define al número **23** como la semilla a utilizar para obtener el ruido. Y, se definen los valores de σ requeridos según el analisis.

Consiguiente a la definición de variables, se definen una serie de lineas para poder graficar y obtener las distintas imagenes utilizadas para el informe. 


```ruby

(...)

'''Para poder visualizar correctamente las diferentes graficas, hay que descomentar la requerida y comentar las demás.'''
# =============================================================================================================

'''Grafica de una unica imagen'''
#plt.imshow(imagen_filtro_adaptado, cmap='gray', vmin=0.0, vmax=1.0)

'''Grafica de curvas RMSE | Valores entre 0 y 6 |'''
#graficar_rmse(sigmas, resultados)

'''Grafica del mapeo de σ(x, y)'''
#plt.imshow(interpolacion_sigma(imagen_suavizada, puntos_control), cmap="viridis")

'''Grafica de la curva de σ = F(μ̂) '''
#graficar_F_mu(puntos_control[:,0], puntos_control[:,1])

'''Grafica de comparación de flitros y su diferencia'''
#comparar_filtros_imagen(imagen_filtro, imagen_filtro_adaptado)
#================================================================================================================

plt.show()
```
Para el uso correcto **se debe descomentar la linea a deseada, mientras las demás siguen comentadas** y posteriormente ejecutar el código.  


>Al hacer uso de un solo `plt.show()` un uso incorrecto podria superponer las diferentes graficas dificultando el analisis y la correcta visualización de las figuras.