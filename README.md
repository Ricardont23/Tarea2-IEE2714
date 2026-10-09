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

sigma_kernel_global = "Valor σ para filtro global"

kernel_global, kernel_referencia = kernel_gauss(sigma_kernel_global), kernel_gauss("Valor σ para estimación de μ̂")

imagen_filtro = convolucion(kernel_global, imagen_ruido)

sigmas = np.arange(0, 6.05, 0.1)
resultados = rmse(imagen_ruido, imagen_original, sigmas)

imagen_suavizada = convolucion(kernel_referencia, imagen_ruido)
puntos_control = np.array([(0.15, 1.8), (0.45, 1.4), (0.80, 1.5)])


mapa_sigma = interpolacion_sigma(imagen_suavizada, puntos_control)


imagen_filtro_adaptado = filtro_gaussiano_adaptativo(imagen_ruido, mapa_sigma)

(...)
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

Por último, se encuentra el código en el que se comparan los valores RMSE de ambos filtros usados.

```ruby
rmse_globales = calcular_rmse(imagen_filtro, imagen_original)
rmse_adaptativos = calcular_rmse(imagen_filtro_adaptado, imagen_original)

print("=== Comparación de RMSE ===")
for region in ["fondo", "cuadrado", "circulo", "global"]:
    err_glob = rmse_globales[region]
    err_adap = rmse_adaptativos[region]
    
    print(f"Region: {region}")
    print(f"  Filtro Global (sigma={sigma_kernel_global }): {err_glob:.6f}")
    print(f"  Filtro Adaptativo:     {err_adap:.6f}")
    print(f"  Mejora: {(err_glob - err_adap) / err_glob * 100:.2f}%")
```

> Este resultado se muestra en la consola. Si se está visualizando una imagen, se debe cerrar para poder que se impriman los datos.

## Pregunta 2

El archivo `DifusiónAnisotropica.py` contiene todas las funciones necesarias para la correcta ejecución del código. Se desarrolla la lectura de la imagen, el añadido de ruido Gaussiano y la aplicación filtro de  difusión anisotrópica con los tres coeficientes diseñados: Aproximación de Variación Total (TV), un filtro propuesto con Laplaciano y un filtro basado en gradiente.

Posterior a la definición de las funciones, se encuentran las líneas del código principal a ser ejecutadas.

```ruby
(...)

imagen = cv2.imread("Imagenes/gato.jfif")
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
imagen_gris = imagen_gris.astype(np.float64) / 255.0
imagen_ruido = ruido_gauss(imagen_gris)

imagen_dif_an_tv = difusion_anisotropica(imagen_ruido, c_tv, lambd = 0.125, iteraciones=50, epsilon= 0.5)

imagen_dif_an_lap = difusion_anisotropica(imagen_ruido, funcion_c_laplaciano, lambd=0.125, iteraciones=50, gamma=0.5, epsilon=1.0)

imagen_dif_an_grad = difusion_anisotropica(imagen_ruido, funcion_c_gradiente, lambd=0.125, iteraciones=50, epsilon=1.0)

(...)
```
En este fragmento de código se definen los argumentos de los filtros. Las iteraciones, lambda, y los parametros de la función que define la constante **c**.

> La imagen `gato.jfif` se encuentra dentro de la carpeta imagenes en el repositorio

Consiguiente a la definición de variables, se definen una serie de líneas para poder graficar y obtener las distintas imágenes e indicadores utilizados para el informe.

```ruby
(...)

'''Para poder visualizar correctamente las diferentes graficas, hay que descomentar la requerida y comentar las demás.'''
#====================================================================================================================================

'''Grafica comparativa de la imagen ruidosa y los tres filtros aplicados'''

'''fig, axs = plt.subplots(1, 4, figsize=(16,6))
axs[0].imshow(imagen_ruido, cmap="gray", vmin=0, vmax=1); axs[0].set_title("Ruidosa")
axs[1].imshow(imagen_dif_an_tv, cmap="gray", vmin=0, vmax=1); axs[1].set_title("Filtro TV")
axs[2].imshow(imagen_dif_an_lap, cmap="gray", vmin=0, vmax=1); axs[2].set_title("Filtro Laplaciano")
axs[3].imshow(imagen_dif_an_grad, cmap="gray", vmin=0, vmax=1); axs[3].set_title("Filtro Propuesto 2")'''

'''Grafica del mapa del coeficiente de difusión c para un filtro específico'''
#graficar_mapa_c(imagen_ruido, funcion_c_laplaciano, epsilon=0.9, gamma=1.1)

'''Grafica de una unica imagen resultante'''
#plt.imshow(imagen_dif_an_grad, cmap="gray", vmin=0, vmax=1)

#=====================================================================================================================================


plt.show()

```

En última instancia, se encuentra el código para imprimir los valores RMSE.

```ruby
(...)

'''Para poder visualizar correctamente las diferentes graficas, hay que descomentar la requerida y comentar las demás.'''
resultados = {
    "Ruidosa":      imagen_ruido,
    "TV":           imagen_dif_an_tv,
    "Propuesta 1 (Laplaciano)":  imagen_dif_an_lap,
    "Propuesta 2":  imagen_dif_an_grad
}

for nombre, img in resultados.items():
    print(f"{nombre}: RMSE = {rmse_imagenes(img, imagen_gris):.5f}")
```

### Advertencias y consideraciones

**Para el uso correcto se debe descomentar la línea deseada, mientras las demás siguen comentadas y posteriormente ejecutar el código.**

**Si se busca graficar el mapeo de c de cualquiera de las funciones implementadas, se debe quitar o añadir los argumentos entregados a la función, para que sean expactamente los que pide la función c, según corresponda. De lo contrario, se levantará un error**

**Los valores RMSE se imprimen en consola despues de cerrar las figuras graficadas.**

