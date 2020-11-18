Ejecución:

Para el uso de la interfaz que clasifica numeros basta con ejecutar
el archivo main, donde la variable mode, define cual modelo se va a utilizar para
las predicciones, 1 para Deep Learning y 0 para SVM.

Para visualizar gráficas se pueden llamara a las funciones:

plot_deep_learning: recibe una lista con los parámetros que se desean variar y graficar.

Por ejemplo: plot_deep_learning(["epochs", "layers", "neurons", "training_set"])

plot_svm: recibe una lista con los parámetros que se desean variar y graficar.

plot_svm(['kernel', 'C', 'gamma'])


Para realizar la búsqueda con todos los parámetros, basta con llamar a las funciones:

best_dl_model(True)

best_svm_model(True) 

Sin el parámetro True se cargará el mejor modelo que se encuentra previamente almacenado


Si solo se quiere utilizar un modelo con los parámetros se debe instanciar las clases:

SVM() o DeepLearning() Las cuales reciben en su constructor los datos de prueba y entrenamiento admeás de la
configuración de parametros deseados, si no se proveen serán tomados los parámetros por defecto.

Para la creación de datos se utiliza el objeto create_data() de la siguiente forma:

x_train, y_train, x_val, y_val, x_test, y_test = create_data()

Para acelerar por CUDA puede intentar descomentando las siguientes líneas en el archivo deep_learning.py 

physical_devices = tf.config.list_physical_devices('GPU')

tf.config.experimental.set_memory_growth(physical_devices[0], True)
