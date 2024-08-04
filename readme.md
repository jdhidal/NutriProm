# NutriProm

NutriProm es una aplicación de reconocimiento de imágenes que convierte predicciones en gramos y proporciona campos nutricionales para los alimentos. Utiliza TensorFlow para el modelo de predicción, Flask como el servidor backend y Gradio para la interfaz de usuario.


## Cómo Ejecutar el Programa

1. **Clona el repositorio** (si aún no lo has hecho):
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd backend
   crear carpeta models
   descargar md5 de entrenamiento **[Descargar el modelo de TensorFlow](<>)**
   dejar copiarlo en la carpeta models 

2. **Instalar las Dependencias**:
   ```bash
   pip install -r requirements.txt

2. **Configura las variables de entorno**:
   ```bash
   create .env
   MODEL_PATH=ruta/backend/models/fotos-comida-model (1).h5
   LABELS_FILE_PATH=ruta/backend/labels.txt

2. **Ejecuta el programa**:
   ```bash
   python app.py

## Uso

1. **Accede a la aplicación:**

   URL local: http://127.0.0.1:7860
   URL pública: Enlace público (Este enlace caducará en 72 horas).
   Sube una imagen:

2. **Sube la imagen:**

   La aplicación te permitirá subir una imagen de un alimento. El modelo procesará la imagen y mostrará las predicciones en gramos junto con los campos nutricionales relevantes.

## Estructura del Proyecto

- **backend/**: Carpeta para archivos de datos.
  - **app.py/**: El archivo principal que contiene la lógica de la aplicación Flask y la integración con Gradio.
- **requirements.txt**: Archivo de dependencias de Python..
- **.gitignore**: Archivos y carpetas a ignorar por Git.
- **README.md**: Documentación del proyecto.
