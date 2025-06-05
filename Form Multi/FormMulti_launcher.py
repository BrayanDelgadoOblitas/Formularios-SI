import subprocess
import os

# Ruta completa al archivo app.py (en el mismo directorio)
script_path = os.path.join(os.getcwd(), "app.py")

# Ejecutar Streamlit con la opción correcta
subprocess.run([
    "py", "-m", "streamlit", "run", script_path, "--server.runOnSave", "true"
])
