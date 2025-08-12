import subprocess
import sys

def install_metatrader5():
    try:
        # Instalar usando el pip del Python actual
        subprocess.check_call([sys.executable, "-m", "pip", "install", "MetaTrader5"])
        print("MetaTrader5 instalado correctamente")
        
        # Verificar instalación
        import MetaTrader5 as mt5
        print(f"Versión instalada: {mt5.__version__}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error en la instalación: {e}")
    except ImportError:
        print("Error: No se pudo importar MetaTrader5 después de la instalación")

# Ejecutar instalación
install_metatrader5()