import pytest
import sys
import os
import numpy as np


# Agrega el directorio padre al sys.path 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import Registro # Importa la clase Registro que se va a probar.

# --- Clase de Pruebas para la clase Registro ---

class TestRegistro:
    """
    Clase que agrupa pruebas unitarias para la clase `Registro`.
    Cada método de esta clase representa un caso de prueba específico
    para verificar el comportamiento y la lógica de la clase `Registro`.
    """

    def test_creacion_y_atributos_basicos(self):
        """
        Prueba la creación de una instancia de `Registro` y verifica que
        todos los atributos se asignen correctamente durante la inicialización.
        """
        datos_registro = {
            "id": 1, "proyecto": "Proyecto Alfa", "area": "Diseño", "equipo": "Equipo Core",
            "costo_estimado": 1000.0, "costo_real": 1200.0,
            "avance_estimado": 100.0, "avance_real": 80.0,
            "trabajadores": 5
        }
        registro = Registro(**datos_registro)

        assert registro.id == datos_registro["id"]
        assert registro.proyecto == datos_registro["proyecto"]
        assert registro.area == datos_registro["area"]
        assert registro.equipo == datos_registro["equipo"]
        assert registro.costo_estimado == datos_registro["costo_estimado"]
        assert registro.costo_real == datos_registro["costo_real"]
        assert registro.avance_estimado == datos_registro["avance_estimado"]
        assert registro.avance_real == datos_registro["avance_real"]
        assert registro.trabajadores == datos_registro["trabajadores"]

    def test_calculo_eficiencia(self):
        """
        Prueba el método `eficiencia()` de la clase `Registro`.
        Verifica el cálculo en un escenario normal y en un caso borde
        donde el avance_estimado es cero.
        La fórmula de eficiencia es: (avance_real / avance_estimado) * 100.
        Si avance_estimado es 0.
        """
        # Caso normal: avance_estimado > 0
        registro = Registro(id=2, proyecto="P2", area="A2", equipo="E2",
                            costo_estimado=500.0, costo_real=550.0,
                            avance_estimado=100.0, avance_real=75.0,
                            trabajadores=3)
        assert registro.eficiencia() == 75.0

        # Caso borde: avance_estimado == 0
        # Asumiendo que si avance_estimado es 0 y avance_real > 0,
       
        registro_avance_est_cero = Registro(id=3, proyecto="P3", area="A3", equipo="E3",
                                            costo_estimado=100.0, costo_real=100.0,
                                            avance_estimado=0.0, avance_real=50.0, # Avance real sobre un estimado de 0
                                            trabajadores=2)
        # Se espera 100.0 basado en la lógica actual de la clase Registro:
        # if self.avance_estimado == 0: return 100.0 if self.avance_real > 0 else 0.0
        assert registro_avance_est_cero.eficiencia() == 100.0

    def test_calculo_sobrecosto(self):
        """
        Prueba el método `sobrecosto()` de la clase `Registro`.
        Verifica el cálculo tanto cuando hay un sobrecosto (costo_real > costo_estimado)
        como cuando hay un ahorro (costo_real < costo_estimado).
        La fórmula de sobrecosto es: costo_real - costo_estimado.
        """
        # Caso con sobrecosto
        registro_con_sobrecosto = Registro(id=4, proyecto="P4", area="A4", equipo="E4",
                                           costo_estimado=2000.0, costo_real=2500.0, # Sobrecosto: 500
                                           avance_estimado=100.0, avance_real=100.0,
                                           trabajadores=8)
        assert registro_con_sobrecosto.sobrecosto() == 500.0

        # Caso con ahorro (sobrecosto negativo)
        registro_con_ahorro = Registro(id=5, proyecto="P5", area="A5", equipo="E5",
                                       costo_estimado=3000.0, costo_real=2800.0, # Sobrecosto: -200 (ahorro)
                                       avance_estimado=100.0, avance_real=100.0,
                                       trabajadores=6)
        assert registro_con_ahorro.sobrecosto() == -200.0