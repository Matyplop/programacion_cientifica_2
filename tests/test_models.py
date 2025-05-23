# Ejemplo: tests/test_models.py
# import unittest
# from models import Proyecto, Registro # Asumiendo que models.py está en el path o ajusta el import

# class TestProyecto(unittest.TestCase):
#     def test_calculo_costo_total(self):
#         p = Proyecto("Test")
#         r1 = Registro(id=1, proyecto="Test", area="A", equipo="E1", costo_estimado=100, costo_real=120, avance_estimado=10, avance_real=8, trabajadores=5)
#         r2 = Registro(id=2, proyecto="Test", area="A", equipo="E1", costo_estimado=200, costo_real=220, avance_estimado=20, avance_real=18, trabajadores=10)
#         p.agregar_registro(r1)
#         p.agregar_registro(r2)
#         self.assertEqual(p.costo_total_real(), 340)
#         self.assertEqual(p.costo_total_estimado(), 300)

# if __name__ == '__main__':
#     unittest.main()