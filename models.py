import numpy as np # Necesario para np.mean en un ejemplo

class Registro:
    def __init__(self, id, proyecto, area, equipo, costo_estimado, costo_real, avance_estimado, avance_real, trabajadores):
        self.id = id
        self.proyecto = proyecto
        self.area = area
        self.equipo = equipo
        
        # Conversión y manejo de nulos básicos en la entrada
        try:
            self.costo_estimado = float(costo_estimado) if costo_estimado is not None and not np.isnan(float(costo_estimado)) else 0.0
        except ValueError: self.costo_estimado = 0.0
        try:
            self.costo_real = float(costo_real) if costo_real is not None and not np.isnan(float(costo_real)) else 0.0
        except ValueError: self.costo_real = 0.0
        try:
            self.avance_estimado = float(avance_estimado) if avance_estimado is not None and not np.isnan(float(avance_estimado)) else 0.0
        except ValueError: self.avance_estimado = 0.0
        try:
            self.avance_real = float(avance_real) if avance_real is not None and not np.isnan(float(avance_real)) else 0.0
        except ValueError: self.avance_real = 0.0
        try:
            self.trabajadores = int(trabajadores) if trabajadores is not None and not np.isnan(float(trabajadores)) else 0
        except ValueError: self.trabajadores = 0


        if self.costo_estimado < 0 or self.costo_real < 0:
            # En una app real, esto podría registrarse en un log en lugar de levantar error si son muchos datos
            print(f"Advertencia ID {self.id}: Costos negativos detectados y ajustados a 0 o manejados. Estimado: {self.costo_estimado}, Real: {self.costo_real}")
            # Opcional: self.costo_estimado = max(0, self.costo_estimado) y similar para costo_real

    def eficiencia(self):
        if self.avance_estimado == 0:
            return 100.0 if self.avance_real == 0 else 100.0 # O un valor alto si se avanzó sin estimación
        return (self.avance_real / self.avance_estimado) * 100

    def sobrecosto(self):
        return self.costo_real - self.costo_estimado

    def alerta_sobreuso_recursos(self, umbral_sobrecosto_porcentual=20, umbral_baja_eficiencia=80):
        alertas = []
        if self.costo_estimado > 0:
            porcentaje_sobrecosto = ((self.costo_real - self.costo_estimado) / self.costo_estimado) * 100
            if porcentaje_sobrecosto > umbral_sobrecosto_porcentual:
                alertas.append(f"Sobrecosto del {porcentaje_sobrecosto:.2f}%")
        elif self.costo_real > self.costo_estimado: # Estimado 0, pero hubo costo real
             alertas.append(f"Costo Real ${self.costo_real:,.0f} vs Estimado $0")
        
        efc = self.eficiencia()
        if self.avance_estimado > 0 and efc < umbral_baja_eficiencia :
            alertas.append(f"Baja eficiencia: {efc:.2f}%")
        return alertas


class Proyecto:
    def __init__(self, nombre):
        self.nombre = nombre
        self.registros = []

    def agregar_registro(self, registro):
        if not isinstance(registro, Registro):
            raise TypeError("Solo se pueden agregar objetos de tipo Registro")
        self.registros.append(registro)

    def costo_total_estimado(self):
        return sum(r.costo_estimado for r in self.registros)

    def costo_total_real(self):
        return sum(r.costo_real for r in self.registros)

    def desviacion_presupuesto(self):
        return self.costo_total_real() - self.costo_total_estimado()

    def rendimiento_promedio(self):
        if not self.registros: return 0
        total_avance_estimado = sum(r.avance_estimado for r in self.registros)
        total_avance_real = sum(r.avance_real for r in self.registros)
        if total_avance_estimado == 0:
            return 100.0 if total_avance_real == 0 else 100.0 # O manejo especial
        return (total_avance_real / total_avance_estimado) * 100

    def obtener_alertas_proyecto(self, **kwargs):
        alertas_proyecto = {}
        for reg in self.registros:
            alertas_reg = reg.alerta_sobreuso_recursos(**kwargs)
            if alertas_reg:
                alertas_proyecto[f"Reg.ID {reg.id} (Eq:{reg.equipo}, Área:{reg.area})"] = alertas_reg
        return alertas_proyecto

class Area:
    def __init__(self, nombre):
        self.nombre = nombre
        self.registros = []

    def agregar_registro(self, registro):
        if not isinstance(registro, Registro):
            raise TypeError("Solo se pueden agregar objetos de tipo Registro")
        self.registros.append(registro)

    def eficiencia_promedio(self):
        if not self.registros: return 0
        # Usamos la misma lógica que rendimiento_promedio de Proyecto para consistencia
        total_avance_estimado = sum(r.avance_estimado for r in self.registros)
        total_avance_real = sum(r.avance_real for r in self.registros)
        if total_avance_estimado == 0:
            return 100.0 if total_avance_real == 0 else 100.0
        return (total_avance_real / total_avance_estimado) * 100

    def total_sobrecosto(self):
        return sum(r.sobrecosto() for r in self.registros)

    def obtener_alertas_area(self, **kwargs):
        alertas_area = {}
        for reg in self.registros:
            alertas_reg = reg.alerta_sobreuso_recursos(**kwargs)
            if alertas_reg:
                 alertas_area[f"Reg.ID {reg.id} (Proy:{reg.proyecto}, Eq:{reg.equipo})"] = alertas_reg
        return alertas_area