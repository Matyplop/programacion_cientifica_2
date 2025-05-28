import numpy as np 

class Registro:
    def __init__(self, id, proyecto, area, equipo, costo_estimado, costo_real, avance_estimado, avance_real, trabajadores):
  
        self.id = id
        self.proyecto = proyecto
        self.area = area
        self.equipo = equipo
        try:
            self.costo_estimado = float(costo_estimado) if costo_estimado is not None and not (isinstance(costo_estimado, float) and np.isnan(costo_estimado)) else 0.0
        except (ValueError, TypeError): self.costo_estimado = 0.0
        try:
            self.costo_real = float(costo_real) if costo_real is not None and not (isinstance(costo_real, float) and np.isnan(costo_real)) else 0.0
        except (ValueError, TypeError): self.costo_real = 0.0
        try:
            self.avance_estimado = float(avance_estimado) if avance_estimado is not None and not (isinstance(avance_estimado, float) and np.isnan(avance_estimado)) else 0.0
        except (ValueError, TypeError): self.avance_estimado = 0.0
        try:
            self.avance_real = float(avance_real) if avance_real is not None and not (isinstance(avance_real, float) and np.isnan(avance_real)) else 0.0
        except (ValueError, TypeError): self.avance_real = 0.0
        try:
            # Para trabajadores, intentamos convertir a float primero para manejar strings como "5.0" antes de int.
            self.trabajadores = int(float(trabajadores)) if trabajadores is not None and not (isinstance(trabajadores, float) and np.isnan(trabajadores)) else 0
        except (ValueError, TypeError): self.trabajadores = 0

        # Advertencia si se detectan costos negativos. Podrían ajustarse o registrarse.
        if self.costo_estimado < 0 or self.costo_real < 0:
            print(f"Advertencia ID {self.id}: Costos negativos detectados. Estimado: {self.costo_estimado}, Real: {self.costo_real}. Considerar ajuste a 0 o manejo específico.")

    def eficiencia(self) -> float:
     
        if self.avance_estimado == 0:
           
           
            return 100.0 
        return (self.avance_real / self.avance_estimado) * 100

    def sobrecosto(self) -> float:
     
        return self.costo_real - self.costo_estimado

    def alerta_sobreuso_recursos(self, umbral_sobrecosto_porcentual: float = 20.0, umbral_baja_eficiencia: float = 80.0) -> list[str]:
     
        alertas = []
        if self.costo_estimado > 0:
            porcentaje_sobrecosto = ((self.costo_real - self.costo_estimado) / self.costo_estimado) * 100
            if porcentaje_sobrecosto > umbral_sobrecosto_porcentual:
                alertas.append(f"Sobrecosto del {porcentaje_sobrecosto:.2f}%")
        elif self.costo_real > self.costo_estimado: 
            alertas.append(f"Costo Real ${self.costo_real:,.0f} vs Estimado $0 (Costo no previsto)")

        efc = self.eficiencia()
        # Solo alerta por baja eficiencia si había un avance estimado (para evitar alertas si eficiencia es 100% por avance_estimado=0)
        if self.avance_estimado > 0 and efc < umbral_baja_eficiencia:
            alertas.append(f"Baja eficiencia: {efc:.2f}%")
        return alertas


class Proyecto:
    """
    Representa un proyecto que agrupa múltiples registros, áreas y equipos.
    Permite calcular métricas agregadas a nivel de proyecto.
    """
    def __init__(self, nombre: str):
        """
        Inicializa un objeto Proyecto.

        nombre (str): El nombre del proyecto.
        """
        self.nombre = nombre
        self.registros = []  # Lista para almacenar objetos Registro asociados a este proyecto.
        self.areas = {}      # Diccionario para agrupar registros por Área dentro de este proyecto.
                             # Clave: nombre del área (str), Valor: objeto Area.
        self.equipos = {}    # Diccionario para agrupar registros por Equipo dentro de este proyecto.
                             # Clave: nombre del equipo (str), Valor: objeto Equipo.

    def agregar_registro(self, registro: Registro) -> None:
      
        if not isinstance(registro, Registro):
            raise TypeError("Solo se pueden agregar objetos de tipo Registro")
        self.registros.append(registro)

       
        if registro.area: # Asegura que el registro tenga un área definida.
            if registro.area not in self.areas:
                self.areas[registro.area] = Area(registro.area)
            self.areas[registro.area].agregar_registro(registro)

      
        if registro.equipo: # Asegura que el registro tenga un equipo definido.
            if registro.equipo not in self.equipos:
                self.equipos[registro.equipo] = Equipo(registro.equipo)
            self.equipos[registro.equipo].agregar_registro(registro)

    def costo_total_estimado(self) -> float:
      
        return sum(r.costo_estimado for r in self.registros)

    def costo_total_real(self) -> float:
     
        return sum(r.costo_real for r in self.registros)

    def desviacion_presupuesto(self) -> float:
      
        return self.costo_total_real() - self.costo_total_estimado()

    def rendimiento_promedio(self) -> float:
     
        if not self.registros: return 0.0
        total_avance_estimado = sum(r.avance_estimado for r in self.registros)
        total_avance_real = sum(r.avance_real for r in self.registros)
        if total_avance_estimado == 0:
            return 100.0 
        return (total_avance_real / total_avance_estimado) * 100

    def obtener_alertas_proyecto(self, **kwargs) -> dict:
      
        alertas_proyecto = {}
        for reg in self.registros:
            alertas_reg = reg.alerta_sobreuso_recursos(**kwargs)
            if alertas_reg:
                alertas_proyecto[f"Reg.ID {reg.id} (Eq:{reg.equipo}, Área:{reg.area})"] = alertas_reg
        return alertas_proyecto

class Area:
   
    def __init__(self, nombre: str):
        """
        Inicializa un objeto Area.


        nombre (str): El nombre del área.
        """
        self.nombre = nombre
        self.registros = [] 

    def agregar_registro(self, registro: Registro) -> None:
        """
        Agrega un objeto Registro al área.

        registro (Registro): El objeto Registro a agregar.

 
        TypeError: Si el objeto proporcionado no es una instancia de Registro.
        """
        if not isinstance(registro, Registro):
            raise TypeError("Solo se pueden agregar objetos de tipo Registro")
        self.registros.append(registro)

    def eficiencia_promedio(self) -> float:
      
        if not self.registros: return 0.0
        total_avance_estimado = sum(r.avance_estimado for r in self.registros)
        total_avance_real = sum(r.avance_real for r in self.registros)
        if total_avance_estimado == 0:
            return 100.0 
        return (total_avance_real / total_avance_estimado) * 100

    def total_sobrecosto(self) -> float:
       
        return sum(r.sobrecosto() for r in self.registros)

    def obtener_alertas_area(self, **kwargs) -> dict:
     
        alertas_area = {}
        for reg in self.registros:
            alertas_reg = reg.alerta_sobreuso_recursos(**kwargs)
            if alertas_reg:
                alertas_area[f"Reg.ID {reg.id} (Proy:{reg.proyecto}, Eq:{reg.equipo})"] = alertas_reg
        return alertas_area

class Equipo:
    """
    Representa un equipo de trabajo que puede estar asociado a múltiples registros
    a través de diferentes proyectos o áreas.
    """
    def __init__(self, nombre: str):
        """
        Inicializa un objeto Equipo.

      
        nombre (str): El nombre del equipo.
        """
        self.nombre = nombre
        self.registros = [] # Lista para almacenar objetos Registro asociados a este equipo.

    def agregar_registro(self, registro: Registro) -> None:
        """
        Agrega un objeto Registro al equipo.


        registro (Registro): El objeto Registro a agregar.
                                 No realiza una validación de tipo aquí,
        """
        self.registros.append(registro)

    def eficiencia_promedio(self) -> float:
        """
        Calcula la eficiencia promedio del equipo, promediando la eficiencia
        individual de cada uno de sus registros.

        Retorna 0 si el equipo no tiene registros.

        float: El porcentaje de eficiencia promedio del equipo.
        """
        if not self.registros: return 0.0
        eficiencias = [r.eficiencia() for r in self.registros]
        return sum(eficiencias) / len(eficiencias) if eficiencias else 0.0

    def trabajadores_totales(self) -> int:
        """
        Calcula el número total de trabajadores sumando los trabajadores
        de todos los registros asignados a este equipo.
        Solo suma si el atributo `trabajadores` del registro no es None y es numérico.

        int: El número total de trabajadores.
        """
        return sum(r.trabajadores for r in self.registros if r.trabajadores is not None and isinstance(r.trabajadores, (int, float)))

class Indicadores:
   
    @staticmethod
    def ranking_proyectos_por_sobrecosto(proyectos_list: list[Proyecto]) -> list[Proyecto]:
      
        return sorted(proyectos_list, key=lambda p: p.desviacion_presupuesto(), reverse=True)

    @staticmethod
    def eficiencia_general(registros_list: list[Registro]) -> float:
      
        # Filtra registros para incluir solo aquellos con avance_estimado > 0
        # para un cálculo de eficiencia promedio más significativo.
        eficiencias_validas = [r.eficiencia() for r in registros_list if r.avance_estimado > 0]
        return sum(eficiencias_validas) / len(eficiencias_validas) if eficiencias_validas else 0.0
