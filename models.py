import numpy as np 

class Registro:
    """
    Representa un registro individual de seguimiento de un proyecto,
    conteniendo información sobre costos, avances y recursos.
    """
    def __init__(self, id, proyecto, area, equipo, costo_estimado, costo_real, avance_estimado, avance_real, trabajadores):
        """
        Inicializa un objeto Registro.

        id (any): Identificador único del registro.
        proyecto (str): Nombre del proyecto al que pertenece el registro.
        area (str): Área funcional o departamento asociado al registro.
        equipo (str): Equipo específico responsable del trabajo en el registro.
        costo_estimado (float/str/None): Costo estimado para el trabajo del registro.
                                            Se convierte a float; los nulos o inválidos se tratan como 0.0.
        costo_real (float/str/None): Costo real incurrido para el trabajo del registro.
                                        Se convierte a float; los nulos o inválidos se tratan como 0.0.
        avance_estimado (float/str/None): Porcentaje de avance estimado para el registro.
                                            Se convierte a float; los nulos o inválidos se tratan como 0.0.
        avance_real (float/str/None): Porcentaje de avance real completado para el registro.
                                        Se convierte a float; los nulos o inválidos se tratan como 0.0.
        trabajadores (int/str/None): Número de trabajadores asignados al registro.
                                        Se convierte a int; los nulos o inválidos se tratan como 0.
        """
        self.id = id
        self.proyecto = proyecto
        self.area = area
        self.equipo = equipo

        # Conversión y manejo de nulos/errores en la entrada para atributos numéricos.
        # Si la conversión falla o el valor es None/NaN, se asigna un valor por defecto (0.0 o 0).
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
        """
        Calcula la eficiencia del registro como el porcentaje del avance real
        respecto al avance estimado.

        Si el avance estimado es 0:
        - Retorna 100.0 si el avance real también es 0 (se cumplió la estimación de no avanzar).
        - Retorna 100.0 (o un valor que indique "infinito" o "no calculable" si se prefiere)
          si hubo avance real sobre una estimación de 0. La lógica actual devuelve 100.0.

        float: El porcentaje de eficiencia.
        """
        if self.avance_estimado == 0:
           
           
            return 100.0 
        return (self.avance_real / self.avance_estimado) * 100

    def sobrecosto(self) -> float:
        """
        Calcula el sobrecosto del registro como la diferencia entre
        el costo real y el costo estimado.
        Un valor negativo indica un ahorro.

        
        float: El monto del sobrecosto (positivo) o ahorro (negativo).
        """
        return self.costo_real - self.costo_estimado

    def alerta_sobreuso_recursos(self, umbral_sobrecosto_porcentual: float = 20.0, umbral_baja_eficiencia: float = 80.0) -> list[str]:
        """
        Genera alertas si el registro excede umbrales de sobrecosto o
        presenta baja eficiencia.

       
        umbral_sobrecosto_porcentual: El porcentaje de sobrecosto
                (respecto al costo estimado) a partir del cual se genera una alerta.
                Defaults to 20.0.
        umbral_baja_eficiencia: El porcentaje de eficiencia por debajo
                del cual se genera una alerta. Defaults to 80.0.

       
        list[str]: Una lista de mensajes de alerta. Vacía si no hay alertas.
        """
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
        """
        Agrega un objeto Registro al proyecto. También lo asigna al área y equipo
        correspondientes dentro de la estructura interna del proyecto.

        registro (Registro): El objeto Registro a agregar.
       
        TypeError: Si el objeto proporcionado no es una instancia de Registro.
        """
        if not isinstance(registro, Registro):
            raise TypeError("Solo se pueden agregar objetos de tipo Registro")
        self.registros.append(registro)

        # Agrega el registro al objeto Area correspondiente dentro del proyecto.
        # Si el Area no existe en self.areas, se crea una nueva instancia.
        if registro.area: # Asegura que el registro tenga un área definida.
            if registro.area not in self.areas:
                self.areas[registro.area] = Area(registro.area)
            self.areas[registro.area].agregar_registro(registro)

        # Agrega el registro al objeto Equipo correspondiente dentro del proyecto.
        # Si el Equipo no existe en self.equipos, se crea una nueva instancia.
        if registro.equipo: # Asegura que el registro tenga un equipo definido.
            if registro.equipo not in self.equipos:
                self.equipos[registro.equipo] = Equipo(registro.equipo)
            self.equipos[registro.equipo].agregar_registro(registro)

    def costo_total_estimado(self) -> float:
        """
        Calcula el costo total estimado del proyecto sumando los costos
        estimados de todos sus registros.

        float: El costo total estimado.
        
        """
        return sum(r.costo_estimado for r in self.registros)

    def costo_total_real(self) -> float:
        """
        Calcula el costo total real del proyecto sumando los costos
        reales de todos sus registros.

        float: El costo total real.
        """
        return sum(r.costo_real for r in self.registros)

    def desviacion_presupuesto(self) -> float:
        """
        Calcula la desviación del presupuesto del proyecto (costo total real - costo total estimado).
        Un valor positivo indica un sobrecosto, uno negativo un ahorro.

     
        float: La desviación del presupuesto.
        """
        return self.costo_total_real() - self.costo_total_estimado()

    def rendimiento_promedio(self) -> float:
        """
        Calcula el rendimiento promedio del proyecto.
        Se define como (avance real total / avance estimado total) * 100.

        Si el avance estimado total es 0:
        - Retorna 100.0 si el avance real total también es 0.
        - Retorna 100.0 si hubo avance real total sobre una estimación total de 0.
        Retorna 0 si no hay registros.

        float: El porcentaje de rendimiento promedio.
        """
        if not self.registros: return 0.0
        total_avance_estimado = sum(r.avance_estimado for r in self.registros)
        total_avance_real = sum(r.avance_real for r in self.registros)
        if total_avance_estimado == 0:
            return 100.0 
        return (total_avance_real / total_avance_estimado) * 100

    def obtener_alertas_proyecto(self, **kwargs) -> dict:
        """
        Recopila las alertas de sobreuso de recursos de todos los registros
        asociados a este proyecto.

      
        **kwargs: Argumentos opcionales que se pasarán a la función
                      `alerta_sobreuso_recursos` de cada Registro


        dict: Un diccionario donde las claves son identificadores de registros
                   
        """
        alertas_proyecto = {}
        for reg in self.registros:
            alertas_reg = reg.alerta_sobreuso_recursos(**kwargs)
            if alertas_reg:
                alertas_proyecto[f"Reg.ID {reg.id} (Eq:{reg.equipo}, Área:{reg.area})"] = alertas_reg
        return alertas_proyecto

class Area:
    """
    Representa un área funcional o departamento dentro de un proyecto o a nivel global.
    Agrupa registros pertenecientes a esta área.
    """
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
        """
        Calcula la eficiencia promedio de todos los registros en esta área.
        Utiliza la misma lógica que `Proyecto.rendimiento_promedio` para consistencia:
        (avance real total del área / avance estimado total del área) * 100.

        Si el avance estimado total del área es 0:
        - Retorna 100.0 si el avance real total del área también es 0.
        - Retorna 100.0 si hubo avance real total sobre una estimación total de 0.
        Retorna 0 si no hay registros en el área.


        float: El porcentaje de eficiencia promedio del área.
        """
        if not self.registros: return 0.0
        total_avance_estimado = sum(r.avance_estimado for r in self.registros)
        total_avance_real = sum(r.avance_real for r in self.registros)
        if total_avance_estimado == 0:
            return 100.0 
        return (total_avance_real / total_avance_estimado) * 100

    def total_sobrecosto(self) -> float:
        """
        Calcula el sobrecosto total acumulado de todos los registros en esta área.

   
        float: El sobrecosto total del área.
        """
        return sum(r.sobrecosto() for r in self.registros)

    def obtener_alertas_area(self, **kwargs) -> dict:
        """
        Recopila las alertas de sobreuso de recursos de todos los registros
        asociados a esta área.

    
        **kwargs: Argumentos opcionales que se pasarán a la función
                      `alerta_sobreuso_recursos` de cada Registro

    
        dict: Un diccionario donde las claves son identificadores de registros
                  
        """
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
    """
    Clase estática que proporciona métodos para calcular indicadores globales
    o rankings a partir de listas de objetos Proyecto o Registro.
    """
    @staticmethod
    def ranking_proyectos_por_sobrecosto(proyectos_list: list[Proyecto]) -> list[Proyecto]:
        """
        Clasifica una lista de objetos Proyecto por su desviación de presupuesto (sobrecosto),
        de mayor sobrecosto a mayor ahorro (menor sobrecosto).

        proyectos_list (list[Proyecto]): Una lista de objetos Proyecto.


        list[Proyecto]: La lista de proyectos ordenada por sobrecosto descendente.
        """
        return sorted(proyectos_list, key=lambda p: p.desviacion_presupuesto(), reverse=True)

    @staticmethod
    def eficiencia_general(registros_list: list[Registro]) -> float:
        """
        Calcula la eficiencia general promediando la eficiencia de una lista de registros.
        Solo considera registros donde el avance estimado es mayor que cero para evitar
        distorsiones por la lógica de `Registro.eficiencia()` cuando `avance_estimado` es 0.

        registros_list (list[Registro]): Una lista de objetos Registro.

        float: El porcentaje de eficiencia general. Retorna 0 si no hay registros
                   válidos (con avance_estimado > 0) para calcularla.
        """
        # Filtra registros para incluir solo aquellos con avance_estimado > 0
        # para un cálculo de eficiencia promedio más significativo.
        eficiencias_validas = [r.eficiencia() for r in registros_list if r.avance_estimado > 0]
        return sum(eficiencias_validas) / len(eficiencias_validas) if eficiencias_validas else 0.0
