import traci

# Inicializa la simulación con TraCI
sumo_binary = "sumo-gui"  # Usa "sumo" para la simulación sin interfaz gráfica
sumo_cmd = ["sumo-gui", "-c", "simulacion.sumocfg"]  # Asegúrate de que el archivo .sumocfg esté en la misma carpeta o proporciona la ruta
traci.start(sumo_cmd)

# Función para modificar la duración de los semáforos dinámicamente
def modify_traffic_lights():
    light_id = "J1"  # ID del semáforo a modificar, ajusta esto a tus necesidades
    logic = traci.trafficlight.getAllProgramLogics(light_id)[0]  # Obtén la lógica actual del semáforo

    # Crear una nueva lógica de semáforo con fases modificadas
    new_phases = []
    for phase in logic.phases:
        if "G" in phase.state:  # Si la fase está en verde, aumenta su duración
            new_phases.append(traci.trafficlight.Phase(phase.duration + 5, phase.state))  # Sumar 5 segundos a la duración
        elif "r" in phase.state:  # Si la fase está en rojo, reduce su duración
            new_phases.append(traci.trafficlight.Phase(max(phase.duration - 2, 1), phase.state))  # Restar 2 segundos pero nunca menos de 1
        else:
            new_phases.append(phase)  # Mantener las fases no verdes ni rojas igual

    # Crear una nueva lógica de semáforo con las fases modificadas
    new_logic = traci.trafficlight.Logic(
        logic.programID, logic.type, 0, new_phases
    )
    traci.trafficlight.setProgramLogic(light_id, new_logic)  # Asigna la nueva lógica al semáforo

# Ejecuta la simulación paso a paso
step = 0
while step < 3600:  # Por ejemplo, simular por 1 hora (3600 segundos)
    traci.simulationStep()  # Avanza un paso en la simulación
    if step % 100 == 0:  # Modificar semáforos cada 100 pasos
        modify_traffic_lights()  # Modificar la lógica de los semáforos
    step += 1

# Finaliza la simulación
traci.close()
