import carla
import time
import random
import numpy as np
import networkx as nx

def dist_heuristica(n1, n2, locations):
    p1, p2 = locations[n1], locations[n2]
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def main():
    # 1. SETUP INICIAL
    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    carla_map = world.get_map()
    
    # 2. MAPEAMENTO (A*)
    print("Mapeando estradas...")
    graph = nx.DiGraph()
    node_locs = {}
    for seg in carla_map.get_topology():
        w1, w2 = seg[0], seg[1]
        node_locs[w1.id], node_locs[w2.id] = w1.transform.location, w2.transform.location
        graph.add_edge(w1.id, w2.id, weight=w1.transform.location.distance(w2.transform.location))

    # 3. SPAWN DO VEÍCULO
    bp = world.get_blueprint_library().filter("model3")[0]
    spawn_points = carla_map.get_spawn_points()
    random.shuffle(spawn_points)
    vehicle = None
    for point in spawn_points:
        vehicle = world.try_spawn_actor(bp, point)
        if vehicle is not None: break
    
    if vehicle is None: return

    # 4. CONFIGURAÇÃO DO TRAFFIC MANAGER (ESTRATÉGICO)
    tm = client.get_trafficmanager(8000)
    vehicle.set_autopilot(True, tm.get_port())
    
    tm.set_global_distance_to_leading_vehicle(1.0) # Bolha de segurança curta para permitir manobras
    tm.global_percentage_speed_difference(60.0)    # Velocidade média constante (~25-30 km/h)
    tm.auto_lane_change(vehicle, False)            # Nós controlamos as faixas

    # 5. SENSOR LIDAR COM DETECÇÃO MULTI-FAIXA
    lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '35')
    lidar_sensor = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(x=1.6, z=1.7)), attach_to=vehicle)

    def lidar_callback(data):
        points = np.frombuffer(data.raw_data, dtype=np.dtype('f4')).reshape(-1, 4)
        
        # FILTROS DE SEGURANÇA (Ignorando o chão z > -1.1)
        # 1. Obstáculo Frontal Distante (15 metros)
        front_distante = points[(points[:,0] > 1.0) & (points[:,0] < 15.0) & (np.abs(points[:,1]) < 1.2) & (points[:,2] > -1.1)]
        
        # 2. Obstáculo Frontal Crítico (7 metros)
        front_critico = points[(points[:,0] > 1.0) & (points[:,0] < 7.0) & (np.abs(points[:,1]) < 1.2) & (points[:,2] > -1.1)]

        # 3. Verificação de Faixas Laterais
        left_busy = points[(points[:,0] > -2) & (points[:,0] < 10) & (points[:,1] > 1.8) & (points[:,1] < 4.5) & (points[:,2] > -1.0)]
        right_busy = points[(points[:,0] > -2) & (points[:,0] < 10) & (points[:,1] < -1.8) & (points[:,1] > -4.5) & (points[:,2] > -1.0)]

        # LÓGICA DE DECISÃO
        if len(front_distante) > 10:
            print(f"Obstáculo detectado a {np.min(front_distante[:,0]):.1f}m. Analisando manobra...")
            
            # Prioridade 1: Tentar Esquerda
            if len(left_busy) < 3:
                print(">>> Manobra: Esquerda livre. Ultrapassando...")
                tm.force_lane_change(vehicle, True)
            
            # Prioridade 2: Tentar Direita
            elif len(right_busy) < 3:
                print(">>> Manobra: Direita livre. Ultrapassando...")
                tm.force_lane_change(vehicle, False)
            
            # Prioridade 3: Se ambas ocupadas, frear preventivamente
            else:
                print("Caminho bloqueado. Reduzindo velocidade...")
                vehicle.apply_control(carla.VehicleControl(throttle=0.1, brake=0.2))

        # FREIO DE EMERGÊNCIA (Independente da manobra)
        if len(front_critico) > 15:
            print("!!! PERIGO IMINENTE: FREANDO !!!")
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.8))

    lidar_sensor.listen(lidar_callback)

    # 6. CÂMERA E LOOP
    spectator = world.get_spectator()
    try:
        while True:
            v_trans = vehicle.get_transform()
            fwd = v_trans.get_forward_vector()
            cam_loc = v_trans.location - fwd * 12 + carla.Location(z=5)
            spectator.set_transform(carla.Transform(cam_loc, carla.Transform(cam_loc, v_trans.rotation).rotation))
            time.sleep(0.02)
    except KeyboardInterrupt: pass
    finally:
        lidar_sensor.destroy()
        vehicle.destroy()

if __name__ == '__main__':
    main()
