import carla
import time
import random
import numpy as np

def main():
    # 1. CONEXÃO E SETUP
    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    carla_map = world.get_map()
    blueprint_library = world.get_blueprint_library()

    # 2. SPAWN ALEATÓRIO (PROTEÇÃO CONTRA OS 90 CARROS)
    bp = blueprint_library.filter("model3")[0]
    spawn_points = carla_map.get_spawn_points()
    random.shuffle(spawn_points)

    vehicle = None
    for point in spawn_points:
        vehicle = world.try_spawn_actor(bp, point)
        if vehicle is not None:
            print(f"Sucesso! Veículo gerado em: {point.location}")
            break

    if vehicle is None:
        print("Erro: Nenhum ponto de spawn livre encontrado.")
        return

    # 3. CONFIGURAÇÃO DO TRAFFIC MANAGER (TM) - MODO AGRESSIVO
    tm = client.get_trafficmanager(8000)
    vehicle.set_autopilot(True, tm.get_port())

    # Reduzimos a distância para 0.5m para forçar o TM a aceitar manobras próximas
    tm.set_global_distance_to_leading_vehicle(0.5)
    tm.global_percentage_speed_difference(50.0) # ~30km/h para apresentação
    tm.auto_lane_change(vehicle, False) 
    
    # Ignorar uma pequena % de colisão ajuda o TM a ter "coragem" de mudar de faixa
    tm.ignore_vehicles_percentage(vehicle, 5)

    # 4. SENSOR LIDAR (FILTROS DE ALTO DESEMPENHO)
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '30')
    lidar_bp.set_attribute('rotation_frequency', '25')
    lidar_sensor = world.spawn_actor(
        lidar_bp,
        carla.Transform(carla.Location(x=1.6, z=1.7)),
        attach_to=vehicle
    )

    def lidar_callback(data):
        points = np.frombuffer(data.raw_data, dtype=np.dtype('f4')).reshape(-1, 4)

        # FILTRO FRONTAL: Foca apenas em objetos na largura da faixa e acima do chão
        front_obs = points[
            (points[:, 0] > 1.5) & (points[:, 0] < 12) &
            (np.abs(points[:, 1]) < 1.2) &
            (points[:, 2] > -1.1)
        ]

        # FILTRO ESQUERDA (CALIBRADO): Expandimos a área para detectar o carro vizinho por inteiro
        left_lane_busy = points[
            (points[:, 0] > -5) & (points[:, 0] < 15) & 
            (points[:, 1] > 1.5) & (points[:, 1] < 4.8) &
            (points[:, 2] > -1.0) # Ignora o asfalto rigorosamente
        ]

        # FILTRO DIREITA (CALIBRADO)
        right_lane_busy = points[
            (points[:, 0] > -5) & (points[:, 0] < 15) &
            (points[:, 1] > -4.8) & (points[:, 1] < -1.5) &
            (points[:, 2] > -1.0)
        ]

        if len(front_obs) > 20:
            # DEBUG NO TERMINAL para você ver o que o sensor vê em tempo real
            print(f"Obstáculo à frente! Analisando: Esq={len(left_lane_busy)} pts | Dir={len(right_lane_busy)} pts")

            # Prioridade 1: ESQUERDA
            if len(left_lane_busy) < 3:
                print(">>> EXECUTANDO MANOBRA PARA ESQUERDA <<<")
                tm.force_lane_change(vehicle, True)
                # Ajuda o carro a não travar por falta de torque
                vehicle.apply_control(carla.VehicleControl(throttle=0.3))

            # Prioridade 2: DIREITA
            elif len(right_lane_busy) < 3:
                print(">>> EXECUTANDO MANOBRA PARA DIREITA <<<")
                tm.force_lane_change(vehicle, False)
                vehicle.apply_control(carla.VehicleControl(throttle=0.3))

            else:
                print("Caminho bloqueado em ambas as faixas. Reduzindo velocidade.")
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.5))

    lidar_sensor.listen(lidar_callback)

    # 5. LOOP DE CÂMERA (CHASE CAM)
    spectator = world.get_spectator()

    try:
        print("--- SISTEMA DE NAVEGAÇÃO ATIVO ---")
        while True:
            v_trans = vehicle.get_transform()
            fwd = v_trans.get_forward_vector()

            # Câmera 12m atrás e 5m acima para uma visão clara da ultrapassagem
            cam_loc = v_trans.location - fwd * 12 + carla.Location(z=5)
            cam_rot = v_trans.rotation
            cam_rot.pitch = -20

            spectator.set_transform(carla.Transform(cam_loc, cam_rot))
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nSimulação encerrada.")
    finally:
        print("Limpando atores do mundo...")
        if 'lidar_sensor' in locals(): lidar_sensor.destroy()
        if 'vehicle' in locals(): vehicle.destroy()
        print("Finalizado com sucesso.")

if __name__ == '__main__':
    main()
