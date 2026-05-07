import carla
import time
import random
import numpy as np

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    carla_map = world.get_map()

    # SPAWN
    bp = world.get_blueprint_library().filter("model3")[0]
    spawn_points = carla_map.get_spawn_points()
    random.shuffle(spawn_points)

    vehicle = None
    for point in spawn_points:
        vehicle = world.try_spawn_actor(bp, point)
        if vehicle:
            break

    if vehicle is None:
        print("Erro ao spawnar veículo")
        return

    # TRAFFIC MANAGER
    tm = client.get_trafficmanager(8000)
    vehicle.set_autopilot(True, tm.get_port())

    tm.set_global_distance_to_leading_vehicle(1.5)
    tm.global_percentage_speed_difference(50.0)
    tm.auto_lane_change(vehicle, False)

    # ESTADO INTERNO
    state = {
        "is_changing_lane": False,
        "lane_change_start": 0,
        "lane_change_duration": 3.0,
        "last_decision_time": 0,
        "decision_cooldown": 1.5
    }

    # LIDAR
    lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('range', '35')

    lidar_sensor = world.spawn_actor(
        lidar_bp,
        carla.Transform(carla.Location(x=1.6, z=1.7)),
        attach_to=vehicle
    )

    def compute_brake(distance):
        if distance < 4:
            return 1.0
        elif distance < 6:
            return 0.7
        elif distance < 10:
            return 0.4
        elif distance < 15:
            return 0.2
        return 0.0

    def lidar_callback(data):
        points = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
        now = time.time()

        # REGIÕES
        front = points[
            (points[:,0] > 1.0) & (points[:,0] < 15.0) &
            (np.abs(points[:,1]) < 1.3) &
            (points[:,2] > -1.1)
        ]

        left = points[
            (points[:,0] > -2) & (points[:,0] < 10) &
            (points[:,1] > 1.8) & (points[:,1] < 4.5) &
            (points[:,2] > -1.0)
        ]

        right = points[
            (points[:,0] > -2) & (points[:,0] < 10) &
            (points[:,1] < -1.8) & (points[:,1] > -4.5) &
            (points[:,2] > -1.0)
        ]

        min_dist = None
        if len(front) > 0:
            min_dist = float(np.min(front[:,0]))

        if state["is_changing_lane"]:
            if now - state["lane_change_start"] > state["lane_change_duration"]:
                print("✔ Mudança de faixa concluída")
                state["is_changing_lane"] = False
            return 

        # 2. EMERGÊNCIA
        if min_dist is not None and min_dist < 5:
            brake = compute_brake(min_dist)
            print(f"!!! Emergência: {min_dist:.1f}m | brake={brake}")
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=brake))
            return

        # 3. DECISÃO COM COOLDOWN
        if now - state["last_decision_time"] < state["decision_cooldown"]:
            return

        state["last_decision_time"] = now

        # 4. OBSTÁCULO À FRENTE
        if min_dist is not None and min_dist < 15:
            print(f"Obstáculo a {min_dist:.1f}m")

            left_free = len(left) < 5
            right_free = len(right) < 5

            # PRIORIDADE: ESQUERDA
            if left_free:
                print(">>> Ultrapassagem pela esquerda")
                tm.force_lane_change(vehicle, True)
                state["is_changing_lane"] = True
                state["lane_change_start"] = now

            elif right_free:
                print(">>> Ultrapassagem pela direita")
                tm.force_lane_change(vehicle, False)
                state["is_changing_lane"] = True
                state["lane_change_start"] = now

            else:
                # Travagem
                brake = compute_brake(min_dist)
                print(f"Sem espaço. Travando suavemente ({brake})")
                vehicle.apply_control(
                    carla.VehicleControl(throttle=0.2, brake=brake)
                )

    lidar_sensor.listen(lidar_callback)

    # CÂMERA
    spectator = world.get_spectator()

    try:
        while True:
            v_trans = vehicle.get_transform()
            fwd = v_trans.get_forward_vector()

            cam_loc = v_trans.location - fwd * 12 + carla.Location(z=5)
            spectator.set_transform(carla.Transform(cam_loc, v_trans.rotation))

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("Interrompido")

    finally:
        lidar_sensor.destroy()
        vehicle.destroy()


if __name__ == "__main__":
    main()
