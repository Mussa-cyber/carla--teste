import carla
import time
import random
import numpy as np
import cv2

def main():
    # 1. SETUP DO MUNDO
    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    carla_map = world.get_map()
    
    # 2. SPAWN DO VEÍCULO
    bp = world.get_blueprint_library().filter("model3")[0]
    spawn_points = carla_map.get_spawn_points()
    random.shuffle(spawn_points)
    vehicle = None
    for point in spawn_points:
        vehicle = world.try_spawn_actor(bp, point)
        if vehicle is not None: break
    if vehicle is None: return

    # 3. CONFIGURAÇÃO DO TRAFFIC MANAGER (Ajustado para Visão)
    tm = client.get_trafficmanager(8000)
    vehicle.set_autopilot(True, tm.get_port())
    tm.set_global_distance_to_leading_vehicle(1.5)
    tm.global_percentage_speed_difference(50.0) 
    tm.auto_lane_change(vehicle, False)

    # 4. SENSOR DE SEGMENTAÇÃO SEMÂNTICA
    sem_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
    sem_bp.set_attribute('image_size_x', '640')
    sem_bp.set_attribute('image_size_y', '480')
    sem_bp.set_attribute('fov', '110')
    sem_cam = world.spawn_actor(sem_bp, carla.Transform(carla.Location(x=1.6, z=1.7)), attach_to=vehicle)

    def camera_callback(image):
        # --- CORREÇÃO AQUI: CityScapesPalette ---
        image.convert(carla.ColorConverter.CityScapesPalette)
        
        # Converter para array NumPy (BGRA -> BGR)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3] 

        # LÓGICA DE DETECÇÃO (ROI - Região de Interesse)
        # Analisamos o centro da imagem para detectar carros (Cor Azul no CityScapes)
        # Veículos no CityScapesPalette são [0, 0, 142] em RGB (ou 142, 0, 0 em BGR)
        roi = array[300:460, 200:440]
        
        # Criar máscara para a cor azul (Veículos)
        lower_blue = np.array([140, 0, 0])
        upper_blue = np.array([145, 10, 10])
        mask = cv2.inRange(roi, lower_blue, upper_blue)
        vehicle_pixels = cv2.countNonZero(mask)

        # Tomada de decisão baseada na visão
        if vehicle_pixels > 400:
            print(f"Visão detectou obstáculo: {vehicle_pixels}px. Mudando de faixa...")
            tm.force_lane_change(vehicle, True) # Tenta esquerda
            
            # Frenagem de emergência se o objeto ocupar muito da visão
            if vehicle_pixels > 2000:
                vehicle.apply_control(carla.VehicleControl(brake=0.5))

        # Desenhar um retângulo na tela do OpenCV para mostrar o que a IA está "olhando"
        display_img = array.copy()
        cv2.rectangle(display_img, (200, 300), (440, 460), (0, 255, 0), 2)
        
        cv2.imshow('Visao Computacional - CityScapes', display_img)
        cv2.waitKey(1)

    sem_cam.listen(camera_callback)

    # 5. LOOP DO SPECTATOR
    spectator = world.get_spectator()
    try:
        print("Sistema de Visão Ativo. Pressione Ctrl+C para encerrar.")
        while True:
            v_trans = vehicle.get_transform()
            fwd = v_trans.get_forward_vector()
            cam_loc = v_trans.location - fwd * 12 + carla.Location(z=5)
            spectator.set_transform(carla.Transform(cam_loc, v_trans.rotation))
            time.sleep(0.01)
    except KeyboardInterrupt: pass
    finally:
        print("Limpando...")
        cv2.destroyAllWindows()
        sem_cam.destroy()
        vehicle.destroy()

if __name__ == '__main__':
    main()
