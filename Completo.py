# meu primeiro agente no carla!!
# aprendi sobre o carla semana passada e resolvi tentar fazer um agente que:
# - anda sozinho
# - detecta obstaculos na frente
# - freia automaticamente
# - tenta trocar de faixa quando tem obstaculos
#
# Referencias que usei:
# https://carla.readthedocs.io/en/latest/
# varios exemplos do carla/PythonAPI/examples/
#
# pra rodar:
#   pip install carla numpy
#   python carla_agente_iniciante.py

import carla
import time
import math
import random

# ---- configuracoes basicas ----
# coloca o ip do servidor aqui, no meu caso ta rodando local
HOST = "localhost"
PORT = 2000

VELOCIDADE_ALVO = 30  # km/h, comecei com 50 mas batia muito hahaha

# distancias para o sensor de obstaculo
DISTANCIA_PERIGO   = 8.0   # metros - freia tudo
DISTANCIA_ALERTA   = 18.0  # metros - comeca a frear

# variavel global pra guardar o que o radar ta vendo
# (sei que nao é a forma mais elegante mas funciona)
distancia_obstaculo = 999.0
houve_colisao = False


# ---- conecta no carla ----
print("conectando no carla...")
cliente = carla.Client(HOST, PORT)
cliente.set_timeout(10.0)

mundo = cliente.get_world()
print("conectado! mapa:", mundo.get_map().name)


# ---- pega um carro pra spawnar ----
biblioteca = mundo.get_blueprint_library()

# tentei varios carros, gostei do tesla
carro_bp = biblioteca.find("vehicle.tesla.model3")

# pontos de spawn do mapa
pontos_spawn = mundo.get_map().get_spawn_points()
print(f"tem {len(pontos_spawn)} pontos de spawn disponiveis")

# escolhe um ponto aleatorio
ponto_inicial = random.choice(pontos_spawn)

# spawna o carro
veiculo = mundo.spawn_actor(carro_bp, ponto_inicial)
print("carro spawnado! id:", veiculo.id)

# deixa a camera do simulador seguindo o carro
# (copiei isso de um exemplo e adaptei)
spectator = mundo.get_spectator()


def atualiza_camera():
    t = veiculo.get_transform()
    # camera um pouco atras e acima do carro
    loc_camera = t.location + carla.Location(x=-7, z=4)
    spectator.set_transform(carla.Transform(
        loc_camera,
        carla.Rotation(pitch=-15, yaw=t.rotation.yaw)
    ))


# ---- sensor de radar (detecta o que ta na frente) ----
radar_bp = biblioteca.find("sensor.other.radar")

# configurei esses valores depois de testar bastante
radar_bp.set_attribute("range", "40")
radar_bp.set_attribute("horizontal_fov", "25")
radar_bp.set_attribute("vertical_fov", "10")
radar_bp.set_attribute("points_per_second", "1500")

# coloca o radar na frente do carro
transform_radar = carla.Transform(carla.Location(x=2.2, z=1.0))
radar = mundo.spawn_actor(radar_bp, transform_radar, attach_to=veiculo)


def callback_radar(dados_radar):
    global distancia_obstaculo
    leituras = []
    for deteccao in dados_radar:
        # so pega o que ta bem na frente (azimute perto de 0)
        if abs(math.degrees(deteccao.azimuth)) < 12:
            leituras.append(deteccao.depth)

    if leituras:
        distancia_obstaculo = min(leituras)
    else:
        distancia_obstaculo = 999.0


radar.listen(callback_radar)


# ---- sensor de colisao ----
colisao_bp = biblioteca.find("sensor.other.collision")
sensor_colisao = mundo.spawn_actor(
    colisao_bp,
    carla.Transform(),
    attach_to=veiculo
)


def callback_colisao(evento):
    global houve_colisao
    print(f"  !! BATEU em: {evento.other_actor.type_id}")
    houve_colisao = True


sensor_colisao.listen(callback_colisao)

print("sensores configurados!")


# ---- logica de troca de faixa ----
# essa parte foi mais dificil de fazer, tive que ler bastante sobre waypoints

# variaveis de controle da troca de faixa
trocando_faixa     = False
lado_troca         = None    # "esquerda" ou "direita"
ticks_troca        = 0       # contador de ticks desde o inicio da troca
DURACAO_TROCA      = 60      # quantos ticks dura a troca (~3 segundos)
cooldown_troca     = 0       # espera antes de tentar trocar de novo


def verifica_faixa_livre(lado):
    """
    Verifica se tem faixa disponivel no lado escolhido.
    Retorna True se tiver faixa livre, False caso contrario.
    Ainda nao checo se tem carro nessa faixa (TODO: melhorar isso)
    """
    mapa = mundo.get_map()
    wp_atual = mapa.get_waypoint(veiculo.get_location())

    if lado == "esquerda":
        wp_lado = wp_atual.get_left_lane()
    else:
        wp_lado = wp_atual.get_right_lane()

    # verifica se existe faixa e se é do mesmo tipo (pra nao ir pra calcada)
    if wp_lado is None:
        return False
    if wp_lado.lane_type != carla.LaneType.Driving:
        return False

    return True


def calcula_steering_troca(lado):
    """
    Calcula o steering para trocar de faixa.
    Usa uma curva suave: vai devagar pro lado, depois volta pro centro.
    """
    # divide a troca em 3 fases
    fase1 = DURACAO_TROCA * 0.3   # vira pro lado
    fase2 = DURACAO_TROCA * 0.7   # mantem
    # fase3 é o resto - volta ao centro

    if ticks_troca < fase1:
        # indo pro lado
        steering = 0.18 if lado == "direita" else -0.18
    elif ticks_troca < fase2:
        # pouca correcao no meio
        steering = 0.04 if lado == "direita" else -0.04
    else:
        # voltando ao centro da nova faixa
        steering = -0.1 if lado == "direita" else 0.1

    return steering


def calcula_steering_normal():
    """
    Steering normal seguindo a faixa.
    Pega o waypoint a frente e calcula o angulo necessario.
    """
    mapa = mundo.get_map()
    loc  = veiculo.get_location()
    t    = veiculo.get_transform()

    wp = mapa.get_waypoint(loc)
    proximos = wp.next(5.0)

    if not proximos:
        return 0.0

    wp_frente = proximos[0]
    alvo = wp_frente.transform.location

    dx = alvo.x - loc.x
    dy = alvo.y - loc.y

    angulo_alvo   = math.atan2(dy, dx)
    angulo_carro  = math.radians(t.rotation.yaw)

    erro = angulo_alvo - angulo_carro
    # normaliza entre -pi e pi
    while erro >  math.pi: erro -= 2 * math.pi
    while erro < -math.pi: erro += 2 * math.pi

    steer = erro * 0.35  # ganho proporcional
    # limita o steer
    steer = max(-1.0, min(1.0, steer))
    return steer


# ---- loop principal ----
print("\niniciando simulacao! ctrl+c pra parar\n")

contador = 0
tempo_inicio = time.time()

try:
    while True:
        time.sleep(0.05)   # ~20fps (nao to usando modo sincrono ainda, TODO)
        contador += 1

        # pega velocidade atual
        vel = veiculo.get_velocity()
        velocidade_ms  = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        velocidade_kmh = velocidade_ms * 3.6

        # ---- decide throttle e brake ----
        throttle = 0.0
        brake    = 0.0

        if distancia_obstaculo <= DISTANCIA_PERIGO:
            # PERIGO - freia forte
            throttle = 0.0
            brake    = 1.0
            if contador % 20 == 0:
                print(f"  [PERIGO] obstaculo a {distancia_obstaculo:.1f}m - freio total!")

        elif distancia_obstaculo <= DISTANCIA_ALERTA:
            # ALERTA - freia gradualmente
            # quanto mais perto, mais freio (calculo simples de proporcao)
            proporcao = 1.0 - (distancia_obstaculo - DISTANCIA_PERIGO) / (DISTANCIA_ALERTA - DISTANCIA_PERIGO)
            brake    = proporcao * 0.7
            throttle = 0.0
            if contador % 20 == 0:
                print(f"  [alerta] obstaculo a {distancia_obstaculo:.1f}m - freiando... (brake={brake:.2f})")

            # tenta trocar de faixa se nao ta trocando ja e cooldown zerou
            if not trocando_faixa and cooldown_troca <= 0:
                # tenta a esquerda primeiro, se nao tiver tenta direita
                if verifica_faixa_livre("esquerda"):
                    trocando_faixa = True
                    lado_troca     = "esquerda"
                    ticks_troca    = 0
                    print("  -> tentando trocar pra ESQUERDA")
                elif verifica_faixa_livre("direita"):
                    trocando_faixa = True
                    lado_troca     = "direita"
                    ticks_troca    = 0
                    print("  -> tentando trocar pra DIREITA")
                else:
                    print("  -> sem faixa disponivel, so freiar mesmo")

        else:
            # caminho livre - acelera ate a velocidade alvo
            erro_vel = VELOCIDADE_ALVO - velocidade_kmh
            if erro_vel > 2:
                throttle = min(0.6, erro_vel / VELOCIDADE_ALVO)
                brake    = 0.0
            elif erro_vel < -3:
                throttle = 0.0
                brake    = 0.15
            else:
                throttle = 0.3
                brake    = 0.0

        # ---- calcula steering ----
        if trocando_faixa:
            steer = calcula_steering_troca(lado_troca)
            ticks_troca += 1

            if ticks_troca >= DURACAO_TROCA:
                print(f"  -> troca de faixa concluida ({lado_troca})")
                trocando_faixa  = False
                lado_troca      = None
                ticks_troca     = 0
                cooldown_troca  = 80   # espera ~4s antes de trocar de novo
        else:
            steer = calcula_steering_normal()
            if cooldown_troca > 0:
                cooldown_troca -= 1

        # ---- aplica controle ----
        controle = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake),
            hand_brake=False,
            reverse=False
        )
        veiculo.apply_control(controle)

        # atualiza camera
        atualiza_camera()

        # ---- colisao: para um pouco e reseta ----
        if houve_colisao:
            veiculo.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
            time.sleep(1.5)
            houve_colisao  = False
            trocando_faixa = False
            cooldown_troca = 40

        # ---- log a cada ~2 segundos ----
        if contador % 40 == 0:
            elapsed = time.time() - tempo_inicio
            print(
                f"[{elapsed:.0f}s] "
                f"vel={velocidade_kmh:.1f}km/h  "
                f"obstaculo={distancia_obstaculo:.0f}m  "
                f"throttle={throttle:.2f}  brake={brake:.2f}  steer={steer:.3f}  "
                f"trocando={'sim ('+lado_troca+')' if trocando_faixa else 'nao'}"
            )

except KeyboardInterrupt:
    print("\nencerrando...")

finally:
    # para o carro antes de destruir
    veiculo.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
    time.sleep(0.5)

    print("destruindo sensores e veiculo...")
    radar.stop()
    radar.destroy()
    sensor_colisao.stop()
    sensor_colisao.destroy()
    veiculo.destroy()
    print("feito! ate mais")