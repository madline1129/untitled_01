# CARLA Knowledge Base for Static Scene Setup

- **Server**: 127.0.0.1:2002
- **CARLA Version**: 0..
- **Current Map**: Carla/Maps/Town10HD_Opt
- **Generated**: auto

---
# 1. Available Maps (Towns)

| Map | Description | Available |
|---|---|---|
| `Town04` | Large rural/industrial area with a highway loop and long straight roads. | Yes |
| `Town05` | Large grid town with many intersections, crosswalks, and varied districts. | Yes |
| `Town02` | Small grid town with dead ends and curves. Slightly more complex than Town01. | Yes |
| `Town10HD` | Highly detailed urban environment with dense buildings, pedestrians, and realistic city blocks. | Yes |
| `Town04_Opt` | Optimized version of Town04. | Yes |
| `Town01` | Basic small grid town with T-junctions and straight roads. Ideal for simple scenarios. | Yes |
| `Town03` | Medium-sized urban layout with roundabouts, junctions, and mixed road types. | Yes |
| `Town06` | Long highway with multiple lanes and exit ramps. Good for high-speed scenarios. | Yes |
| `Town02_Opt` | Optimized version of Town02. | Yes |
| `Town01_Opt` | Optimized version of Town01 with fewer polygons for better performance. | Yes |
| `Town05_Opt` | Optimized version of Town05. | Yes |
| `Town07` | Rural environment with narrow roads, curves, and minimal infrastructure. | Yes |
| `Town03_Opt` | Optimized version of Town03. | Yes |
| `Town10HD_Opt` | Optimized version of Town10HD. | Yes |
| `Town11` | Large urban area with varied architecture and complex road networks. | Yes |
| `Town15` | Mixed-use area combining urban and suburban elements. | Yes |
| `Town12` | Large-scale city with multiple districts, bridges, and tunnels. | Yes |
| `Town13` | Small town with residential streets and open areas. | Yes |
| `Town06_Opt` | Optimized version of Town06. | No (may need manual load) |
| `Town07_Opt` | Optimized version of Town07. | No (may need manual load) |
| `Town_Safebench_Light` | Custom SafeBench map designed for daytime/adversarial scenarios. | No (may need manual load) |
| `Town_Safebench_Dark` | Custom SafeBench map designed for low-light/nighttime scenarios. | No (may need manual load) |

> **Tip**: Load a map with `client.load_world('TownXX')`. The `_Opt` variants have lower poly counts for faster rendering.

---
# 2. Blueprints

## 2.1 Blueprint Categories Overview

| Category | Count | Examples |
|---|---|---|
| `controller` | 1 | controller.ai.walker |
| `sensor` | 16 | sensor.other.lane_invasion, sensor.camera.depth, sensor.camera.normals |
| `static` | 97 | static.prop.haybalelb, static.prop.barbeque, static.prop.plasticchair |
| `util` | 1 | util.actor.empty |
| `vehicle` | 41 | vehicle.micro.microlino, vehicle.chevrolet.impala, vehicle.mercedes.coupe |
| `walker` | 51 | walker.pedestrian.0027, walker.pedestrian.0035, walker.pedestrian.0008 |

## 2.2 Category: `controller` (1 blueprints)

| Blueprint ID | Tags | Attributes |
|---|---|---|
| `controller.ai.walker` | walker, ai, controller | ros_name=controller.ai.walker; role_name=default |

## 2.2 Category: `sensor` (16 blueprints)

| Blueprint ID | Tags | Attributes |
|---|---|---|
| `sensor.camera.depth` | depth, camera, sensor | lens_y_size=0.08; lens_kcube=0.0; lens_circle_multiplier=0.0; lens_circle_falloff=5.0; lens_k=-1.0; fov=90.0 |
| `sensor.camera.dvs` | dvs, camera, sensor | refractory_period_ns=0; sigma_negative_threshold=0.0; sigma_positive_threshold=0.0; negative_threshold=0.3; positive_threshold=0.3; chromatic_aberration_offset=0.0 |
| `sensor.camera.instance_segmentation` | instance_segmentation, camera, sensor | lens_y_size=0.08; lens_kcube=0.0; lens_circle_multiplier=0.0; lens_circle_falloff=5.0; lens_k=-1.0; fov=90.0 |
| `sensor.camera.normals` | normals, camera, sensor | lens_y_size=0.08; lens_kcube=0.0; lens_circle_multiplier=0.0; lens_circle_falloff=5.0; lens_k=-1.0; fov=90.0 |
| `sensor.camera.optical_flow` | optical_flow, camera, sensor | lens_y_size=0.08; lens_kcube=0.0; lens_circle_multiplier=0.0; lens_circle_falloff=5.0; lens_k=-1.0; fov=90.0 |
| `sensor.camera.rgb` | rgb, camera, sensor | chromatic_aberration_intensity=0.0; tint=0.0; shoulder=0.26; toe=0.55; white_clip=0.04; slope=0.88 |
| `sensor.camera.semantic_segmentation` | semantic_segmentation, camera, sensor | lens_y_size=0.08; lens_kcube=0.0; lens_circle_multiplier=0.0; lens_circle_falloff=5.0; lens_k=-1.0; fov=90.0 |
| `sensor.lidar.ray_cast` | ray_cast, lidar, sensor | dropoff_zero_intensity=0.4; atmosphere_attenuation_rate=0.004; horizontal_fov=360.0; lower_fov=-30.0; upper_fov=10.0; rotation_frequency=10.0 |
| `sensor.lidar.ray_cast_semantic` | ray_cast_semantic, lidar, sensor | lower_fov=-30.0; upper_fov=10.0; rotation_frequency=10.0; points_per_second=56000; range=10.0; channels=32 |
| `sensor.other.collision` | collision, other, sensor | ros_name=sensor.other.collision; role_name=front, back, left, right, front_left, front_right, back_left, back_right |
| `sensor.other.gnss` | gnss, other, sensor | noise_alt_bias=0.0; noise_alt_stddev=0.0; noise_lon_bias=0.0; noise_lon_stddev=0.0; noise_lat_bias=0.0; noise_lat_stddev=0.0 |
| `sensor.other.imu` | other, imu, sensor | noise_gyro_stddev_y=0.0; noise_gyro_stddev_x=0.0; noise_gyro_stddev_z=0.0; noise_accel_stddev_y=0.0; noise_accel_stddev_x=0.0; noise_accel_stddev_z=0.0 |
| `sensor.other.lane_invasion` | lane_invasion, other, sensor | ros_name=sensor.other.lane_invasion; role_name=front, back, left, right, front_left, front_right, back_left, back_right |
| `sensor.other.obstacle` | obstacle, other, sensor | debug_linetrace=false; hit_radius=0.5; distance=5.0; sensor_tick=0.0; ros_name=sensor.other.obstacle; only_dynamics=false |
| `sensor.other.radar` | radar, other, sensor | points_per_second=1500; range=100; vertical_fov=30; sensor_tick=0.0; noise_seed=0; horizontal_fov=30 |
| `sensor.other.rss` | rss, other, sensor | ros_name=sensor.other.rss; role_name=front, back, left, right, front_left, front_right, back_left, back_right |

## 2.2 Category: `static` (97 blueprints)

| Blueprint ID | Tags | Attributes |
|---|---|---|
| `static.prop.advertisement` | advertisement, prop, static | ros_name=static.prop.advertisement; role_name=prop |
| `static.prop.atm` | prop, atm, static | ros_name=static.prop.atm; role_name=prop |
| `static.prop.barbeque` | barbeque, prop, static | ros_name=static.prop.barbeque; role_name=prop |
| `static.prop.barrel` | barrel, prop, static | ros_name=static.prop.barrel; role_name=prop |
| `static.prop.bench01` | bench01, prop, static | ros_name=static.prop.bench01; role_name=prop |
| `static.prop.bench02` | bench02, prop, static | ros_name=static.prop.bench02; role_name=prop |
| `static.prop.bench03` | bench03, prop, static | ros_name=static.prop.bench03; role_name=prop |
| `static.prop.bike helmet` | bike helmet, prop, static | ros_name=static.prop.bike helmet; role_name=prop |
| `static.prop.bin` | bin, prop, static | ros_name=static.prop.bin; role_name=prop |
| `static.prop.box01` | box01, prop, static | ros_name=static.prop.box01; role_name=prop |
| `static.prop.box02` | box02, prop, static | ros_name=static.prop.box02; role_name=prop |
| `static.prop.box03` | box03, prop, static | ros_name=static.prop.box03; role_name=prop |
| `static.prop.briefcase` | briefcase, prop, static | ros_name=static.prop.briefcase; role_name=prop |
| `static.prop.brokentile01` | brokentile01, prop, static | ros_name=static.prop.brokentile01; role_name=prop |
| `static.prop.brokentile02` | brokentile02, prop, static | ros_name=static.prop.brokentile02; role_name=prop |
| `static.prop.brokentile03` | brokentile03, prop, static | ros_name=static.prop.brokentile03; role_name=prop |
| `static.prop.brokentile04` | brokentile04, prop, static | ros_name=static.prop.brokentile04; role_name=prop |
| `static.prop.busstop` | busstop, prop, static | ros_name=static.prop.busstop; role_name=prop |
| `static.prop.busstoplb` | busstoplb, prop, static | ros_name=static.prop.busstoplb; role_name=prop |
| `static.prop.calibrator` | calibrator, prop, static | ros_name=static.prop.calibrator; role_name=prop |
| `static.prop.chainbarrier` | prop, chainbarrier, static | ros_name=static.prop.chainbarrier; role_name=prop |
| `static.prop.chainbarrierend` | chainbarrierend, prop, static | ros_name=static.prop.chainbarrierend; role_name=prop |
| `static.prop.clothcontainer` | clothcontainer, prop, static | ros_name=static.prop.clothcontainer; role_name=prop |
| `static.prop.clothesline` | clothesline, prop, static | ros_name=static.prop.clothesline; role_name=prop |
| `static.prop.colacan` | colacan, prop, static | ros_name=static.prop.colacan; role_name=prop |
| `static.prop.constructioncone` | constructioncone, prop, static | ros_name=static.prop.constructioncone; role_name=prop |
| `static.prop.container` | container, prop, static | ros_name=static.prop.container; role_name=prop |
| `static.prop.creasedbox01` | prop, creasedbox01, static | ros_name=static.prop.creasedbox01; role_name=prop |
| `static.prop.creasedbox02` | creasedbox02, prop, static | ros_name=static.prop.creasedbox02; role_name=prop |
| `static.prop.creasedbox03` | creasedbox03, prop, static | ros_name=static.prop.creasedbox03; role_name=prop |
| `static.prop.dirtdebris01` | dirtdebris01, prop, static | ros_name=static.prop.dirtdebris01; role_name=prop |
| `static.prop.dirtdebris02` | dirtdebris02, prop, static | ros_name=static.prop.dirtdebris02; role_name=prop |
| `static.prop.dirtdebris03` | dirtdebris03, prop, static | ros_name=static.prop.dirtdebris03; role_name=prop |
| `static.prop.doghouse` | doghouse, prop, static | ros_name=static.prop.doghouse; role_name=prop |
| `static.prop.foodcart` | foodcart, prop, static | ros_name=static.prop.foodcart; role_name=prop |
| `static.prop.fountain` | fountain, prop, static | ros_name=static.prop.fountain; role_name=prop |
| `static.prop.garbage01` | garbage01, prop, static | ros_name=static.prop.garbage01; role_name=prop |
| `static.prop.garbage02` | garbage02, prop, static | ros_name=static.prop.garbage02; role_name=prop |
| `static.prop.garbage03` | garbage03, prop, static | ros_name=static.prop.garbage03; role_name=prop |
| `static.prop.garbage04` | garbage04, prop, static | ros_name=static.prop.garbage04; role_name=prop |
| `static.prop.garbage05` | garbage05, prop, static | ros_name=static.prop.garbage05; role_name=prop |
| `static.prop.garbage06` | garbage06, prop, static | ros_name=static.prop.garbage06; role_name=prop |
| `static.prop.gardenlamp` | gardenlamp, prop, static | ros_name=static.prop.gardenlamp; role_name=prop |
| `static.prop.glasscontainer` | glasscontainer, prop, static | ros_name=static.prop.glasscontainer; role_name=prop |
| `static.prop.gnome` | prop, gnome, static | ros_name=static.prop.gnome; role_name=prop |
| `static.prop.guitarcase` | guitarcase, prop, static | ros_name=static.prop.guitarcase; role_name=prop |
| `static.prop.haybale` | haybale, prop, static | ros_name=static.prop.haybale; role_name=prop |
| `static.prop.haybalelb` | haybalelb, prop, static | ros_name=static.prop.haybalelb; role_name=prop |
| `static.prop.ironplank` | ironplank, prop, static | ros_name=static.prop.ironplank; role_name=prop |
| `static.prop.kiosk_01` | kiosk_01, prop, static | ros_name=static.prop.kiosk_01; role_name=prop |
| `static.prop.mailbox` | mailbox, prop, static | ros_name=static.prop.mailbox; role_name=prop |
| `static.prop.maptable` | maptable, prop, static | ros_name=static.prop.maptable; role_name=prop |
| `static.prop.mesh` | mesh, prop, static | mass=; scale=1.0f; mesh_path=; ros_name=static.prop.mesh; role_name=default |
| `static.prop.mobile` | prop, mobile, static | ros_name=static.prop.mobile; role_name=prop |
| `static.prop.motorhelmet` | prop, motorhelmet, static | ros_name=static.prop.motorhelmet; role_name=prop |
| `static.prop.pergola` | pergola, prop, static | ros_name=static.prop.pergola; role_name=prop |
| `static.prop.plantpot01` | plantpot01, prop, static | ros_name=static.prop.plantpot01; role_name=prop |
| `static.prop.plantpot02` | plantpot02, prop, static | ros_name=static.prop.plantpot02; role_name=prop |
| `static.prop.plantpot03` | plantpot03, prop, static | ros_name=static.prop.plantpot03; role_name=prop |
| `static.prop.plantpot04` | prop, plantpot04, static | ros_name=static.prop.plantpot04; role_name=prop |
| `static.prop.plantpot05` | plantpot05, prop, static | ros_name=static.prop.plantpot05; role_name=prop |
| `static.prop.plantpot06` | prop, plantpot06, static | ros_name=static.prop.plantpot06; role_name=prop |
| `static.prop.plantpot07` | plantpot07, prop, static | ros_name=static.prop.plantpot07; role_name=prop |
| `static.prop.plantpot08` | plantpot08, prop, static | ros_name=static.prop.plantpot08; role_name=prop |
| `static.prop.plasticbag` | plasticbag, prop, static | ros_name=static.prop.plasticbag; role_name=prop |
| `static.prop.plasticchair` | plasticchair, prop, static | ros_name=static.prop.plasticchair; role_name=prop |
| `static.prop.plastictable` | plastictable, prop, static | ros_name=static.prop.plastictable; role_name=prop |
| `static.prop.platformgarbage01` | platformgarbage01, prop, static | ros_name=static.prop.platformgarbage01; role_name=prop |
| `static.prop.purse` | prop, purse, static | ros_name=static.prop.purse; role_name=prop |
| `static.prop.shoppingbag` | shoppingbag, prop, static | ros_name=static.prop.shoppingbag; role_name=prop |
| `static.prop.shoppingcart` | shoppingcart, prop, static | ros_name=static.prop.shoppingcart; role_name=prop |
| `static.prop.shoppingtrolley` | shoppingtrolley, prop, static | ros_name=static.prop.shoppingtrolley; role_name=prop |
| `static.prop.slide` | slide, prop, static | ros_name=static.prop.slide; role_name=prop |
| `static.prop.streetbarrier` | streetbarrier, prop, static | ros_name=static.prop.streetbarrier; role_name=prop |
| `static.prop.streetfountain` | streetfountain, prop, static | ros_name=static.prop.streetfountain; role_name=prop |
| `static.prop.streetsign` | streetsign, prop, static | ros_name=static.prop.streetsign; role_name=prop |
| `static.prop.streetsign01` | streetsign01, prop, static | ros_name=static.prop.streetsign01; role_name=prop |
| `static.prop.streetsign04` | streetsign04, prop, static | ros_name=static.prop.streetsign04; role_name=prop |
| `static.prop.swing` | swing, prop, static | ros_name=static.prop.swing; role_name=prop |
| `static.prop.swingcouch` | swingcouch, prop, static | ros_name=static.prop.swingcouch; role_name=prop |
| `static.prop.table` | table, prop, static | ros_name=static.prop.table; role_name=prop |
| `static.prop.trafficcone01` | trafficcone01, prop, static | ros_name=static.prop.trafficcone01; role_name=prop |
| `static.prop.trafficcone02` | trafficcone02, prop, static | ros_name=static.prop.trafficcone02; role_name=prop |
| `static.prop.trafficwarning` | trafficwarning, prop, static | ros_name=static.prop.trafficwarning; role_name=prop |
| `static.prop.trampoline` | trampoline, prop, static | ros_name=static.prop.trampoline; role_name=prop |
| `static.prop.trashbag` | trashbag, prop, static | ros_name=static.prop.trashbag; role_name=prop |
| `static.prop.trashcan01` | trashcan01, prop, static | ros_name=static.prop.trashcan01; role_name=prop |
| `static.prop.trashcan02` | trashcan02, prop, static | ros_name=static.prop.trashcan02; role_name=prop |
| `static.prop.trashcan03` | trashcan03, prop, static | ros_name=static.prop.trashcan03; role_name=prop |
| `static.prop.trashcan04` | trashcan04, prop, static | ros_name=static.prop.trashcan04; role_name=prop |
| `static.prop.trashcan05` | trashcan05, prop, static | ros_name=static.prop.trashcan05; role_name=prop |
| `static.prop.travelcase` | travelcase, prop, static | ros_name=static.prop.travelcase; role_name=prop |
| `static.prop.vendingmachine` | vendingmachine, prop, static | ros_name=static.prop.vendingmachine; role_name=prop |
| `static.prop.warningaccident` | warningaccident, prop, static | ros_name=static.prop.warningaccident; role_name=prop |
| `static.prop.warningconstruction` | warningconstruction, prop, static | ros_name=static.prop.warningconstruction; role_name=prop |
| `static.prop.wateringcan` | wateringcan, prop, static | ros_name=static.prop.wateringcan; role_name=prop |
| `static.trigger.friction` | friction, trigger, static | extent_y=1.0f; extent_z=1.0f; extent_x=1.0f; friction=3.5f; ros_name=static.trigger.friction; role_name=default |

## 2.2 Category: `util` (1 blueprints)

| Blueprint ID | Tags | Attributes |
|---|---|---|
| `util.actor.empty` | empty, actor, util | ros_name=util.actor.empty; role_name=default |

## 2.2 Category: `vehicle` (41 blueprints)

| Blueprint ID | Tags | Attributes |
|---|---|---|
| `vehicle.audi.a2` | a2, audi, vehicle | terramechanics=false; sticky_control=true; color=168,0,27, 0,39,105, 16,44,21, 33,33,33; ros_name=vehicle.audi.a2; role_name=autopilot, scenario, ego_vehicle |
| `vehicle.audi.etron` | etron, audi, vehicle | terramechanics=false; sticky_control=true; color=168,0,27, 0,39,105, 33,33,33, 195,168,228; ros_name=vehicle.audi.etron; role_name=autopilot, scenario, ego_vehicle |
| `vehicle.audi.tt` | tt, audi, vehicle | terramechanics=false; sticky_control=true; color=168,0,27, 0,39,105, 16,44,21, 33,33,33; ros_name=vehicle.audi.tt; role_name=autopilot, scenario, ego_vehicle |
| `vehicle.bh.crossbike` | crossbike, bh, vehicle | terramechanics=false; sticky_control=true; driver_id=2, 3, 7, 8; color=20,78,217, 0,112,39, 255,202,0, 180,0,0; ros_name=vehicle.bh.crossbike; role_name=autopilot, scenario, ego_vehicle |
| `vehicle.bmw.grandtourer` | grandtourer, bmw, vehicle | terramechanics=false; sticky_control=true; color=109,109,109, 12,38,88, 68,68,77, 255,255,255, 255,21,0, 0,0,0; ros_name=vehicle.bmw.grandtourer; role_name=autopilot, scenario, ego_vehicle |
| `vehicle.carlamotors.carlacola` | carlacola, carlamotors, vehicle | terramechanics=false; sticky_control=true; color=255,68,0; ros_name=vehicle.carlamotors.carlacola; role_name=autopilot, scenario, ego_vehicle |
| `vehicle.carlamotors.european_hgv` | european_hgv, carlamotors, vehicle | terramechanics=false; sticky_control=true; color=231,0,0, 255,255,255, 28,28,28, 42,61,156, 127,127,127, 255,103,0; ros_name=vehicle.carlamotors.european_hgv; role_name=autopilot, scenario, ego_vehicle |
| `vehicle.carlamotors.firetruck` | carlamotors, firetruck, vehicle | terramechanics=false; sticky_control=true; color=234,0,0; ros_name=vehicle.carlamotors.firetruck; role_name=autopilot, scenario, ego_vehicle |
| `vehicle.chevrolet.impala` | impala, chevrolet, vehicle | terramechanics=false; sticky_control=true; color=61,86,143, 0,12,49, 0,0,0, 71,12,12; ros_name=vehicle.chevrolet.impala; role_name=autopilot, scenario, ego_vehicle |
| `vehicle.citroen.c3` | c3, citroen, vehicle | terramechanics=false; sticky_control=true; color=217,217,217, 168,0,27, 0,39,105, 16,44,21, 33,33,33; ros_name=vehicle.citroen.c3; role_name=autopilot, scenario, ego_vehicle |
| `vehicle.diamondback.century` | century, diamondback, vehicle | terramechanics=false; sticky_control=true; driver_id=4, 5, 6, 7; color=214,0,0, 50,96,242, 78,247,119; ros_name=vehicle.diamondback.century; role_name=autopilot, scenario, ego_vehicle |
| `vehicle.dodge.charger_2020` | charger_2020, dodge, vehicle | terramechanics=false; sticky_control=true; color=73,0,0, 0,39,105, 0,0,0, 211,142,0; ros_name=vehicle.dodge.charger_2020; role_name=autopilot, scenario, ego_vehicle |
| `vehicle.dodge.charger_police` | charger_police, dodge, vehicle | terramechanics=false; sticky_control=true; color=0,0,0, 8,53,0; ros_name=vehicle.dodge.charger_police; role_name=autopilot, scenario, ego_vehicle |
| `vehicle.dodge.charger_police_2020` | charger_police_2020, dodge, vehicle | terramechanics=false; sticky_control=true; color=0,0,0; ros_name=vehicle.dodge.charger_police_2020; role_name=autopilot, scenario, ego_vehicle |
| `vehicle.ford.ambulance` | ambulance, ford, vehicle | terramechanics=false; sticky_control=true; color=231,231,231; ros_name=vehicle.ford.ambulance; role_name=autopilot, scenario, ego_vehicle |
| `vehicle.ford.crown` | crown, ford, vehicle | terramechanics=false; sticky_control=true; color=255,185,0; ros_name=vehicle.ford.crown; role_name=autopilot, scenario, ego_vehicle |
| `vehicle.ford.mustang` | mustang, ford, vehicle | terramechanics=false; sticky_control=true; color=0,12,58, 85,0,0, 160,160,160, 0,21,0, 0,0,0; ros_name=vehicle.ford.mustang; role_name=autopilot, scenario, ego_vehicle |
| `vehicle.gazelle.omafiets` | omafiets, gazelle, vehicle | terramechanics=false; sticky_control=true; driver_id=0, 1, 3, 5; color=0,0,0, 41,73,217, 202,88,176; ros_name=vehicle.gazelle.omafiets; role_name=autopilot, scenario, ego_vehicle |
| `vehicle.harley-davidson.low_rider` | low_rider, harley-davidson, vehicle | terramechanics=false; sticky_control=true; driver_id=0, 3, 7, 4; color=67,67,67, 255,0,0, 255,154,0, 0,38,132; ros_name=vehicle.harley-davidson.low_rider; role_name=autopilot, scenario, ego_vehicle |
| `vehicle.jeep.wrangler_rubicon` | wrangler_rubicon, jeep, vehicle | terramechanics=false; sticky_control=true; color=217,217,217, 168,0,27, 0,39,105, 16,44,21, 33,33,33; ros_name=vehicle.jeep.wrangler_rubicon; role_name=autopilot, scenario, ego_vehicle |
| `vehicle.kawasaki.ninja` | ninja, kawasaki, vehicle | terramechanics=false; sticky_control=true; driver_id=0, 7, 3, 4; color=11,129,0, 255,203,39, 168,0,0, 0,0,0; ros_name=vehicle.kawasaki.ninja; role_name=autopilot, scenario, ego_vehicle |
| `vehicle.lincoln.mkz_2017` | mkz_2017, lincoln, vehicle | terramechanics=false; sticky_control=true; color=16,16,16; ros_name=vehicle.lincoln.mkz_2017; role_name=autopilot, scenario, ego_vehicle |
| `vehicle.lincoln.mkz_2020` | mkz_2020, lincoln, vehicle | terramechanics=false; sticky_control=true; driver_id=0; color=0,0,0; ros_name=vehicle.lincoln.mkz_2020; role_name=autopilot, scenario, ego_vehicle |
| `vehicle.mercedes.coupe` | coupe, mercedes, vehicle | terramechanics=false; sticky_control=true; color=217,217,217, 168,0,27, 0,39,105, 16,44,21, 33,33,33; ros_name=vehicle.mercedes.coupe; role_name=autopilot, scenario, ego_vehicle |
| `vehicle.mercedes.coupe_2020` | coupe_2020, mercedes, vehicle | terramechanics=false; sticky_control=true; color=73,0,0, 0,21,81, 0,0,0, 187,187,187; ros_name=vehicle.mercedes.coupe_2020; role_name=autopilot, scenario, ego_vehicle |
| `vehicle.mercedes.sprinter` | sprinter, mercedes, vehicle | terramechanics=false; sticky_control=true; color=167,166,175, 255,255,255, 115,115,115; ros_name=vehicle.mercedes.sprinter; role_name=autopilot, scenario, ego_vehicle |
| `vehicle.micro.microlino` | microlino, micro, vehicle | terramechanics=false; sticky_control=true; color=21,158,255, 145,255,181, 255,33,0, 255,208,0, 181,255,0; ros_name=vehicle.micro.microlino; role_name=autopilot, scenario, ego_vehicle |
| `vehicle.mini.cooper_s` | cooper_s, mini, vehicle | terramechanics=false; sticky_control=true; color=217,217,217, 168,0,27, 0,39,105, 16,44,21, 33,33,33; ros_name=vehicle.mini.cooper_s; role_name=autopilot, scenario, ego_vehicle |
| `vehicle.mini.cooper_s_2021` | mini, cooper_s_2021, vehicle | terramechanics=false; sticky_control=true; color=73,0,0, 22,31,73, 215,88,0, 0,28,0; ros_name=vehicle.mini.cooper_s_2021; role_name=autopilot, scenario, ego_vehicle |
| `vehicle.mitsubishi.fusorosa` | fusorosa, mitsubishi, vehicle | terramechanics=false; sticky_control=true; color=255,255,255; ros_name=vehicle.mitsubishi.fusorosa; role_name=autopilot, scenario, ego_vehicle |
| `vehicle.nissan.micra` | micra, nissan, vehicle | terramechanics=false; sticky_control=true; color=243,243,243, 83,83,86, 187,0,0, 28,0,46, 0,0,0, 17,21,99; ros_name=vehicle.nissan.micra; role_name=autopilot, scenario, ego_vehicle |
| `vehicle.nissan.patrol` | patrol, nissan, vehicle | terramechanics=false; sticky_control=true; color=183,187,162, 0,0,0, 66,52,33, 22,24,43, 201,147,0; ros_name=vehicle.nissan.patrol; role_name=autopilot, scenario, ego_vehicle |
| `vehicle.nissan.patrol_2021` | nissan, patrol_2021, vehicle | terramechanics=false; sticky_control=true; color=217,217,217, 159,0,0, 0,38,105, 12,42,12, 0,0,0; ros_name=vehicle.nissan.patrol_2021; role_name=autopilot, scenario, ego_vehicle |
| `vehicle.seat.leon` | leon, seat, vehicle | terramechanics=false; sticky_control=true; color=42,61,214, 21,38,98, 38,38,38, 79,33,85, 155,0,0, 0,0,0; ros_name=vehicle.seat.leon; role_name=autopilot, scenario, ego_vehicle |
| `vehicle.tesla.cybertruck` | cybertruck, tesla, vehicle | terramechanics=false; ros_name=vehicle.tesla.cybertruck; sticky_control=true; role_name=autopilot, scenario, ego_vehicle |
| `vehicle.tesla.model3` | tesla, model3, vehicle | terramechanics=false; sticky_control=true; color=17,37,103, 75,86,173, 180,42,42, 0,0,0, 137,0,0; ros_name=vehicle.tesla.model3; role_name=autopilot, scenario, ego_vehicle |
| `vehicle.toyota.prius` | prius, toyota, vehicle | terramechanics=false; sticky_control=true; color=255,0,0, 215,255,143, 96,102,99, 28,46,58, 23,51,236, 227,227,227, 0,0,0; ros_name=vehicle.toyota.prius; role_name=autopilot, scenario, ego_vehicle |
| `vehicle.vespa.zx125` | zx125, vespa, vehicle | terramechanics=false; sticky_control=true; driver_id=0, 2, 3, 6; color=255,203,39, 30,121,154, 168,0,0, 0,0,0, 208,208,208; ros_name=vehicle.vespa.zx125; role_name=autopilot, scenario, ego_vehicle |
| `vehicle.volkswagen.t2` | volkswagen, t2, vehicle | terramechanics=false; sticky_control=true; color=73,12,12, 22,31,73, 215,89,0, 105,103,0, 0,33,5, 158,40,82; ros_name=vehicle.volkswagen.t2; role_name=autopilot, scenario, ego_vehicle |
| `vehicle.volkswagen.t2_2021` | t2_2021, volkswagen, vehicle | terramechanics=false; sticky_control=true; color=73,12,12, 22,31,73, 215,89,0, 105,103,0, 0,33,5, 158,40,82; ros_name=vehicle.volkswagen.t2_2021; role_name=autopilot, scenario, ego_vehicle |
| `vehicle.yamaha.yzf` | yzf, yamaha, vehicle | terramechanics=false; sticky_control=true; driver_id=0, 2, 3, 6; color=0,0,0, 206,206,206, 0,12,49, 42,99,159, 61,68,66, 33,55,61, 255,0,0, 58,21,21, 155,0,0, 255,213,0, 127,130,135, 141,0,58; ros_name=vehicle.yamaha.yzf; role_name=autopilot, scenario, ego_vehicle |

## 2.2 Category: `walker` (51 blueprints)

| Blueprint ID | Tags | Attributes |
|---|---|---|
| `walker.pedestrian.0001` | 0001, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0001; speed=0.0, 1.8, 4.0; role_name=pedestrian |
| `walker.pedestrian.0002` | 0002, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0002; speed=0.0, 1.7, 4.0; role_name=pedestrian |
| `walker.pedestrian.0003` | 0003, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0003; speed=0.0, 1.7, 4.0; role_name=pedestrian |
| `walker.pedestrian.0004` | 0004, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0004; speed=0.0, 1.7, 4.0; role_name=pedestrian |
| `walker.pedestrian.0005` | 0005, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0005; speed=0.0, 1.8, 4.0; role_name=pedestrian |
| `walker.pedestrian.0006` | pedestrian, 0006, walker | is_invincible=true; ros_name=walker.pedestrian.0006; speed=0.0, 1.8, 4.0; role_name=pedestrian |
| `walker.pedestrian.0007` | 0007, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0007; speed=0.0, 1.8, 4.0; role_name=pedestrian |
| `walker.pedestrian.0008` | 0008, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0008; speed=0.0, 1.8, 4.0; role_name=pedestrian |
| `walker.pedestrian.0009` | 0009, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0009; speed=0.0, 1.1, 2.0; role_name=pedestrian |
| `walker.pedestrian.0010` | 0010, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0010; speed=0.0, 1.1, 2.0; role_name=pedestrian |
| `walker.pedestrian.0011` | 0011, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0011; speed=0.0, 1.1, 2.0; role_name=pedestrian |
| `walker.pedestrian.0012` | 0012, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0012; speed=0.0, 1.1, 2.0; role_name=pedestrian |
| `walker.pedestrian.0013` | 0013, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0013; speed=0.0, 1.1, 2.0; role_name=pedestrian |
| `walker.pedestrian.0014` | 0014, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0014; speed=0.0, 1.1, 2.0; role_name=pedestrian |
| `walker.pedestrian.0015` | 0015, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0015; speed=0.0, 1.8, 4.0; role_name=pedestrian |
| `walker.pedestrian.0016` | 0016, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0016; speed=0.0, 1.7, 4.0; role_name=pedestrian |
| `walker.pedestrian.0017` | 0017, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0017; speed=0.0, 1.7, 4.0; role_name=pedestrian |
| `walker.pedestrian.0018` | 0018, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0018; speed=0.0, 1.7, 4.0; role_name=pedestrian |
| `walker.pedestrian.0019` | 0019, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0019; speed=0.0, 1.7, 4.0; role_name=pedestrian |
| `walker.pedestrian.0020` | 0020, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0020; speed=0.0, 1.7, 4.0; role_name=pedestrian |
| `walker.pedestrian.0021` | 0021, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0021; speed=0.0, 1.7, 4.0; role_name=pedestrian |
| `walker.pedestrian.0022` | 0022, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0022; speed=0.0, 1.7, 4.0; role_name=pedestrian |
| `walker.pedestrian.0023` | 0023, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0023; speed=0.0, 1.7, 4.0; role_name=pedestrian |
| `walker.pedestrian.0024` | pedestrian, 0024, walker | is_invincible=true; ros_name=walker.pedestrian.0024; speed=0.0, 1.7, 4.0; role_name=pedestrian |
| `walker.pedestrian.0025` | 0025, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0025; speed=0.0, 1.7, 4.0; role_name=pedestrian |
| `walker.pedestrian.0026` | pedestrian, 0026, walker | is_invincible=true; ros_name=walker.pedestrian.0026; speed=0.0, 1.7, 4.0; role_name=pedestrian |
| `walker.pedestrian.0027` | 0027, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0027; speed=0.0, 1.7, 4.0; role_name=pedestrian |
| `walker.pedestrian.0028` | pedestrian, 0028, walker | is_invincible=true; ros_name=walker.pedestrian.0028; speed=0.0, 1.7, 4.0; role_name=pedestrian |
| `walker.pedestrian.0029` | 0029, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0029; speed=0.0, 1.7, 4.0; role_name=pedestrian |
| `walker.pedestrian.0030` | 0030, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0030; speed=0.0, 1.7, 4.0; role_name=pedestrian |
| `walker.pedestrian.0031` | 0031, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0031; speed=0.0, 1.7, 4.0; role_name=pedestrian |
| `walker.pedestrian.0032` | 0032, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0032; speed=0.0, 1.7, 4.0; role_name=pedestrian |
| `walker.pedestrian.0033` | 0033, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0033; speed=0.0, 1.7, 4.0; role_name=pedestrian |
| `walker.pedestrian.0034` | 0034, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0034; speed=0.0, 1.7, 4.0; role_name=pedestrian |
| `walker.pedestrian.0035` | 0035, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0035; speed=0.0, 1.7, 4.0; role_name=pedestrian |
| `walker.pedestrian.0036` | 0036, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0036; speed=0.0, 1.7, 4.0; role_name=pedestrian |
| `walker.pedestrian.0037` | 0037, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0037; speed=0.0, 1.7, 4.0; role_name=pedestrian |
| `walker.pedestrian.0038` | 0038, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0038; speed=0.0, 1.7, 4.0; role_name=pedestrian |
| `walker.pedestrian.0039` | 0039, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0039; speed=0.0, 1.7, 4.0; role_name=pedestrian |
| `walker.pedestrian.0040` | 0040, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0040; speed=0.0, 1.7, 4.0; role_name=pedestrian |
| `walker.pedestrian.0041` | 0041, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0041; speed=0.0, 1.7, 4.0; role_name=pedestrian |
| `walker.pedestrian.0042` | 0042, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0042; speed=0.0, 1.7, 4.0; role_name=pedestrian |
| `walker.pedestrian.0043` | 0043, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0043; speed=0.0, 1.7, 4.0; role_name=pedestrian |
| `walker.pedestrian.0044` | 0044, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0044; speed=0.0, 1.7, 4.0; role_name=pedestrian |
| `walker.pedestrian.0045` | 0045, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0045; speed=0.0, 1.7, 4.0; role_name=pedestrian |
| `walker.pedestrian.0046` | 0046, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0046; speed=0.0, 1.7, 4.0; role_name=pedestrian |
| `walker.pedestrian.0047` | pedestrian, 0047, walker | is_invincible=true; ros_name=walker.pedestrian.0047; speed=0.0, 1.7, 4.0; role_name=pedestrian |
| `walker.pedestrian.0048` | 0048, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0048; speed=0.0, 1.1, 2.0; role_name=pedestrian |
| `walker.pedestrian.0049` | 0049, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0049; speed=0.0, 1.1, 2.0; role_name=pedestrian |
| `walker.pedestrian.0050` | 0050, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0050; speed=0.0, 1.1, 2.0; role_name=pedestrian |
| `walker.pedestrian.0051` | 0051, pedestrian, walker | is_invincible=true; ros_name=walker.pedestrian.0051; speed=0.0, 0.85, 2.0; role_name=pedestrian |

---
# 3. Weather

## 3.1 Weather Presets

| Name | Constant | Cloudiness | Precipitation | Precip. Deposits | Wind | Sun Azimuth | Sun Altitude | Fog Density | Fog Distance | Wetness |
|---|---|---|---|---|---|---|---|---|---|---|
| Clear Night | `ClearNight` | 5.0 | 0.0 | 0.0 | 10.0 | -1.0 | -90.0 | 60.0 | 75.0 | 0.0 |
| Clear Noon | `ClearNoon` | 5.0 | 0.0 | 0.0 | 10.0 | -1.0 | 45.0 | 2.0 | 0.75 | 0.0 |
| Clear Sunset | `ClearSunset` | 5.0 | 0.0 | 0.0 | 10.0 | -1.0 | 15.0 | 2.0 | 0.75 | 0.0 |
| Cloudy Night | `CloudyNight` | 60.0 | 0.0 | 0.0 | 10.0 | -1.0 | -90.0 | 60.0 | 0.75 | 0.0 |
| Cloudy Noon | `CloudyNoon` | 60.0 | 0.0 | 0.0 | 10.0 | -1.0 | 45.0 | 3.0 | 0.75 | 0.0 |
| Cloudy Sunset | `CloudySunset` | 60.0 | 0.0 | 0.0 | 10.0 | -1.0 | 15.0 | 3.0 | 0.75 | 0.0 |
| Default | `Default` | -1.0 | -1.0 | -1.0 | -1.0 | -1.0 | -1.0 | -1.0 | -1.0 | -1.0 |
| Dust Storm | `DustStorm` | 100.0 | 0.0 | 0.0 | 100.0 | -1.0 | 45.0 | 2.0 | 0.75 | 0.0 |
| Hard Rain Night | `HardRainNight` | 100.0 | 100.0 | 90.0 | 100.0 | -1.0 | -90.0 | 100.0 | 0.75 | 100.0 |
| Hard Rain Noon | `HardRainNoon` | 100.0 | 100.0 | 90.0 | 100.0 | -1.0 | 45.0 | 7.0 | 0.75 | 0.0 |
| Hard Rain Sunset | `HardRainSunset` | 100.0 | 100.0 | 90.0 | 100.0 | -1.0 | 15.0 | 7.0 | 0.75 | 0.0 |
| Mid Rain Sunset | `MidRainSunset` | 60.0 | 60.0 | 60.0 | 60.0 | -1.0 | 15.0 | 3.0 | 0.75 | 0.0 |
| Mid Rainy Night | `MidRainyNight` | 80.0 | 60.0 | 60.0 | 60.0 | -1.0 | -90.0 | 60.0 | 0.75 | 80.0 |
| Mid Rainy Noon | `MidRainyNoon` | 60.0 | 60.0 | 60.0 | 60.0 | -1.0 | 45.0 | 3.0 | 0.75 | 0.0 |
| Soft Rain Night | `SoftRainNight` | 60.0 | 30.0 | 50.0 | 30.0 | -1.0 | -90.0 | 60.0 | 0.75 | 60.0 |
| Soft Rain Noon | `SoftRainNoon` | 20.0 | 30.0 | 50.0 | 30.0 | -1.0 | 45.0 | 3.0 | 0.75 | 0.0 |
| Soft Rain Sunset | `SoftRainSunset` | 20.0 | 30.0 | 50.0 | 30.0 | -1.0 | 15.0 | 2.0 | 0.75 | 0.0 |
| Wet Cloudy Night | `WetCloudyNight` | 60.0 | 0.0 | 50.0 | 10.0 | -1.0 | -90.0 | 60.0 | 0.75 | 60.0 |
| Wet Cloudy Noon | `WetCloudyNoon` | 60.0 | 0.0 | 50.0 | 10.0 | -1.0 | 45.0 | 3.0 | 0.75 | 0.0 |
| Wet Cloudy Sunset | `WetCloudySunset` | 60.0 | 0.0 | 50.0 | 10.0 | -1.0 | 15.0 | 2.0 | 0.75 | 0.0 |
| Wet Night | `WetNight` | 5.0 | 0.0 | 50.0 | 10.0 | -1.0 | -90.0 | 60.0 | 75.0 | 60.0 |
| Wet Noon | `WetNoon` | 5.0 | 0.0 | 50.0 | 10.0 | -1.0 | 45.0 | 3.0 | 0.75 | 0.0 |
| Wet Sunset | `WetSunset` | 5.0 | 0.0 | 50.0 | 10.0 | -1.0 | 15.0 | 2.0 | 0.75 | 0.0 |

## 3.2 Weather Parameter Reference

| Parameter | Range | Description |
|---|---|---|
| `cloudiness` | 0–100 | Percentage of cloud cover. 0 = clear, 100 = overcast.
| `precipitation` | 0–100 | Rain intensity. 0 = no rain, 100 = heavy rain.
| `precipitation_deposits` | 0–100 | Water puddles on the road. 0 = dry, 100 = flooded.
| `wind_intensity` | 0–100 | Wind strength affecting trees, particles, and rain direction.
| `sun_azimuth_angle` | 0–360 | Sun's horizontal orientation. Affects shadow direction.
| `sun_altitude_angle` | −90–90 | Sun's height. Negative = night/dusk, 90 = noon directly overhead.
| `fog_density` | 0–100 | Fog thickness. 0 = no fog, 100 = dense fog.
| `fog_distance` | 0–inf (m) | Distance at which fog starts. Shorter = earlier fog onset.
| `wetness` | 0–100 | Road surface wetness. Affects reflections and tire friction.

## 3.3 Common Weather Combinations for Scene Types

| Scene Type | Recipe |
|---|---|
| Clear sunny day | `WeatherParameters(sun_altitude_angle=70)` |
| Cloudy day | `WeatherParameters(cloudiness=80, sun_altitude_angle=50)` |
| Rainy day | `WeatherParameters(precipitation=60, precipitation_deposits=40, cloudiness=90, sun_altitude_angle=40)` |
| Heavy storm | `WeatherParameters(precipitation=100, precipitation_deposits=100, wind_intensity=80, cloudiness=100, sun_altitude_angle=20)` |
| Night | `WeatherParameters(sun_altitude_angle=-10)` |
| Foggy morning | `WeatherParameters(fog_density=60, fog_distance=30, sun_altitude_angle=15, cloudiness=70)` |
| Sunset | `WeatherParameters(sun_altitude_angle=5, sun_azimuth_angle=0, cloudiness=30)` |

---
# 4. Scene Setup Knowledge

## 4.1 Basic CARLA Workflow

1. **Connect**: `client = carla.Client(HOST, PORT)` + `client.set_timeout(10.0)`
2. **Load world**: `world = client.load_world('TownXX')`
3. **Set synchronous mode**:
   ```python
   settings = world.get_settings()
   settings.synchronous_mode = True
   settings.fixed_delta_seconds = 0.1
   world.apply_settings(settings)
   ```
4. **Set weather**: `world.set_weather(carla.WeatherParameters(...))`
5. **Spawn actors**: Use `world.get_blueprint_library().filter('vehicle.*')` to select blueprints, then `world.spawn_actor()` or `world.try_spawn_actor()` with a transform
6. **Tick**: In synchronous mode, call `world.tick()` to advance simulation
7. **Sensors**: Attach cameras/LIDAR/collision sensors to actors with `world.spawn_actor(sensor_bp, transform, attach_to=parent)`
8. **Cleanup**: Destroy all actors before exit: `client.apply_batch([carla.command.DestroyActor(a.id) for a in world.get_actors()])`

## 4.2 Spawn Points

The current map `Carla/Maps/Town10HD_Opt` has **155 spawn points**.
Use `carla_map.get_spawn_points()` to retrieve all spawn locations. Each is a `carla.Transform` with a `location` (x, y, z) and `rotation` (pitch, yaw, roll).

Sample spawn points:

| Index | Location (x, y, z) | Rotation (pitch, yaw, roll) |
|---|---|---|
| 0 | (-64.6, 24.5, 0.6) | (0.0, 0.2, 0.0) |
| 1 | (-67.3, 28.0, 0.6) | (0.0, 0.2, 0.0) |
| 2 | (-87.6, 13.0, 0.6) | (0.0, -179.8, 0.0) |
| 3 | (-84.9, 16.5, 0.6) | (0.0, -179.8, 0.0) |
| 4 | (-103.2, -14.4, 0.6) | (0.0, -89.4, 0.0) |
| 5 | (-106.6, -17.1, 0.6) | (0.0, -89.4, 0.0) |
| 6 | (-111.0, 59.7, 0.6) | (0.0, 90.6, 0.0) |
| 7 | (-114.4, 56.9, 0.6) | (0.0, 90.6, 0.0) |
| 8 | (-111.1, 72.9, 0.6) | (0.0, 90.6, 0.0) |
| 9 | (-114.6, 70.1, 0.6) | (0.0, 90.6, 0.0) |
| ... | ... (145 more) | ... |

## 4.3 Coordinate System

- **CARLA uses Unreal Engine coordinates**: X-forward, Y-right, Z-up
- **Location**: `carla.Location(x, y, z)` in meters
- **Rotation**: `carla.Rotation(pitch, yaw, roll)` in degrees
  - Yaw: 0 = forward (+X), 90 = right (+Y), 180 = backward (-X), 270 = left (-Y)
  - Pitch: positive = looking down, negative = looking up
  - Roll: positive = tilting right, negative = tilting left
- **Transform**: `carla.Transform(location, rotation)` combines location and rotation

## 4.4 Actor Types and Filtering

Use `world.get_blueprint_library().filter(pattern)` to find blueprints:

| Pattern | What it matches |
|---|---|
| `vehicle.*` | All vehicles (cars, trucks, bicycles, motorcycles) |
| `vehicle.audi.*` | All Audi vehicles |
| `vehicle.*.police` | Police vehicles |
| `walker.*` | All walkers/pedestrians |
| `sensor.*` | All sensors (cameras, LIDAR, etc.) |
| `static.*` | Static props (cones, barrels, etc.) |
| `static.prop.*` | All prop objects |

## 4.5 Common Sensor Blueprints

| Blueprint | Use | Key Attributes |
|---|---|---|
| `sensor.camera.rgb` | RGB camera | `image_size_x`, `image_size_y`, `fov`, `sensor_tick` |
| `sensor.camera.depth` | Depth camera | `image_size_x`, `image_size_y`, `fov` |
| `sensor.camera.semantic_segmentation` | Semantic segmentation | `image_size_x`, `image_size_y`, `fov` |
| `sensor.lidar.ray_cast` | LIDAR | `channels`, `range`, `points_per_second`, `rotation_frequency` |
| `sensor.other.collision` | Collision detection | No special attrs (read via callback) |
| `sensor.other.lane_detector` | Lane invasion | No special attrs |
| `sensor.other.gnss` | GPS | `noise_alt_stddev`, `noise_lat_stddev`, `noise_lon_stddev` |
| `sensor.other.imu` | IMU | `noise_accel_stddev_x`, `noise_gyro_stddev_x` |

## 4.6 Common Blueprint Attribute Modifications

For vehicles:
- `color`: Paint color (available colors vary by blueprint)
- `role_name`: Logical name (e.g., 'hero', 'scenario', 'ego')
- `sticky_control`: Whether the vehicle keeps applying last control (default True in older versions)

For cameras:
- `image_size_x`, `image_size_y`: Resolution (e.g., 800x600)
- `fov`: Field of view in degrees (default 90)
- `sensor_tick`: Capture interval in seconds (0 = every tick)

## 4.7 Destroying Actors

```python
for actor in world.get_actors():
    if actor.type_id.startswith('vehicle') or actor.type_id.startswith('walker'):
        actor.destroy()
```
Or use batch commands for efficiency:
```python
client.apply_batch([carla.command.DestroyActor(a.id) for a in world.get_actors()])
```

## 4.8 Traffic Manager

- Create: `tm = client.get_trafficmanager(TM_PORT)` (typically PORT + 6000)
- Set auto-pilot: `vehicle.set_autopilot(True, tm_port)`
- Key TM parameters: `tm.global_percentage_speed_difference()`, `tm.set_global_distance_to_leading_vehicle()`

## 4.9 Waypoints and Road Information

- Get map: `carla_map = world.get_map()`
- Generate waypoints: `carla_map.generate_waypoints(distance)` — returns `carla.Waypoint` objects every `distance` meters along drivable lanes
- Get closest waypoint: `carla_map.get_waypoint(location)`
- Waypoint properties: `.transform`, `.lane_id`, `.road_id`, `.lane_type`, `.is_junction`, `.lane_width`, `.right_lane_marking()`, `.left_lane_marking()`
- Topology: `carla_map.get_topology()` returns a list of (waypoint, waypoint) tuples representing road segments

## 4.10 Loading OpenDRIVE Maps

```python
with open('path/to/map.xodr', 'r') as f:
    world = client.generate_opendrive_world(f.read())
```

## 4.11 Static Scene Setup Checklist

To set up a static scene from a text description:

1. **Select map** that matches the scenario (urban → Town03/05/10HD, highway → Town04/06, rural → Town07)
2. **Set weather** matching described conditions (sunny, rainy, foggy, night)
3. **Set ego vehicle** at a spawn point or custom location
4. **Spawn adversarial vehicles** at selected positions using filtered blueprints
5. **Spawn walkers** if pedestrians are needed
6. **Attach sensors** (RGB camera, collision, LIDAR) to ego vehicle
7. **Set synchronous mode** for deterministic simulation
8. **Tick** to advance frame-by-frame
